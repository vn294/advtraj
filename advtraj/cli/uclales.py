"""
Interface for producing trajectories from UCLA-LES model output

Model version with advective tracer trajectories implemented:
https://github.com/leifdenby/uclales/tree/advective-trajectories
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr

# from .cstruct_selection import trajectory_cstruct_ref
from .. import integrate_trajectories
from ..utils.cli import optional_debugging
from ..utils.grid import find_coord_grid_spacing

# from ..utils.object_tools import get_bounding_boxes
# from ..utils.object_tools import get_bounding_boxes
# from ..utils.object_tools import get_field_avg
from ..utils.interpolation import interpolate_3d_fields
from ..utils.object_tools import get_object_labels, unsplit_objects

# from ..utils.matching_objects import matching_object_list
# from ..family.traj_family import find_matching_objects_ref
# from ..family.traj_family import find_family_matching_objects
# from ..family.traj_family import print_matching_objects
# from ..family.traj_family import graph_matching_objects
# from ..family.traj_family import print_matching_objects_graph
# from ..family.traj_family import draw_object_graph

var_properties = {
    "u": [True, False, False],
    "v": [False, True, False],
    "w": [False, False, True],
    "t": [False, False, False],
    "p": [False, False, False],
    "q": [False, False, False],
    "l": [False, False, False],
    "tracer_rad1": [False, False, False],
    "tracer_rad2": [False, False, False],
    "tracer_rad3": [False, False, False],
}


def center_staggered_field(phi_da):
    """
    Create cell-centered values for staggered (velocity) fields
    """
    dim = [d for d in phi_da.dims if d.endswith("m")][0]
    newdim = dim.replace("m", "t")
    s_left, s_right = slice(0, -1), slice(1, None)

    # average vertical velocity to cell centers
    coord_vals = 0.5 * (
        phi_da[dim].isel(**{dim: s_left}).values
        + phi_da[dim].isel(**{dim: s_right}).values
    )
    coord = xr.DataArray(
        coord_vals, coords={newdim: coord_vals}, attrs=dict(units="m"), dims=(newdim,)
    )

    # create new coordinates for cell-centered vertical velocity
    coords = OrderedDict(phi_da.coords)
    del coords[dim]
    coords[newdim] = coord

    phi_cc_vals = 0.5 * (
        phi_da.isel(**{dim: s_left}).values + phi_da.isel(**{dim: s_right}).values
    )

    dims = list(phi_da.dims)
    dims[dims.index(dim)] = newdim

    phi_cc = xr.DataArray(
        phi_cc_vals,
        coords=coords,
        dims=dims,
        attrs=dict(units=phi_da.units, long_name=phi_da.long_name),
    )

    phi_cc.name = phi_da.name
    return phi_cc


def load_data(files, fields_to_keep=["l"]):
    tracer_fields = ["atrc_xr", "atrc_xi", "atrc_yr", "atrc_yi", "atrc_zr"]

    def preprocess(ds):
        return ds[fields_to_keep + tracer_fields]

    def sortkey(p):
        idx_string = p.name.split(".")[-2]
        i, j = int(idx_string[:4]), int(idx_string[4:])
        return j, i

    # files = sorted(files, key=sortkey)

    ds = xr.open_mfdataset(files, parallel=True)  # , preprocess=preprocess)
    # ds = xr.open_dataset(files,engine='netcdf4')#, preprocess=preprocess)
    ds.u.attrs["long_name"] = ds.u.longname
    ds.v.attrs["long_name"] = ds.v.longname
    ds.w.attrs["long_name"] = ds.w.longname
    ds.l.attrs["long_name"] = ds.l.longname
    ds.p.attrs["long_name"] = ds.p.longname
    ds.t.attrs["long_name"] = ds.t.longname
    ds.q.attrs["long_name"] = ds.q.longname

    # for vel_field in ["u", "v", "w"]:
    #    if vel_field in fields_to_keep:
    #        ds[vel_field] = center_staggered_field(ds[vel_field])

    ds = ds.rename(dict(xt="x", yt="y", zt="z"))

    for v in tracer_fields:
        ds = ds.rename({v: "traj_tracer_{}".format(v.split("_")[-1])})

    # simulations with UCLA-LES always have periodic boundary conditions
    ds.attrs["xy_periodic"] = True

    # add the grid-spacing and domain extent as attributes
    # dx, dy, dz and Lx, Ly, Lz respectively to speed up calculations
    # For velocity fields defined at cell face centers, the coordinates used
    # are xm,ym and zm. Add grid spacings as attributes to these as well.
    for c in "xyz":
        ds[c].attrs[f"d{c}"] = find_coord_grid_spacing(
            da_coord=ds[c], show_warnings=False
        )
        if c in "xy":
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values) + ds[c].attrs[f"d{c}"]
        else:
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values)

    for vel_field in ["u", "v", "w"]:
        if vel_field in fields_to_keep:
            if vel_field == "w":
                ds["zm"].attrs["dz"] = find_coord_grid_spacing(
                    da_coord=ds["zm"], show_warnings=False
                )
            elif vel_field == "v":
                ds["ym"].attrs["dy"] = find_coord_grid_spacing(
                    da_coord=ds["ym"], show_warnings=False
                )
            elif vel_field == "u":
                ds["xm"].attrs["dx"] = find_coord_grid_spacing(
                    da_coord=ds["xm"], show_warnings=False
                )

    return ds


def main(data_path, file_prefix, output_path):
    thresh = 1.0e-6
    fields_to_keep = ["l", "q", "u", "w", "p", "t", "v"]
    # variable_list=None
    interp_order = 5
    # unsplit = False

    files = list(Path(data_path).glob(f"{file_prefix}.*.nc"))

    ds = load_data(files=files, fields_to_keep=fields_to_keep)

    tracers = [
        "traj_tracer_xi",
        "traj_tracer_xr",
        "traj_tracer_yi",
        "traj_tracer_yr",
        "traj_tracer_zr",
    ]
    # t_ref = ds.time[int(ds.time.count()) // 2]
    first = True
    # ds_family_list = []
    # max_at_ref = []
    for i in range(18, 40):
        # as an example take the timestep half-way through the available data and
        # from 300m altitude and up
        # ds_subset = ds.isel(time=int(ds.time.count()) // 2)#.sel(z=slice(300, None))
        ref_time = ds.time[i]
        print("ref_time", ref_time.values)
        ds_subset = ds.isel(time=i)
        # ds_fields_subset = ds_subset[fields_to_keep]

        # we'll use as starting points for the trajectories all points where the
        # liquid water mixing ratio is greater than a threshold

        # From PC email 05/07/22
        # mask_w = ds_subset.w > 0.0
        # ds_sub = ds_subset.where(mask_w)
        # mask = (ds_subset.l > thresh)
        # ds_subset = ds_sub.where(mask)

        mask = ds_subset.l > thresh
        mask.name = "cloud_mask"
        ds_poi = (
            mask.where(mask, drop=True)
            .stack(trajectory_number=("x", "y", "z"))
            .dropna(dim="trajectory_number")
        )

        # now we'll turn this 1D dataset where (x, y, z) are coordinates into one
        # where they are variables instead
        ds_starting_points = (
            ds_poi.reset_index("trajectory_number")
            .assign_coords(
                trajectory_number=np.arange(ds_poi.trajectory_number.count())
            )
            .reset_coords(["x", "y", "z"])[["x", "y", "z"]]
        )

        # Set up matching error DataArrays
        for c in "xyz":
            ds_starting_points[f"{c}_err"] = xr.zeros_like(ds_starting_points[c])

        # Find contiguous objects and return a set of numeric labels matching
        # the ‘trajectory_number’ in ds_starting_points.
        # Note this takes account of objects that straddle the periodic x and y
        # boundaries.
        olab = get_object_labels(mask).drop("time")
        print("No of objects", olab.nobjects)

        # Field values at the initial trajectory positions # Replace with interpolation call
        # print('Starting interpolation')
        ds_starting_fields = interpolate_3d_fields(
            ds=ds_subset[fields_to_keep],  # Fields at current time (t_ref)
            ds_positions=ds_starting_points,
            interpolator=None,
            interp_order=interp_order,
            cyclic_boundaries="xy" if ds_subset[fields_to_keep].xy_periodic else None,
        )
        # print('Finishing interpolation')
        ds_position_scalars = ds[tracers]  # Only the Lagrangian label tracers
        ds_fields = ds[
            fields_to_keep
        ]  # To be used to interpolate and determine field values
        ds_traj = integrate_trajectories(
            ds_position_scalars=ds_position_scalars,
            ds_fields=ds_fields,
            ds_starting_fields=ds_starting_fields,
            ds_starting_points=ds_starting_points,
            interp_order=interp_order,
        )
        # Add the object labels as a non-dimensional coordinate.
        ds_traj = ds_traj.assign_coords({"object_label": olab})
        # Objects may straddle the periodic boundaries at some times. This is an
        # issue for code that needs spatial coherence, e.g. measuring the horizontal size
        # of an object.
        # This function attempts to gather points together, with the result that some
        # will be outside the domain.
        # Only use this if you need it!
        Lx = ds.x.Lx
        Ly = ds.y.Ly
        # if unsplit:
        ds_traj = unsplit_objects(ds_traj, Lx, Ly)

        # Create an object mask. This will be used to identify trajectories
        # (back and forward from a ref time) of interest within the set and create
        # a bounding box
        # mask = (ds_traj.l > thresh)
        # mask.name = 'object_mask'

        # ds_traj = ds_traj.assign(object_mask = ds_traj.l > thresh)
        # ds_bounds = get_bounding_boxes(ds_traj,use_mask=False)

        # Find fully developed clouds (LWC is maximum) if any, at the reference time
        # ds_field_mean = get_field_avg(ds_traj,use_mask=False)
        # maxobjvar = (ds_field_mean.LWC == ds_field_mean.LWC.max())
        # max_at_ref_time = []
        # for obj in maxobjvar.object_label:
        #    max_objvar = (ds_field_mean.LWC[obj] == ds_field_mean.LWC[obj].max())
        #    time_max_objvar = ds_field_mean.time.values[(max_objvar.values)]
        #    if (time_max_objvar.size > 0):
        #        if (time_max_objvar == ref_time.values):
        #           #max_at_ref[ref_time] = obj.object_label.values
        #            max_at_ref_time.append(obj.object_label.values.item())
        # max_at_ref.append(max_at_ref_time)
        # print('max at reference time ', ref_time.values,'is', max_at_ref)
        if first:
            ds_traj_final = ds_traj.copy()
            #    ds_bounds_final = ds_bounds.copy()
            first = False
        else:
            ds_traj_final = xr.concat([ds_traj_final, ds_traj], dim="ref_time")
        #    ds_bounds_final = xr.concat([ds_bounds_final, ds_bounds],dim="ref_time")

        # Add any attributes you want.
        # attrs = {'interp_order':5,
        #         'solver':minim,
        # }
        # attrs['maxiter'] = options['pioptions']['maxiter']
        # attrs['tol'] = options['pioptions']['tol']
        # attrs['minimize_maxiter'] =
        # options['pioptions']['minoptions']['minimize_options']['maxiter']
        # ds_traj.attrs = attrs
        # ds_bounds.attrs["Lx"] = Lx
        # ds_bounds.attrs["Ly"] = Ly
        # ds_bounds.attrs["dx"] = ds["x"].attrs["dx"]
        # ds_bounds.attrs["dy"] = ds["y"].attrs["dy"]
        ds_traj.attrs["Lx"] = Lx
        ds_traj.attrs["Ly"] = Ly
        ds_traj.attrs["dx"] = ds["x"].attrs["dx"]
        ds_traj.attrs["dy"] = ds["y"].attrs["dy"]
        # ds_family_list.append(tuple((ds_traj,ds_bounds)))

    output_path = output_path.format(file_prefix=file_prefix)
    ds_traj_final.to_netcdf(output_path)

    # mol = matching_object_list(ds_out_final,master_ref=i-2,select_object=[0])
    # mol = find_matching_objects_ref(ds_family_list,
    #                                master_ref_time=None,
    #                                select=[0],
    #                                ref_time_only = True,
    #                                forward=True,
    #                                adjacent_only=True,
    #                                fast=True,
    #                                use_numpy=False)
    # print('ds_family_list attrs',ds_family_list.attrs)
    # mol_family = find_family_matching_objects(ds_family_list,
    #                                          select = max_at_ref,
    #                                          ref_time_only = True,
    #                                          forward = True,
    #                                          adjacent_only = False,
    #                                          fast = True,
    #                                          use_numpy=False)

    # print('mol',mol_family)
    # print_matching_objects(mol_family)
    # print(f"Trajectories saved to {output_path}")
    # matching_obj_to_nodelist(mol_family)
    # G = graph_matching_objects(mol_family)
    # print_matching_objects_graph(G)
    # draw_object_graph(G,save_file='fig_1hr.pdf')


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=Path)
    argparser.add_argument("file_prefix", type=Path)
    argparser.add_argument("--debug", default=False, action="store_true")
    argparser.add_argument(
        "--output", type=str, default="{file_prefix}.trajectories.nc"
    )
    args = argparser.parse_args()

    with optional_debugging(args.debug):
        main(
            data_path=args.data_path,
            file_prefix=args.file_prefix,
            output_path=args.output,
        )


# def trajectory_init(
#    dataset,
#    time_index,
#    variable_list,
#    deltax,
#    deltay,
#    deltaz,
#    refprof,
#    traj_pos,
#    interp_method="tri_lin",
#    interp_order=1,
# ):
#    """
#    Set up origin of back and forward trajectories.
#    Args:
#        dataset       : Netcdf file handle.
#        time_index    : Index of required time in file.
#        variable_list : List of variable names.
#        refprof       : Dict with reference theta profile arrays.
#        traj_pos      : array[n,3] of initial 3D positions.
#    Returns
#    -------
#        Trajectory variables::
#            trajectory     : position of origin point.
#            data_val       : associated data so far.
#            traj_error     : estimated trajectory errors so far.
#            traj_times     : trajectory times so far.
#            coords             : Dict containing model grid info.
#                'xcoord': grid xcoordinate of model space.
#                'ycoord': grid ycoordinate of model space.
#                'zcoord': grid zcoordinate of model space.
#                'deltax': Model x grid spacing in m.
#                'deltay': Model y grid spacing in m.
#                'z'     : Model z grid m.
#                'zn'    : Model z grid m.
#    @author: Peter Clark
#    """
#    th = dataset.variables["th"][0, ...]
#    (nx, ny, nz) = np.shape(th)
#
#    xcoord = np.arange(nx, dtype="float")
#    ycoord = np.arange(ny, dtype="float")
#    zcoord = np.arange(nz, dtype="float")

#    z, zn = get_z_zn(dataset, deltaz * zcoord)
#
#   coords = {
#        "xcoord": xcoord,
#        "ycoord": ycoord,
#        "zcoord": zcoord,
#        "deltax": deltax,
#        "deltay": deltay,
#        "z": z,
#        "zn": zn,
#    }

# data_list, varlist, varp_list, time = load_traj_step_data(dataset,
#                                                time_index, variable_list,
#                                                refprof, coords )
# print("Starting at time {}".format(time))

#    out = data_to_pos(
#        data_list,
#        varp_list,
#        traj_pos,
#        xcoord,
#        ycoord,
#        zcoord,
#        interp_method=interp_method,
#        interp_order=interp_order,
#    )
#    traj_pos_new, n_pvar = extract_pos(nx, ny, out)

#    data_val = list([np.vstack(out[n_pvar:]).T])

#    if debug:

#        print(np.shape(data_val))
#        print(np.shape(traj_pos))
#        print("xorg", traj_pos[:, 0])
#        print("yorg", traj_pos[:, 1])
#        print("zorg", traj_pos[:, 2])
#        print("x", traj_pos_new[:, 0])
#        print("y", traj_pos_new[:, 1])
#        print("z", traj_pos_new[:, 2])

#    trajectory = list([traj_pos])
#    traj_error = list([np.zeros_like(traj_pos)])
#    trajectory.insert(0, traj_pos_new)
#    traj_error.insert(0, np.zeros_like(traj_pos_new))
#    traj_times = list([time])
#
#    return trajectory, data_val, traj_error, traj_times, coords


# def extract_pos(nx, ny, dat):
#    """
#    Extract 3D position from data array.
#    Args:
#        nx        : Number of points in x direction.
#        ny        : Number of points in y direction.
#        dat       : Array[m,n] where n>=5 if cyclic_xy or 3 if not.
#    Returns
#    -------
#        pos       : Array[m,3]
#        n_pvar    : Number of dimensions in input data used for pos.
#    """
#    global cyclic_xy
#    if cyclic_xy:
#        n_pvar = 5
#        xpos = phase(dat[0], dat[1], nx)
#        ypos = phase(dat[2], dat[3], ny)
#        pos = np.array([xpos, ypos, dat[4]]).T
#    else:
#        n_pvar = 3
#        pos = np.array([dat[0], dat[1], dat[2]]).T#
#
#   return pos, n_pvar


# def data_to_pos(
#    data,
#    varp_list,
#    pos,
#    xcoord,
#    ycoord,
#    zcoord,
#    interp_method="tri_lin",
#    interp_order=1,
#    maxindex=None,
# ):
#    """
#    Interpolate data to pos.
#    Args:
#        data      : list of data array.
#        varp_list: list of grid info lists.
#        pos       : array[n,3] of n 3D positions.
#        xcoord,ycoord,zcoord: 1D arrays giving coordinate spaces of data.
#    Returns
#    -------
#        list of arrays containing interpolated data.
#    @author: Peter Clark
#    """
#    if interp_method == "tri_lin":
#        if interp_order != 1:
#            raise ValueError(
#                f"Variable interp_order must be set to 1 for tri_lin_interp; "
#                f"currently {interp_order}."
#            )
#        output = tri_lin_interp(
#            data, varp_list, pos, xcoord, ycoord, zcoord, maxindex=maxindex
#        )
#    elif interp_method == "grid_interp":
#        output = multi_dim_lagrange_interp(
#            data, pos, order=interp_order, wrap=[True, True, False]
#        )
#    elif interp_method == "fast_interp":
#        output = fast_interp_3D(data, pos, order=interp_order, wrap=[True, True, False])
#    else:
#        output = list([])
#        for l in range(len(data)):
#                        print 'Calling map_coordinates'
#                        print np.shape(data[l]), np.shape(traj_pos)
#            out = ndimage.map_coordinates(data[l], pos, mode="wrap", order=interp_order)
#            output.append(out)
#    return output


def compute_traj_boxes(traj, in_obj_func, kwargs={}):
    """
    Compute two rectangular boxes containing all and in_obj points.
    For each trajectory object and time, plus some associated data.
    Args:
        traj               : trajectory object.
        in_obj_func        : function to determine which points are inside an object.
        kwargs             : any additional keyword arguments to ref_func (dict).
    Returns
    -------
        Properties of in_obj boxes::
            data_mean        : mean of points (for each data variable) in in_obj.
            in_obj_data_mean : mean of points (for each data variable) in in_obj.
            num_in_obj       : number of in_obj points.
            traj_centroid    : centroid of in_obj box.
            in_obj__centroid : centroid of in_obj box.
            traj_box         : box coordinates of each trajectory set.
            in_obj_box       : box coordinates for in_obj points in each trajectory set.
    @author: Peter Clark
    """
    scalar_shape = (np.shape(traj.data)[0], traj.nobjects)
    centroid_shape = (np.shape(traj.data)[0], traj.nobjects, 3)
    mean_obj_shape = (np.shape(traj.data)[0], traj.nobjects, np.shape(traj.data)[2])
    box_shape = (np.shape(traj.data)[0], traj.nobjects, 2, 3)

    data_mean = np.zeros(mean_obj_shape)
    in_obj_data_mean = np.zeros(mean_obj_shape)
    objvar_mean = np.zeros(scalar_shape)
    num_in_obj = np.zeros(scalar_shape, dtype=int)
    traj_centroid = np.zeros(centroid_shape)
    in_obj_centroid = np.zeros(centroid_shape)
    traj_box = np.zeros(box_shape)
    in_obj_box = np.zeros(box_shape)

    in_obj_mask, objvar = in_obj_func(traj, **kwargs)

    for iobj in range(traj.nobjects):

        data = traj.data[:, traj.labels == iobj, :]
        data_mean[:, iobj, :] = np.mean(data, axis=1)
        obj = traj.trajectory[:, traj.labels == iobj, :]

        traj_centroid[:, iobj, :] = np.mean(obj, axis=1)
        traj_box[:, iobj, 0, :] = np.amin(obj, axis=1)
        traj_box[:, iobj, 1, :] = np.amax(obj, axis=1)

        objdat = objvar[:, traj.labels == iobj]

        for it in np.arange(0, np.shape(obj)[0]):
            mask = in_obj_mask[it, traj.labels == iobj]
            num_in_obj[it, iobj] = np.size(np.where(mask))
            if num_in_obj[it, iobj] > 0:
                in_obj_data_mean[it, iobj, :] = np.mean(data[it, mask, :], axis=0)
                objvar_mean[it, iobj] = np.mean(objdat[it, mask])
                in_obj_centroid[it, iobj, :] = np.mean(obj[it, mask, :], axis=0)
                in_obj_box[it, iobj, 0, :] = np.amin(obj[it, mask, :], axis=0)
                in_obj_box[it, iobj, 1, :] = np.amax(obj[it, mask, :], axis=0)
    return (
        data_mean,
        in_obj_data_mean,
        objvar_mean,
        num_in_obj,
        traj_centroid,
        in_obj_centroid,
        traj_box,
        in_obj_box,
    )


# vn294 implementation of labels from ReadingClouds/trajectories
# Find initial positions and labels using user-defined function.
# labels,nobjects = trajectory_cstruct_ref(ds,int(ds.time.count()) //2,
#                                                  thresh,find_objects=True)
# trajectory_number = np.arange(ds_poi.trajectory_number.count())
# traj_labels = xr.DataArray(labels,coords={"trajectory_number": trajectory_number})
# times = ref_times
# trajectory, data_val, traj_error, traj_times, coords \
#    = trajectory_init(dataset, time_index, variable_list,
#                      deltax, deltay, deltaz, refprof, traj_pos,
#                      interp_method = interp_method,
#                      interp_order = interp_order,
#    )
# ref_index = 0

# ds_starting = ds_starting_points.assign(label=traj_labels)
