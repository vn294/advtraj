"""
Functionality for computing trajectories backward from a set of starting points
at a single point in time using the position scalars.
"""
import numpy as np
import xarray as xr
from tqdm import tqdm

from ..utils.grid_mapping import (
    estimate_3d_position_from_grid_indecies,
    estimate_initial_grid_indecies,
)
from ..utils.interpolation import interpolate_3d_fields


def calc_trajectory_previous_position(
    ds_position_scalars,
    ds_traj_posn,
    interp_order=5,
    interpolator=None,
):
    """
    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` interpolate the
    "position scalars" to find their value at `(x,y,z,t)`
    2) estimate the initial indecies that the fluid at `(x,y,z,t)` came from by
    converting the "position scalars" back to position
    """
    # interpolate the position scalar values at the current trajectory
    # position

    ds_initial_position_scalar_locs = interpolate_3d_fields(
        ds=ds_position_scalars,
        ds_positions=ds_traj_posn,
        interpolator=interpolator,
        interp_order=interp_order,
        cyclic_boundaries="xy" if ds_position_scalars.xy_periodic else None,
    )

    # convert these position scalar values to grid positions so we can estimate
    # what grid positions the fluid was advected from
    nx, ny = int(ds_position_scalars.x.count()), int(ds_position_scalars.y.count())
    ds_traj_init_grid_idxs = estimate_initial_grid_indecies(
        ds_position_scalars=ds_initial_position_scalar_locs, N_grid=dict(x=nx, y=ny)
    )

    # interpolate these grid-positions from the position scalars so that we can
    # get an actual xyz-position
    ds_traj_posn_prev = estimate_3d_position_from_grid_indecies(
        ds_grid=ds_position_scalars,
        i=ds_traj_init_grid_idxs.i,
        j=ds_traj_init_grid_idxs.j,
        k=ds_traj_init_grid_idxs.k,
    )

    return ds_traj_posn_prev


def backward(
    ds_position_scalars,
    ds_fields,
    ds_starting_fields,
    ds_starting_point,
    da_times,
    interp_order=5,
):
    """
    Using the position scalars `ds_position_scalars` integrate backwards from
    `ds_starting_point` to the times in `da_times`
    """
    # create a list into which we will accumulate the trajectory points
    # while doing this we turn the time variable into a coordinate
    datasets = [ds_starting_point]
    datasets_fields = [ds_starting_fields]
    # datasets_merged = [ds_starting_point, ds_starting_fields]

    # step back in time, `t_current` represents the time we're of the next
    # point (backwards) of the trajectory
    # start at 1 because this provides position for previous time.
    for t_current in tqdm(da_times.values[1:][::-1], desc="backward"):

        ds_traj_posn_origin = datasets[-1].drop_vars("time")

        ds_position_scalars_current = ds_position_scalars.sel(time=t_current).drop_vars(
            "time"
        )

        ds_traj_posn_est = calc_trajectory_previous_position(
            ds_position_scalars=ds_position_scalars_current,
            ds_traj_posn=ds_traj_posn_origin,
            interp_order=interp_order,
        )
        # find the previous time so that we can construct a new dataset to contain
        # the trajectory position at the previous time
        try:
            t_previous = ds_position_scalars.time.sel(time=slice(None, t_current)).isel(
                time=-2
            )
        except IndexError:
            # this will happen if we're trying to integrate backwards from the
            # very first timestep, which we can't (and shouldn't). Just check
            # we have as many trajectory points as we're aiming for
            if len(datasets) == da_times.count():
                break
            else:
                raise

        ds_traj_posn_prev = ds_traj_posn_est.rename(
            {"x_est": "x", "y_est": "y", "z_est": "z"}
        ).assign_coords({"time": t_previous.values})

        # Error in back trajectory is not quantifiable. Set to NaN.
        for c in "xyz":
            ds_traj_posn_prev[f"{c}_err"] = xr.full_like(ds_traj_posn_prev[c], np.NaN)
        # vn294
        # ds_fields_current = ds_fields.sel(time=t_current).drop_vars("time")
        ds_fields_prev = ds_fields.sel(time=t_previous).drop_vars("time")
        # print('Backward time',t_previous)
        ds_fields_locs = interpolate_3d_fields(
            ds=ds_fields_prev,
            ds_positions=ds_traj_posn_prev,
            interpolator=None,
            interp_order=interp_order,
            cyclic_boundaries="xy" if ds_fields_prev.xy_periodic else None,
        )
        # print('l min in fields',ds_fields_prev.l.min(skipna=True))
        # print('l min',ds_fields_prev.l.values[20:30,120:125,200:205])
        # print('l min index',ds_fields_locs.l.idxmin())
        # print('traj at min',ds_traj_posn_prev.x.values[49512],ds_traj_posn_prev.y.values[49512],ds_traj_posn_prev.z.values[49512])
        # print('Backward l min',ds_fields_locs.l.min())
        ds_fields_locs_prev = ds_fields_locs.assign_coords({"time": t_previous.values})
        # - ---------------------------------------------------------------

        datasets.append(ds_traj_posn_prev)
        datasets_fields.append(ds_fields_locs_prev)  # vn294

    # ds_traj = xr.concat(datasets[::-1], dim="time")
    ds_traj = xr.concat(datasets[::-1], dim="time", coords="minimal")  # vn294
    ds_fields_along_traj = xr.concat(datasets_fields[::-1], dim="time")  # vn294
    # ds_final_output = ds_traj.merge(
    #    ds_fields_along_traj, combine_attrs="drop_conflicts"
    # )  # vn294
    return ds_traj, ds_fields_along_traj
