import xarray as xr
import numpy as np
import pandas as pd

def _simulate_battery_numpy(
    power_np,           # shape: (..., time)
    cons_np,            # shape: (..., time) OR broadcastable
    battery_cap_np,     # shape: (...)
    *,
    initial_charge,
    max_supplied,
    supplied_efficiency,
    battery_loss,
):
    STATE_SHAPE = power_np.shape[:-1]
    T = power_np.shape[-1]

    independent_hours = np.zeros((*STATE_SHAPE, T), dtype=bool)
    total_excess = np.zeros(STATE_SHAPE, dtype=np.float64)
    total_cons = np.zeros(STATE_SHAPE, dtype=np.float64)
    battery_charge = np.full(STATE_SHAPE, initial_charge, dtype=np.float64)
    battery_charge = np.minimum(battery_charge, battery_cap_np)

    for t in range(T):
        gen = power_np[..., t].copy()
        cons = np.broadcast_to(cons_np[..., t], STATE_SHAPE).copy()

        can_supplied = np.maximum(0.0, (battery_cap_np - battery_charge) / supplied_efficiency)
        gen_to_batt = np.minimum(np.minimum(gen, can_supplied), max_supplied)
        gen -= gen_to_batt
        battery_charge += gen_to_batt * supplied_efficiency

        need = cons / supplied_efficiency
        gen_to_cons = np.minimum(np.minimum(gen, need), max_supplied)
        gen -= gen_to_cons
        cons -= gen_to_cons * supplied_efficiency
        total_cons += gen_to_cons * supplied_efficiency

        need = cons / supplied_efficiency
        batt_to_cons = np.minimum(np.minimum(battery_charge, need), max_supplied)
        battery_charge -= batt_to_cons
        cons -= batt_to_cons * supplied_efficiency
        total_cons += batt_to_cons * supplied_efficiency

        gen_excess = np.maximum(0, gen)
        total_excess += gen_excess * supplied_efficiency

        independent_hours[..., t] = cons <= 1e-5

        battery_charge *= (1 - battery_loss)

    return independent_hours, total_excess, total_cons


def simulate_battery_system_xr(
    power_data: xr.DataArray,
    consumption_curve: xr.DataArray,
    battery_capacity_arr: xr.DataArray,
    **params,
):
    ORDER = ("power_nominal", "city", "height", "battery_cap", "time")

    hourly_consumption = (
        consumption_curve
        .sel(time=power_data.time.dt.hour)
        .assign_coords(time=power_data.time)
    )

    battery_capacity_arr = (
        battery_capacity_arr
        .expand_dims(power_data.dims[:-1])
        .assign_coords(
            battery_cap=("battery_cap", battery_capacity_arr.data)
        )
        .transpose(*ORDER[:-1])
    )

    power_data = (
        power_data
        .expand_dims(battery_cap=battery_capacity_arr.battery_cap)
        .transpose(*ORDER)
    )

    hourly_consumption = (
        hourly_consumption
        .expand_dims(power_data.dims[:-1])
        .transpose(*ORDER)
    )

    indep, excess, cons = _simulate_battery_numpy(
        power_data.data,
        hourly_consumption.data,
        battery_capacity_arr.data,
        **params,
    )

    coords_state = {
        "power_nominal": power_data.power_nominal,
        "city": power_data.city,
        "height": power_data.height,
        "battery_cap": battery_capacity_arr.battery_cap,
    }

    return xr.Dataset(
        {
            "independent_hours": xr.DataArray(
                indep,
                dims=(*coords_state.keys(), "time"),
                coords={**coords_state, "time": power_data.time},
            ),
            "total_excess": xr.DataArray(
                excess,
                dims=coords_state.keys(),
                coords=coords_state,
            ),
            "total_consumption": xr.DataArray(
                cons,
                dims=coords_state.keys(),
                coords=coords_state,
            ),
        }
    )


def simulate_battery_system(
        power_data: pd.DataFrame,     # shape = (n_hours,1)
        consumption_curve: pd.Series, # shape = (24,)
        battery_capacity_arr: list,   # shape = (n_battery_cap,)

        initial_charge=0,
        max_supplied=2500,
        supplied_efficiency=0.95,
        battery_loss=0.005,
    ):    

    hourly_consumption = consumption_curve.loc[power_data.index.hour]
    hourly_consumption.index = power_data.index

    n_hours = len(power_data)
    n_battery_cap = len(battery_capacity_arr)
    
    gen_arr = np.broadcast_to(power_data.values, shape=(n_hours, n_battery_cap)).copy()
    cons_arr = np.broadcast_to(hourly_consumption.values, shape=(n_hours, n_battery_cap)).copy()

    independent_hours_arr = np.zeros((n_hours, n_battery_cap), dtype=float)
    total_excess_arr =      np.zeros(n_battery_cap, dtype=float)
    total_cons_arr =        np.zeros(n_battery_cap, dtype=float)
    battery_charge_arr =    np.full(n_battery_cap, initial_charge, dtype=float)

    for t in range(n_hours):
        # 1) Charge battery from generation
        can_supplied_to_battery = (battery_capacity_arr - battery_charge_arr) / supplied_efficiency
        gen_supplied_to_battery = np.minimum(np.minimum(gen_arr[t], can_supplied_to_battery), max_supplied)
        gen_arr[t] -= gen_supplied_to_battery
        battery_charge_arr += gen_supplied_to_battery * supplied_efficiency

        # 2) Use generation to cover consumption
        need = cons_arr[t] / supplied_efficiency
        gen_supplied_to_cons = np.minimum(np.minimum(gen_arr[t], need), max_supplied)
        gen_arr[t] -= gen_supplied_to_cons
        cons_arr[t] -= gen_supplied_to_cons * supplied_efficiency
        total_cons_arr += gen_supplied_to_cons * supplied_efficiency

        # 3) Use battery to cover remaining consumption
        need = cons_arr[t] / supplied_efficiency
        battery_supplied_to_cons = np.minimum(np.minimum(battery_charge_arr, need), max_supplied)
        battery_charge_arr -= battery_supplied_to_cons
        cons_arr[t] -= battery_supplied_to_cons * supplied_efficiency
        total_cons_arr += battery_supplied_to_cons * supplied_efficiency

        # 4) Excess generation
        gen_excess = np.maximum(0, gen_arr[t])
        total_excess_arr += gen_excess * supplied_efficiency

        # 5) Mark independent hours
        independent_hours_arr[t] = cons_arr[t] <= 1e-5

        # 6) Battery self-discharge
        battery_charge_arr *= (1 - battery_loss)

    return {
        "independent_hours_arr": independent_hours_arr, 
        "total_excess_arr": total_excess_arr, 
        "total_consumption_arr": total_cons_arr
    }


def simulation_non_battery_system(
        hourly_generation, 
        consumption_curve, 
        max_supplied=2000,
        supplied_efficiency=0.95    
    ):
    
    independent_count = 0

    for i, gen in hourly_generation.iterrows():
        hour = i.hour
        cons = consumption_curve[hour]
        gen = gen.values[0]

        supplied_possible = min(gen, max_supplied)
        if supplied_possible * supplied_efficiency >= cons:
            independent_count += 1

    return independent_count