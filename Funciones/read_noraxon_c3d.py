# -*- coding: utf-8 -*-
"""
Created on Fry Mar 08 12:17:37 2024

@author: josel
"""

# =============================================================================
# %% Carga librerías
# =============================================================================
import warnings  # para quitar warnings de no encontrar points

import numpy as np
import pandas as pd
import xarray as xr

import c3d
import time

# import sys
# sys.path.append('F:\Programacion\Python\Mios\Functions')
# #sys.path.append('G:\Mi unidad\Programacion\Python\Mios\Functions')

__author__ = "Jose Luis Lopez Elvira"
__version__ = "0.0.1"
__date__ = "08/03/2024"

"""
Modificaciones:    
    08/03/2024, v0.0.1
            - Empezado tomando trozos sueltos.
            
"""


# =============================================================================
# %% Carga trayectorias desde c3d
# =============================================================================
def read_noraxon_c3d_xr(file, section=None, n_vars_load=None, coincidence="similar"):
    if section not in [
        "Trajectories",
        "Model Outputs",
        "EMG",
    ]:  # not ('Trajectories' in section or 'Model Outputs'in section or 'Forces' in section or 'EMG'in section):
        raise Exception(
            'Section header not found, try "Trajectories", "Model outputs", "Forces" or "EMG"'
        )
        return

    timer = time.perf_counter()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()  # inicia el contador de tiempo
        # print(f'Loading section {section}, file: {file.name}')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freq = reader.point_rate
                freq_analog = reader.analog_rate

                points = []
                analog = []
                # for i, (_, p, a) in enumerate(reader.read_frames()):
                for _, p, a in reader.read_frames():
                    points.append(p)
                    analog.append(a)
                    # if not i % 10000 and i:
                    #     print("Extracted %d point frames", len(points))

        # Trajectiories and Modeled outputs
        if "Trajectories" in section or "Model Outputs" in section:
            labels = [s.replace(" ", "") for s in reader.point_labels]
            data = np.asarray(points)[:, :, :3]

            coords = {
                "time": np.arange(data.shape[0]) / freq,
                "n_var": labels,
                "axis": ["x", "y", "z"],
            }
            da = xr.DataArray(
                data,  # =np.expand_dims(data, axis=0),
                dims=coords.keys(),
                coords=coords,
                name="Trajectories",
                attrs={
                    "freq": freq,
                    "units": "mm",
                },
            ).transpose("n_var", "axis", "time")
            if "Trajectories" in section:
                # Delete unnamed trajectories and modeled outputs
                da = da.sel(
                    n_var=(
                        ~da.n_var.str.startswith("*") & ~da.n_var.str.contains("USERMO")
                    )
                )
            if "Model Outputs" in section:
                da = da.sel(n_var=da.n_var.str.contains("USERMO"))

        # Analogs
        elif section in ["Forces", "EMG"]:  # ('Forces' in section or 'EMG' in section):
            labels_analog = [
                s.split(".")[1].replace(" ", "") for s in reader.analog_labels
            ]
            data_analog = np.concatenate(analog, axis=1)

            # data_analog.shape
            coords = {
                "n_var": labels_analog,
                "time": np.arange(data_analog.shape[1]) / freq_analog,
            }
            da_analog = xr.DataArray(
                data=data_analog,
                dims=coords.keys(),
                coords=coords,
                attrs={"freq": freq_analog},
            )

            # EMG
            if section == "EMG":
                if da_analog.n_var.str.contains("EMG").any():
                    da = da_analog.sel(n_var=da_analog.n_var.str.contains("EMG"))
                    da.attrs["units"] = "mV"
                    # da.n_var.sortby('n_var')
                    # da.plot.line(x='time', col='n_var', col_wrap=3)
                else:
                    da = xr.DataArray()
                    raise Exception("No EMG data in file")

        da.time.attrs["units"] = "s"
        da.name = section

        # print('Tiempo {0:.3f} s \n'.format(time.perf_counter()-timerSub))

    except Exception as err:
        da = xr.DataArray()
        print(f"\nATENCIÓN. No se ha podido procesar {file.name}, {err}\n")

    if n_vars_load is not None and "n_var" in da.coords:
        da = da.sel(n_var=n_vars_load)

    return da  # daTrajec, daModels, daForce, daEMG


def read_vicon_c3d_xr_global(
    file, section=None, n_vars_load=None, coincidence="similar"
):
    # if section not in ['Trajectories', 'Model Outputs', 'Forces', 'EMG']:
    # MEJORAR
    if not (
        "Trajectories" in section
        or "Model Outputs" in section
        or "Forces" in section
        or "EMG" in section
    ):
        raise Exception(
            'Section header not found, try "Trajectories", "Model outputs", "Forces" or "EMG"'
        )
        return

    timer = time.time()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.time()  # inicia el contador de tiempo
        print(f"Loading section {section}, file: {file.name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freq = reader.point_rate
                freq_analog = reader.analog_rate

                points = []
                analog = []
                for i, (_, p, a) in enumerate(reader.read_frames()):
                    points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print("Extracted %d point frames", len(points))

        # Trajectiories and Modeled outputs
        if "Trajectories" in section or "Model Outputs" in section:
            labels = [s.replace(" ", "") for s in reader.point_labels]
            data = np.asarray(points)[:, :, :3]

            coords = {
                "time": np.arange(data.shape[0]) / freq,
                "n_var": labels,
                "axis": ["x", "y", "z"],
            }
            da = xr.DataArray(
                data,  # =np.expand_dims(data, axis=0),
                dims=coords.keys(),
                coords=coords,
                name="Trajectories",
                attrs={
                    "freq": freq,
                    "units": "mm",
                },
            ).transpose("n_var", "axis", "time")
            if "Trajectories" in section:
                # Delete unnamed trajectories and modeled outputs
                daTraj = da.sel(
                    n_var=(
                        ~da.n_var.str.startswith("*") & ~da.n_var.str.contains("USERMO")
                    )
                )
            if "Model Outputs" in section:
                daMod = da.sel(n_var=da.n_var.str.contains("USERMO"))

        # Analogs
        elif "Forces" in section or "EMG" in section:
            labels_analog = [
                s.split(".")[0].replace(" ", "") for s in reader.analog_labels
            ]
            data_analog = np.concatenate(analog, axis=1)

            # data_analog.shape
            coords = {
                "n_var": labels_analog,
                "time": np.arange(data_analog.shape[1]) / freq_analog,
            }
            da_analog = xr.DataArray(
                data=data_analog,
                dims=coords.keys(),
                coords=coords,
                attrs={"freq": freq_analog},
            )

            # Forces
            if da_analog.n_var.str.contains(
                "Force"
            ).any():  #'Force' in da_analog.n_var:
                daForces = da_analog.sel(
                    n_var="Force"
                )  # .sel(n_var=da_analog.n_var.str.contains('Force'))
                if len(daForces.n_var) == 3:  # 1 platform
                    x = daForces.isel(n_var=0)
                    y = daForces.isel(n_var=1)
                    z = daForces.isel(n_var=2)
                    daForces = (
                        xr.concat([x, y, z], dim="axis")
                        .assign_coords(n_var="plat1")
                        .assign_coords(axis=["x", "y", "z"])
                        .expand_dims({"n_var": 1})
                    )
                elif len(daForces.n_var) == 6:  # 2 platforms
                    x = daForces.isel(n_var=[0, 3])
                    y = daForces.isel(n_var=[1, 4])
                    z = daForces.isel(n_var=[2, 5])
                    daForces = (
                        xr.concat([x, y, z], dim="axis")
                        .assign_coords(n_var=["plat1", "plat2"])
                        .assign_coords(axis=["x", "y", "z"])
                    )
                    daForces.time.attrs["units"] = "s"
                # da.plot.line(x='time', col='axis', hue='n_var')
            else:
                daFor = xr.DataArray()

            # EMG
            if da_analog.n_var.str.contains("EMG").any():
                daEMG = da_analog.sel(n_var=da_analog.n_var.str.contains("EMG"))
                daEMG.time.attrs["units"] = "s"
                # daEMG.n_var.sortby('n_var')
                # daEMG.plot.line(x='time', col='n_var', col_wrap=3)
            else:
                daEMG = xr.DataArray()

        # da.time.attrs['units']='s'

        print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))

    except Exception as err:
        print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")

    if n_vars_load:
        da = da.sel(n_var=n_vars_load)

    daRet = []
    if "Trajectories" in section:
        daRet.append(daTraj)
    if "Model Outputs" in section:
        daRet.append(
            daMod,
        )
    if "Forces" in section:
        daRet.append(daForces)
    if "EMG" in section:
        daRet.append(daEMG)

    if len(daRet) == 1:
        daRet = daRet[0]

    return daRet  # daTrajec, daModels, daForce, daEMG


def read_vicon_c3d_xr_global_ds(
    file, section="Trajectories", n_vars_load=None, coincidence="similar"
):

    timer = time.time()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.time()  # inicia el contador de tiempo
        print("Cargando archivo: {0:s}".format(file.name))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freq = reader.point_rate
                freq_analog = reader.analog_rate

                points = []
                analog = []
                for i, (_, p, a) in enumerate(reader.read_frames()):
                    points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print("Extracted %d point frames", len(points))

                labels = [s.replace(" ", "") for s in reader.point_labels]
                labels_analog = [
                    s.split(".")[0].replace(" ", "") for s in reader.analog_labels
                ]
        data = np.asarray(points)[:, :, :3]
        data_analog = np.concatenate(analog, axis=1)

        # Trajectiories and Modeled outputs
        coords = {
            "time": np.arange(data.shape[0]) / freq,
            "n_var": labels,
            "axis": ["x", "y", "z"],
        }
        da = xr.DataArray(
            data,  # =np.expand_dims(data, axis=0),
            dims=coords.keys(),
            coords=coords,
            name="Trajectories",
            attrs={
                "freq": freq,
                "units": "mm",
            },
        ).transpose("n_var", "axis", "time")
        da.time.attrs["units"] = "s"

        # if section=='Trajectories':
        #     #Delete unnamed trajectories and modeled outputs
        #     da = da.sel(n_var=(~da.n_var.str.startswith('*') & ~da.n_var.str.contains('USERMO')))
        # elif section=='Model Outputs':
        #     da = da.sel(n_var=da.n_var.str.contains('USERMO'))

        daTrajec = da.sel(
            n_var=(~da.n_var.str.startswith("*") & ~da.n_var.str.contains("USERMO"))
        )

        daModels = da.sel(n_var=da.n_var.str.contains("USERMO"))
        # da.isel(axis=0).plot.line(x='time', hue='n_var')

        # Analogs
        # data_analog.shape
        coords = {
            "n_var": labels_analog,
            "time": np.arange(data_analog.shape[1]) / freq_analog,
        }
        da_analog = xr.DataArray(
            data=data_analog,
            dims=coords.keys(),
            coords=coords,
            attrs={"freq": freq_analog},
        )

        # Forces
        if da_analog.n_var.str.contains("Force").any():  #'Force' in da_analog.n_var:
            daForce = da_analog.sel(
                n_var="Force"
            )  # .sel(n_var=da_analog.n_var.str.contains('Force'))
            if len(daForce.n_var) == 3:  # 1 platform
                x = daForce.isel(n_var=0)
                y = daForce.isel(n_var=1)
                z = daForce.isel(n_var=2)
                daForce = (
                    xr.concat([x, y, z], dim="axis")
                    .assign_coords(n_var="plat1")
                    .assign_coords(axis=["x", "y", "z"])
                    .expand_dims({"n_var": 1})
                )
            elif len(daForce.n_var) == 6:  # 2 platforms
                x = daForce.isel(n_var=[0, 3])
                y = daForce.isel(n_var=[1, 4])
                z = daForce.isel(n_var=[2, 5])
                daForce = (
                    xr.concat([x, y, z], dim="axis")
                    .assign_coords(n_var=["plat1", "plat2"])
                    .assign_coords(axis=["x", "y", "z"])
                )
            # daForce.plot.line(x='time', col='axis', hue='n_var')
        else:
            daForce = xr.DataArray()

        # EMG
        if da_analog.n_var.str.contains("EMG").any():
            daEMG = da_analog.sel(n_var=da_analog.n_var.str.contains("EMG"))
            daEMG.n_var.sortby("n_var")
            # daEMG.plot.line(x='time', col='n_var', col_wrap=3)
        else:
            daEMG = xr.DataArray()

        print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))

    except Exception as err:
        print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")

    if n_vars_load:
        da = da.sel(n_var=n_vars_load)

    daTodo = xr.Dataset(
        {"Trajectories": daTrajec, "Modeled": daModels, "Forces": daForce, "EMG": daEMG}
    )

    return daTodo  # daTrajec, daModels, daForce, daEMG


# =============================================================================
# %% MAIN
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path

    file = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.c3d"
    )
    daTrajec = read_noraxon_c3d_xr(file, section="EMG")

    ruta_archivo = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000"
    )
    file = ruta_archivo / "2024-03-08-10-43_PO_S000_carrera_001.c3d"

    daTrajec = read_noraxon_c3d_xr(file, section="EMG")
    daTrajec.isel(n_var=slice(6)).plot.line(x="time", col="n_var", col_wrap=3)
