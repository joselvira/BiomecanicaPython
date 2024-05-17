# %% ----------------

from typing import Optional, Union

import numpy as np
import xarray as xr

import scipy.integrate as integrate


def integrate_window(daData, daWindow=None, daOffset=None, result_return="continuous"):
    """
    result_return: "continuous" or "discrete"
    """

    # If empty, fill daWindow with first and last data
    if daWindow is None:
        daWindow = (
            xr.full_like(daData.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": ["ini", "fin"]}, axis=-1
            )
        ).copy()
        daWindow.loc[dict(event=["ini", "fin"])] = np.array([0, len(daData.time)])

    if daOffset is None:
        daOffset = (xr.full_like(daData.isel(time=0).drop_vars("time"), 0.0)).copy()

    if result_return == "discrete":

        def _integrate_discrete(data, t, offset, ini, fin, ID):
            if np.isnan(ini) or np.isnan(fin):
                return np.nan
            ini = int(ini)
            fin = int(fin)
            # print(ID)
            # plt.plot(data[ini:fin])
            try:
                dat = integrate.cumulative_trapezoid(
                    data[ini:fin] - offset, t[ini:fin], initial=0
                )[-1]
            except:
                # print(f'Fallo al integrar en {ID}')
                dat = np.nan
            return dat

        """
        data = daData[0,0].data
        t = daData.time.data
        ini = daWindow[0,0].isel(event=0).data
        fin = daWindow[0,0].isel(event=1).data
        offset = daOffset[0,0].data
        """
        daInt = xr.apply_ufunc(
            _integrate_discrete,
            daData,
            daData.time,
            daOffset,
            daWindow.isel(event=0),
            daWindow.isel(event=1),
            daData.ID,
            input_core_dims=[["time"], ["time"], [], [], [], []],
            # output_core_dims=[['time']],
            exclude_dims=set(("time",)),
            vectorize=True,
            # join='exact',
        )

    elif result_return == "continuous":

        def _integrate_continuous(data, time, peso, ini, fin):
            # if np.count_nonzero(~np.isnan(data))==0:
            #     return np.nan
            dat = np.full(len(data), np.nan)
            try:
                ini = int(ini)
                fin = int(fin)
                # plt.plot(data[ini:fin])
                dat[ini:fin] = integrate.cumulative_trapezoid(
                    data[ini:fin] - peso, time[ini:fin], initial=0
                )
                # plt.plot(dat)
            except:
                print("Error calculando la integral")
                pass  # dat = np.full(len(data), np.nan)
            return dat

        """
        data = daDatos[2,0].data #.sel(axis='z').data
        time = daDatos.time.data
        peso=daPeso[2,0].sel(stat='media').data
        ini = daEventos[2,0].sel(evento='iniMov').data
        fin = daEventos[2,0].sel(evento='finMov').data
        plt.plot(data[int(ini):int(fin)])
        """
        daInt = xr.apply_ufunc(
            _integrate_continuous,
            daData,
            daData.time,
            daOffset,
            daWindow.isel(event=0),
            daWindow.isel(event=1),
            input_core_dims=[["time"], ["time"], [], [], []],
            output_core_dims=[["time"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            join="exact",
        )

    else:
        raise ValueError("result_return must be 'continuous' or 'discrete'")

    daInt.attrs = daData.attrs

    return daInt


def detrend_dim(da, dim, deg=1):
    """
    Detrend the signal along a single dimension
    """

    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def RMS(daData, daWindow=None):
    """
    Calculate RMS in dataarray with dataarray window
    """
    # If empty, fill daWindow with first and last data
    if daWindow is None:
        daWindow = (
            xr.full_like(daData.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": ["ini", "fin"]}, axis=-1
            )
        ).copy()
        daWindow.loc[dict(event=["ini", "fin"])] = np.array([0, len(daData.time)])

    def _rms(data, ini, fin):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.array(np.nan)
        data = data[int(ini) : int(fin)]
        data = data[~np.isnan(data)]
        return np.linalg.norm(data[~np.isnan(data)]) / np.sqrt(len(data))

    """
    data = daData[0,0,0].data
    ini = daWindow[0,0,0].sel(event='ini').data
    fin = daWindow[0,0,0].sel(event='fin').data
    """
    # daRecortado = recorta_ventana_analisis(daData, daWindow)
    daRMS = xr.apply_ufunc(
        _rms,
        daData,
        daWindow.isel(event=0),
        daWindow.isel(event=1),
        input_core_dims=[["time"], [], []],
        vectorize=True,
    )
    return daRMS


def calculate_distance(point1, point2):
    """
    Calcula la distancia entre dos puntos.
    Requiere dimensión con coordenadas x, y, z con nombre 'axis'
    """
    return np.sqrt(((point1 - point2) ** 2).sum("axis"))


# Función para detectar onsets
"""
Ref: Solnik, S., Rider, P., Steinweg, K., Devita, P., & Hortobágyi, T. (2010). Teager-Kaiser energy operator signal conditioning improves EMG onset detection. European Journal of Applied Physiology, 110(3), 489–498. https://doi.org/10.1007/s00421-010-1521-8

Función sacada de Duarte (https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb)
The Teager–Kaiser Energy operator to improve onset detection
The Teager–Kaiser Energy (TKE) operator has been proposed to increase the accuracy of the onset detection by improving the SNR of the EMG signal (Li et al., 2007).
"""


def tkeo(x):
    r"""Calculates the Teager–Kaiser Energy operator.

    Parameters
    ----------
    x : 1D array_like
        raw signal

    Returns
    -------
    y : 1D array_like
        signal processed by the Teager–Kaiser Energy operator

    Notes
    -----

    See this notebook [1]_.

    References
    ----------
    .. [1] https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb

    """
    x = np.asarray(x)
    y = np.copy(x)
    # Teager–Kaiser Energy operator
    y[1:-1] = x[1:-1] * x[1:-1] - x[:-2] * x[2:]
    # correct the data in the extremities
    y[0], y[-1] = y[1], y[-2]

    return y


# TODO: GENERALIZAR LA FUNCIÓN DE NORMALIZAR
def NormalizaBiela360_xr(
    daData, base_norm_horiz="time", graficas=False
):  # recibe da de daTodos. Versión con numpy
    if base_norm_horiz == "time":
        eje_x = daData.time
    elif base_norm_horiz == "biela":
        try:
            eje_x = daData.sel(n_var="AngBiela", axis="y")
        except:
            eje_x = daData.sel(n_var="AngBiela")
    else:
        print("Base de normalización no reconocida")
        return

    def _normaliza_t_aux(
        data, x, base_norm_horiz
    ):  # Función auxiliar para normalizar con xarray
        # return tnorm(data, k=1, step=-361, show=False)[0]
        if np.isnan(data).all():
            data = np.full(361, np.nan)
        else:  # elimina los nan del final y se ajusta
            data = data[~np.isnan(data)]
            x = x[: len(data)]
            if base_norm_horiz == "biela":
                x = np.unwrap(x)
                x = x - x[0]
            xi = np.linspace(0, x[-1], 361)
            data = np.interp(xi, x, data)  # tnorm(data, k=1, step=-361, show=False)[0]
        return data

    daNorm = xr.apply_ufunc(
        _normaliza_t_aux,
        daData,
        eje_x,
        base_norm_horiz,
        input_core_dims=[["time"], ["time"], []],
        output_core_dims=[["AngBielaInRepe"]],
        exclude_dims=set(("AngBielaInRepe",)),
        vectorize=True,
    ).assign_coords(
        dict(
            AngBielaInRepe=np.arange(
                361
            ),  # hay que meter esto a mano. Coords en grados
            AngBielaInRepe_rad=(
                "AngBielaInRepe",
                np.deg2rad(np.arange(361)),
            ),  # Coords en radianes
        )
    )
    daNorm.AngBielaInRepe.attrs["units"] = "deg"
    daNorm.AngBielaInRepe_rad.attrs["units"] = "rad"
    daNorm.name = daData.name
    daNorm.attrs["units"] = daData.attrs["units"]

    return daNorm


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # %%---- Create a sample
    # =============================================================================

    import numpy as np

    # import pandas as pd
    import xarray as xr
    from scipy.signal import butter, filtfilt
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    def create_time_series_xr(
        rnd_seed=None,
        num_subj=10,
        Fs=100.0,
        IDini=0,
        rango_offset=[-2.0, -0.5],
        rango_amp=[1.0, 2.2],
        rango_frec=[1.8, 2.4],
        rango_af=[0.0, 1.0],
        rango_duracion=[5.0, 5.1],
        amplific_ruido=[0.4, 0.7],
        fc_ruido=[7.0, 12.0],
    ):
        if rnd_seed is not None:
            np.random.seed(
                rnd_seed
            )  # para mantener la consistencia al crear los datos aleatorios
        subjects = []
        for subj in range(num_subj):
            # print(subj)
            a = np.random.uniform(rango_amp[0], rango_amp[1])
            of = np.random.uniform(rango_offset[0], rango_offset[1])
            f = np.random.uniform(rango_frec[0], rango_frec[1])
            af = np.deg2rad(
                np.random.uniform(rango_af[0], rango_af[1])
            )  # lo pasa a radianes
            err = a * np.random.uniform(amplific_ruido[0], amplific_ruido[1])
            fc_err = np.random.uniform(fc_ruido[0], fc_ruido[1])
            duracion = np.random.uniform(rango_duracion[0], rango_duracion[1])

            Ts = 1.0 / Fs  # intervalo de tiempo entre datos en segundos
            t = np.arange(0, duracion, Ts)

            senal = np.array(of + a * np.sin(2 * np.pi * f * t + af))

            # Crea un ruido aleatorio controlado
            pasadas = 2.0  # nº de pasadas del filtro adelante y atrás
            orden = 2
            Cf = (2 ** (1 / pasadas) - 1) ** (
                1 / (2 * orden)
            )  # correction factor. Para 2nd order = 0.802
            Wn = 2 * fc_err / Fs / Cf
            b1, a1 = butter(orden, Wn, btype="low")
            ruido = filtfilt(b1, a1, np.random.uniform(a - err, a + err, len(t)))

            #################################
            subjects.append(senal + ruido)
            # subjects.append(np.expand_dims(senal + ruido, axis=0))
            # sujeto.append(pd.DataFrame(senal + ruido, columns=['value']).assign(**{'ID':'{0:02d}'.format(subj+IDini), 'time':np.arange(0, len(senal)/Fs, 1/Fs)}))

        # Pad data to last the same
        import itertools

        data = np.array(list(itertools.zip_longest(*subjects, fillvalue=np.nan)))

        data = xr.DataArray(
            data=data,
            coords={
                "time": np.arange(data.shape[0]) / Fs,
                "ID": [
                    f"{i:0>2}" for i in range(num_subj)
                ],  # rellena ceros a la izq. f'{i:0>2}' vale para int y str, f'{i:02}' vale solo para int
            },
        )
        return data

    rnd_seed = np.random.seed(
        12340
    )  # fija la aleatoriedad para asegurarse la reproducibilidad
    n = 10
    duracion = 15
    freq = 200.0
    Pre_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[25, 29],
            rango_amp=[40, 45],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["a"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[22, 26],
            rango_amp=[36, 40],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["a"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_a = xr.concat([Pre_a, Post_a], dim="momento")
    var_a.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    # =============================================================================
    # %% TEST INTEGRATE
    # =============================================================================
    daWindow = (
        xr.full_like(var_a.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": ["ini", "fin"]}, axis=-1
        )
    ).copy()
    daWindow.loc[dict(event=["ini", "fin"])] = np.array([100, -300])
    daWindow.loc[dict(event=["ini", "fin"], ID="00")] = np.array([0, len(var_a.time)])

    # Discrete
    integrate_window(var_a, result_type="discrete")
    integrate_window(var_a, daWindow, result_type="discrete")

    # Continuous
    integ = integrate_window(var_a, daWindow, result_type="continuous")
    integ.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    integ = integrate_window(var_a, daWindow, daOffset=60, result_type="continuous")
    integ.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    integ = integrate_window(
        var_a, daWindow, daOffset=var_a.mean("time"), result_type="continuous"
    )
    integ.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    # =============================================================================
    # %% TEST RMS
    # =============================================================================
    daWindow = (
        xr.full_like(var_a.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": ["ini", "fin"]}, axis=-1
        )
    ).copy()
    daWindow.loc[dict(event=["ini", "fin"])] = np.array([100, 300])
    daWindow.loc[dict(event=["ini", "fin"], ID="00")] = np.array([0, len(var_a.time)])

    RMS(var_a, daWindow)
