# %% -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:01:08 2021
Funciones para realizar cortes en señales cíclicas usando un criterio interno o externo.
Basado en xarray.

@author: josel
"""

from typing import Optional, Union, Any

import numpy as np
import xarray as xr

import itertools

import matplotlib.pyplot as plt


__author__ = "Jose Luis Lopez Elvira"
__version__ = "v.4.1.0"
__date__ = "23/04/2024"


"""
Modificaciones:
    23/04/2024, v4.1.0
        - Actualizado versión slice con Polars, puede que sea más rapido.
        - TODO: Probar versión detect_onset que use find_peaks con la señal derivada
    
    22/02/2024, v4.0.0
        - Cambio a funciones independientes para usar con el xarray accessor.
        - Se mantiene la versión con clase por retrocompatibilidad.
        - Versión inicial basada en slice_time_series_phases.py v3.1.2

    10/11/2023, v3.1.2
        - Ahora mantiene las unidades en el dataframe cortado.
    
    11/03/2023, v3.1.1
        - Solucionado error con función find_peaks_aux. Hace copia antes de buscar
        cortes.
        - Incluido parámetro show en función detect_onset_detecta_aux.
    
    13/02/2023, v3.1.0
        - Las funciones find_peaks_aux y detect_onset_detecta_aux admiten un
        argumento para buscar cortes a partir de la media + x veces SD.
    
    09/02/2023, v3.0.0
        - Metido todo dentro de la clase SliceTimeSeriesPhases.
        - Cambiada nomenclatura num_repes, max_repes, descarta_rep_ini y
        descarta_rep_fin a num_cuts, max_cuts, discard_cuts_end y
        discard_cuts_end.
    
    28/01/2023, v2.1.0
        - Función común corta_repes que distribuye según los datos sean Pandas
        o xarray.
        - Función en xarray que devuelve sólo los nº de índice de los events.
        - Cambiada terminología, de repe a corte (para no confundir con repe de
        repeticiones/series).

    26/03/2022, v2.0.1
        - Como variable de referencia (var_referencia) ahora se pasa un dict
        con varias dimensiones y sus coordenadas.
        - Incluida una versión en pruebas para tratar en bloques de dataarray.

    11/12/2021, v2.0.0
        - Incluida una versión con xarray, mucho más rápida.

    24/11/2021, v1.2.0
        - Incluida opción de que incluya al final de cada repetición el primer dato de la siguiente. De esta forma tienen más continuidad.

    08/11/2021, v1.1.1
        - A la función auxiliar detect_onset_detecta_aux se le puede pasar como argumento corte_ini=1 para que coja el final de la ventana encontrada. Por defecto coge el inicio.
        - Además a la misma función cuando se pide que corte con el final de la ventana, le suma 1 para que coja cuando ha superado el umbral.
        - También si el corte_ini=0 quita el primer corte y si es =1 quita el último, porque suelen quedar cortados.
    
    13/10/2021, v1.1.0
        - Incluidos argumentos para eliminar repeticiones iniciales o finales.
        - Falta poder elegir eliminar repeticiones intermedias
    
    30/09/2021, v1.0.0
        - Versión inicial
"""


# =============================================================================
# Custom function to adapt from Detecta detect_onset
# =============================================================================
def detect_onset_detecta_aux(
    data,  #: Optional[np.array],
    event_ini: Optional[int] = 0,
    xSD: Optional[Union[str, dict]] = None,
    show: Optional[bool] = False,
    **args_func_events,
) -> xr.DataArray:

    # If event_ini=1 is passed as an argument, it takes the cut at the end of each window.
    try:
        from detecta import detect_onset
    except ImportError:
        raise Exception(
            "This function needs Detecta to be installed (https://pypi.org/project/detecta/)"
        )

    # try: #detect_onset returns 2 indexes. If not specified, select the first
    #     event_ini=args_func_events['event_ini']
    #     args_func_events.pop('event_ini', None)
    # except:
    #     event_ini=0
    if (
        xSD is not None
    ):  # the threshold is defined by the mean + x times the standard deviation
        if "threshold" in args_func_events:
            args_func_events.pop("threshold", None)
        args_func_events["threshold"] = (
            np.mean(data, where=~np.isnan(data))
            + np.std(data, where=~np.isnan(data)) * xSD
        )
        # print(args_func_events, np.mean(data, where=~np.isnan(data)), np.std(data, where=~np.isnan(data)), xSD)

    events = detect_onset(data, **args_func_events)

    if event_ini == 1:
        events = (
            events[:, event_ini] + 1
        )  # if the end of the window is chosen, 1 is added to start when the threshold has already been exceeded
        events = events[:-1]  # removes the last one because it is usually incomplete
    else:
        events = events[
            :, event_ini
        ]  # keeps the first or second value of each data pair
        events = events[1:]  # removes the last one because it is usually incomplete

    if show:
        SliceTimeSeriesPhases.show_events(
            data, events, threshold=args_func_events["threshold"]
        )

    return events


# =============================================================================
# Custom function to adapt from scipy.signal find_peaks
# =============================================================================
def find_peaks_aux(
    data,  #: Optional[np.array],
    xSD: Optional[Union[str, dict]] = None,
    show: Optional[bool] = False,
    **args_func_events,
) -> xr.DataArray:
    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise Exception("This function needs scipy.signal to be installed")
    if (
        xSD is not None
    ):  # the threshold is defined by the mean + x times the standar deviation
        if isinstance(xSD, list):
            args_func_events["height"] = [
                np.mean(data[~np.isnan(data)]) + xSD[0] * np.std(data[~np.isnan(data)]),
                np.mean(data[~np.isnan(data)]) + xSD[1] * np.std(data[~np.isnan(data)]),
            ]
        else:
            args_func_events["height"] = np.mean(data[~np.isnan(data)]) + xSD * np.std(
                data[~np.isnan(data)]
            )  # , where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))

    data = data.copy()

    # Deal with nans
    data[np.isnan(data)] = -np.inf

    events, _ = find_peaks(data, **args_func_events)

    if show:
        SliceTimeSeriesPhases.show_events(
            data, events, threshold=args_func_events["height"]
        )

    return events  # keeps the first value of each data pair


# =============================================================================
# Custom function to detect onsets based on find peaks from scipy.signal
# UNDER CONSTRUCTION
# =============================================================================
def find_onset_aux(
    data,  #: Optional[np.array],
    xSD: Optional[Union[str, dict]] = None,
    show: Optional[bool] = False,
    **args_func_events,
) -> xr.DataArray:
    try:
        from scipy.signal import find_peaks, detrend
        from scipy import integrate
    except ImportError:
        raise Exception("This function needs scipy.signal to be installed")

    # -------------------------------------
    # ---- Detecta onset a partir de detect peaks e integral
    umbral = 80
    daCortado = detect_events(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(threshold=umbral, show=False),
    )

    data = daTodos[0, 0, 0].values
    do = detect_onset(data, **dict(threshold=umbral))

    integ = integrate.cumulative_trapezoid(data - umbral, daTodos.time, initial=0)
    # integDetr = integ
    integDetr = detrend(integ)
    plt.plot(integDetr)
    plt.plot(daIntegDetr[0, 0, 0])

    fp = find_peaks(integDetr, **dict(height=0))[0]
    do[:7, 1] - fp[:7]

    from general_processing_functions import integrate_window, detrend_dim

    # Integra la señal
    daInteg = integrate_window(daTodos - umbral, daOffset=0)  # daTodos.isel(time=0))
    # daIntegDetr = daInteg
    # Elimina la tendencia
    daIntegDetr = detrend_dim(daInteg, "time")
    (daIntegDetr[2, 0, 0] * 5 + umbral).plot.line(x="time")
    daTodos[2, 0, 0].plot.line(x="time")

    # Busca cortes
    daCortadoIdx = detect_events(
        data=daIntegDetr,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        # reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(height=-200, show=True),
    )
    daTodos[2, 0, 0].plot.line(x="time")
    plt.plot(
        daCortadoIdx[2, 0, 0] / daTodos.freq,
        [umbral] * len(daCortadoIdx[2, 0, 0]),
        "ro",
    )
    (daCortadoIdx[2, 0, 0] / daTodos.freq).plot(marker="o")

    daCortado[2, 0, 0]
    daCortadoIdx[2, 0, 0]

    # -------------------------------------
    if (
        xSD is not None
    ):  # the threshold is defined by the mean + x times the standar deviation
        if isinstance(xSD, list):
            args_func_events["height"] = [
                np.mean(data[~np.isnan(data)]) + xSD[0] * np.std(data[~np.isnan(data)]),
                np.mean(data[~np.isnan(data)]) + xSD[1] * np.std(data[~np.isnan(data)]),
            ]
        else:
            args_func_events["height"] = np.mean(data[~np.isnan(data)]) + xSD * np.std(
                data[~np.isnan(data)]
            )  # , where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))

    data = data.copy()

    # Deal with nans
    data[np.isnan(data)] = -np.inf

    events, _ = find_peaks(data, **args_func_events)

    if show:
        SliceTimeSeriesPhases.show_events(
            data, events, threshold=args_func_events["height"]
        )

    return events  # keeps the first value of each data pair


def detect_events(
    data: Optional[xr.DataArray] = xr.DataArray(),
    freq: Optional[float] = None,
    n_dim_time: Optional[str] = "time",
    reference_var: Optional[Union[str, dict]] = None,
    discard_phases_ini: int = 0,
    n_phases: Optional[int] = None,
    discard_phases_end: int = 0,
    # include_first_next_last: Optional[bool] = False,
    max_phases: int = 100,
    func_events: Optional[Any] = detect_onset_detecta_aux,
    **kwargs_func_events: Optional[dict],
) -> xr.DataArray:
    # TODO: AJUSTAR LA FUNCIÓN PARA QUE ADMITA UMBRALES ESPECÍFICOS DE CADA ENSAYO
    # TODO: OPTIMIZAR CUANDO HAY reference_var QUE BUSQUE CORTES SÓLO EN ESA VARIABLE

    if func_events == None:
        raise Exception("A function to detect the events must be specified")

    if freq is None:
        if "freq" in data.attrs:
            freq = data.attrs["freq"]
        else:
            if not data.isnull().all():
                freq = (
                    np.round(
                        1 / (data[n_dim_time][1] - data[n_dim_time][0]),
                        1,
                    )
                ).data

    def detect_aux_idx(
        dat,
        data_reference_var=None,
        func_events=None,
        max_phases=100,
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **kwargs_func_events,
    ):
        events = np.full(max_phases, np.nan)

        if (
            np.count_nonzero(~np.isnan(dat)) == 0
            or np.count_nonzero(~np.isnan(data_reference_var)) == 0
        ):
            return events

        try:
            evts = func_events(data_reference_var, **kwargs_func_events)
        except:
            return events

        # If necessary, adjust initial an final events
        evts = evts[discard_phases_ini:]

        if n_phases == None:
            evts = evts[: len(evts) - discard_phases_end]
        else:  # if a specific number of phases from the first event is required
            if len(evts) >= n_phases:
                evts = evts[: n_phases + 1]
            else:  # not enought number of events in the block, trunkated to the end
                pass
        events[: len(evts)] = evts
        return events

    """
    dat = data[0,0,0].values
    data_reference_var = data.sel(reference_var)[0].values
    """
    da = xr.apply_ufunc(
        detect_aux_idx,
        data,
        data.sel(reference_var),
        func_events,
        max_phases,
        discard_phases_ini,
        n_phases,
        discard_phases_end,
        input_core_dims=[
            [n_dim_time],
            [n_dim_time],
            [],
            [],
            [],
            [],
            [],
        ],  # lista con una entrada por cada argumento
        output_core_dims=[["n_event"]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        # keep_attrs=True,
        kwargs=kwargs_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .dropna(dim="n_event", how="all")
        .dropna(dim="n_event", how="all")
    )
    da.name = "Events"
    return da


def slice_time_series(
    data: Optional[xr.DataArray] = xr.DataArray(),
    events: Optional[xr.DataArray] = None,
    freq: Optional[float] = None,
    n_dim_time: Optional[str] = "time",
    reference_var: Optional[Union[str, dict]] = None,
    discard_phases_ini: int = 0,
    n_phases: Optional[int] = None,
    discard_phases_end: int = 0,
    include_first_next_last: Optional[bool] = False,
    max_phases: Optional[int] = 100,
    func_events: Optional[Any] = None,
    split_version_function: str = "numpy",  # "polars" or "numpy"
    **kwargs_func_events,
) -> xr.DataArray:
    if events is None:  # if the events are not detected yet, detect them
        events = detect_events(
            data,
            freq,
            n_dim_time,
            reference_var,
            discard_phases_ini,
            n_phases,
            discard_phases_end,
            # include_first_next_last,
            max_phases,
            func_events,
            **kwargs_func_events,
        )

    # Numpy version
    def slice_aux(
        dat, evts, max_phases, max_time, ID, var, include_first_next_last=True
    ):
        phases = np.full((max_phases, max_time), np.nan)
        # print(ID, var)
        if (
            np.count_nonzero(~np.isnan(dat)) == 0
            or np.count_nonzero(~np.isnan(evts)) == 0
        ):
            return phases

        evts = evts[~np.isnan(evts)].astype(int)

        t = np.split(dat, evts)[1:-1]
        try:
            t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
            phases[: t.shape[0], : t.shape[1]] = t
        except:
            pass

        # To include the first value of the next slice as the last of the
        # present. Usefull when graphing cycles
        # TODO: improve vectorizing
        # TODO: CHECK TO INCLUDE ONE LAST FRAME IN EACH PHASE
        if include_first_next_last:
            for sl in range(len(evts) - 1):
                phases[sl, evts[sl + 1] - evts[sl]] = dat[evts[sl + 1]]
                # phases[0, evts[1] - evts[0]-4:evts[1] - evts[0]+2]
                # phases[1, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        return phases

    # Polars version
    if split_version_function in ["polars", "polarspiv"]:
        import polars as pl
    elif split_version_function == "pandas":
        import pandas as pd

    def slice_aux_pl(
        dat, evts, max_phases, max_time, ID, var, include_first_next_last=True
    ):
        # print(ID, var)
        phases = np.full((max_time, max_phases), np.nan)
        try:
            if (
                np.count_nonzero(~np.isnan(dat)) == 0
                or np.count_nonzero(~np.isnan(evts)) == 0
            ):
                return phases
        except Exception as inst:
            print(inst)

        # evts = np.array([0] + evts.tolist() + [len(dat)]).astype(int)
        # ind = np.repeat(range(len(evts) - 1), np.diff(evts))

        try:
            evts = evts[~np.isnan(evts)].astype(int)
            order = np.repeat(range(len(evts) - 1), np.diff(evts))

            df = pl.DataFrame(
                {
                    "data": dat[evts[0] : evts[-1]],
                    "order": order,
                }
            )

            df = pl.DataFrame(
                {"data": dat[evts[0] : evts[-1]], "idx": order}
            ).partition_by(by="idx", as_dict=False, include_key=False)

            # Rename each block to allow concatenate
            df = [df.rename({"data": f"data{n}"}) for n, df in enumerate(df)]
            dfph = pl.concat(df, how="horizontal")

            phases[: dfph.shape[0], : dfph.shape[1]] = dfph.to_numpy()

            # ph = pl.concat(df, how="horizontal").to_numpy().T
            # phases[: ph.shape[0], : ph.shape[1]] = ph

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing
            if include_first_next_last:

                # dfph[np.diff(evts)[:-1]-1,:-1]
                # dfph[135,:-1]

                for sl in range(len(evts) - 2):
                    phases[evts[sl + 1] - evts[sl], sl] = phases[
                        0, sl + 1
                    ]  # dat[evts[sl + 1]]
                    # phases[evts[1] - evts[0]-4:evts[1] - evts[0]+2, 0]
                    # phases[0:4, 1] #evts[2] - evts[1]-4:evts[2] - evts[1]+2, 1]
                    # phases[2, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        except:
            print(f"Error en {ID}, {var}")
        return phases.T

    def slice_aux_pl_pivot(
        dat, evts, max_phases, max_time, ID, var, include_first_next_last=True
    ):
        # print(ID, var)
        phases = np.full((max_time, max_phases), np.nan)
        try:
            if (
                np.count_nonzero(~np.isnan(dat)) == 0
                or np.count_nonzero(~np.isnan(evts)) == 0
            ):
                return phases
        except Exception as inst:
            print(inst)

        # evts = np.array([0] + evts.tolist() + [len(dat)]).astype(int)
        # ind = np.repeat(range(len(evts) - 1), np.diff(evts))

        try:
            evts = evts[~np.isnan(evts)].astype(int)
            phase = np.repeat(range(len(evts) - 1), np.diff(evts))

            repp = []
            for rep in np.unique(phase, return_counts=True)[1]:
                repp.append(np.arange(rep))
            ind = np.concatenate(repp)

            df = pl.DataFrame(
                {"data": dat[evts[0] : evts[-1]], "phase": phase, "ind": ind}
            )

            df = df.pivot(values="data", index="ind", columns="phase")

            phases[: df.shape[0], : df.shape[1]] = df[:, 1:].to_numpy()

            # ph = pl.concat(df, how="horizontal").to_numpy().T
            # phases[: ph.shape[0], : ph.shape[1]] = ph

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing
            if include_first_next_last:

                # dfph[np.diff(evts)[:-1]-1,:-1]
                # dfph[135,:-1]

                for sl in range(len(evts) - 2):
                    phases[evts[sl + 1] - evts[sl], sl] = phases[
                        0, sl + 1
                    ]  # dat[evts[sl + 1]]
                    # phases[evts[1] - evts[0]-4:evts[1] - evts[0]+2, 0]
                    # phases[0:4, 1] #evts[2] - evts[1]-4:evts[2] - evts[1]+2, 1]
                    # phases[2, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        except Exception as inst:
            print(f"Error en {ID}, {var}, {inst}")
        return phases.T

    def slice_aux_pd(
        dat, evts, max_phases, max_time, ID, var, include_first_next_last=True
    ):
        # print(ID, var)
        phases = np.full((max_time, max_phases), np.nan)

        try:
            if (
                np.count_nonzero(~np.isnan(dat)) == 0
                or np.count_nonzero(~np.isnan(evts)) == 0
            ):
                return phases
        except Exception as inst:
            print(inst)

        # evts = np.array([0] + evts.tolist() + [len(dat)]).astype(int)
        # ind = np.repeat(range(len(evts) - 1), np.diff(evts))

        try:

            evts = evts[~np.isnan(evts)].astype(int)
            ind = np.repeat(range(len(evts) - 1), np.diff(evts))

            df = pd.Series(dat[evts[0] : evts[-1]], index=ind)
            df = pd.concat(
                [x.rename(n).reset_index(drop=True) for n, x in df.groupby(df.index)],
                axis=1,
            )
            """
            t = time.perf_counter()
            for i in range(1000):
                ind = np.repeat(range(len(evts) - 1), np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=ind).reset_index()
                df.pivot(columns='index', values=0)                
                
            print(time.perf_counter() - t)
            
            t = time.perf_counter()
            for i in range(1000):
                pdind = pd.Series(range(len(evts) - 1)).repeat(np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=pdind.index)
                pd.concat(
                    [
                        x.rename(n).reset_index(drop=True)
                        for n, x in df.groupby(df.index)
                    ],
                    axis=1,
                )
            print(time.perf_counter() - t)

            t = time.perf_counter()
            for i in range(1000):
                ind = np.repeat(range(len(evts) - 1), np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=ind)
                pd.concat(
                    [
                        x.rename(n).reset_index(drop=True)
                        for n, x in df.groupby(df.index)
                    ],
                    axis=1,
                )
            print(time.perf_counter() - t)

            t = time.perf_counter()
            for i in range(1000):
                ind = np.repeat(range(len(evts) - 1), np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=ind)
                # df.to_frame().reset_index().unstack()
                df.to_frame().reset_index().pivot(
                    columns="index", values=[0]
                )  # , index=range(len(df)))
            print(time.perf_counter() - t)
            """

            phases[: df.shape[0], : df.shape[1]] = df.to_numpy()

            # ph = pl.concat(df, how="horizontal").to_numpy().T
            # phases[: ph.shape[0], : ph.shape[1]] = ph

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing
            if include_first_next_last:

                # dfph[np.diff(evts)[:-1]-1,:-1]
                # dfph[135,:-1]

                for sl in range(len(evts) - 2):
                    phases[evts[sl + 1] - evts[sl], sl] = phases[
                        0, sl + 1
                    ]  # dat[evts[sl + 1]]
                    # phases[evts[1] - evts[0]-4:evts[1] - evts[0]+2, 0]
                    # phases[0:4, 1] #evts[2] - evts[1]-4:evts[2] - evts[1]+2, 1]
                    # phases[2, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        except:
            print(f"Error en {ID}, {var}")
        return phases.T

    if split_version_function == "numpy":
        func_slice = slice_aux
    elif split_version_function == "polars":
        func_slice = slice_aux_pl
    elif split_version_function == "polarspiv":
        func_slice = slice_aux_pl_pivot
    elif split_version_function == "pandas":
        func_slice = slice_aux_pd
    else:
        raise ValueError(f"Unknown split_version_function: {split_version_function}")

    """
    dat=data[0,0,0,1].values
    evts=events[0,0,0,1].values
    """
    max_phases = int(events.n_event[-1])
    max_time = int(events.diff("n_event").max()) + 1
    da = xr.apply_ufunc(
        func_slice,
        data,
        events,
        max_phases,
        max_time,
        data.ID,
        data.n_var,
        include_first_next_last,
        input_core_dims=[[n_dim_time], ["n_event"], [], [], [], [], []],
        output_core_dims=[["n_event", n_dim_time]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        # kwargs=args_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .assign_coords(time=np.arange(0, len(da.time)) / data.freq)
        .dropna(dim="n_event", how="all")
        .dropna(dim=n_dim_time, how="all")
        .rename({"n_event": "phase"})
    )
    # da[0,0,0].plot.line(x="time")
    da.attrs = data.attrs
    try:
        da.time.attrs["units"] = data.time.attrs["units"]
    except:
        pass

    return da


if False:  # PRUEBA CON POLARS
    # PROBANDOOOOO
    cortes_idx = detect_events(
        data=daTodos,
        func_events=detect_onset_detecta_aux,
        **(dict(threshold=60, show=True)),
    )
    dat = data[0, 0, 0].values
    events = cortes_idx[0, 0, 0].values

    da = xr.apply_ufunc(
        slice_aux_PRUEBAAS_pl,
        data,
        events,
        max_phases,
        include_first_next_last,
        input_core_dims=[[n_dim_time], ["n_event"], [], []],
        output_core_dims=[["n_event", n_dim_time]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        # kwargs=args_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .assign_coords(time=np.arange(0, len(da.time)) / data.freq)
        .dropna(dim="n_event", how="all")
        .dropna(dim=n_dim_time, how="all")
        .rename({"n_event": "phase"})
    )


# PRUEBA ANALIZANDO TODO CON POLARS. A MEDIAS
def slice_time_series_pl(
    data: Optional[xr.DataArray] = xr.DataArray(),
    events: Optional[xr.DataArray] = None,
    freq: Optional[float] = None,
    n_dim_time: Optional[str] = "time",
    reference_var: Optional[Union[str, dict]] = None,
    discard_phases_ini: int = 0,
    n_phases: Optional[int] = None,
    discard_phases_end: int = 0,
    include_first_next_last: Optional[bool] = False,
    max_phases: int = 100,
    func_events: Optional[Any] = None,
    # split_version_function: str = "numpy",  # "polars" or "numpy"
    **kwargs_func_events,
) -> xr.DataArray:
    # Polars version
    import polars as pl

    if events is None:  # if the events are not detected yet, detect them
        events = detect_events(
            data,
            freq,
            n_dim_time,
            reference_var,
            discard_phases_ini,
            n_phases,
            discard_phases_end,
            # include_first_next_last,
            max_phases,
            func_events,
            **kwargs_func_events,
        )

    df = pl.from_pandas(data.to_dataframe().reset_index())
    evts = pl.from_pandas(events.to_dataframe().reset_index())

    # TODO:AJUSTARSE A Nº DIMENSIONES VARIABLE
    for n, d in df.group_by(evts.columns[:-2], maintain_order=True):
        print(n)

        evt = evts.filter(pl.col(d.columns[0]) == d[0, 0])
        evt = evt.select(pl.col("Events")).cast(pl.Int32)  # evt[:,-1].astype(int)
        ind = np.repeat(range(len(evt) - 1), np.diff(evt[:, -1]))
        df = pl.DataFrame({"data": dat[evt[0] : evt[-1]], "idx": ind})

        df2 = df.partition_by(by="idx", as_dict=False, include_key=False)
        df3 = [df.rename({"data": f"data{n}"}) for n, df in enumerate(df2)]

        df4 = pl.concat(df3, how="horizontal").to_numpy().T
        # To include the first value of the next slice as the last of the
        # present. Usefull when graphing cycles
        # TODO: improve vectorizing
        if include_first_next_last:
            for sl in range(len(evts) - 2):
                df4[sl, evts[sl + 1] - evts[sl]] = dat[evts[sl + 1]]
                # df4[0,-4:]
                # df4[1,:4]
        return df4


# PRUEBA CREANDO COORDENADA CON Nº DE PHASE Y SECCIONANDO DESPUÉS
# HACER DIFERENTE SI HAY VARIABLE SELECCIONADORA PARA TODO O SI EL CIRTERIO ES UNO A UNO
def slice_time_series2(
    data: Optional[xr.DataArray] = xr.DataArray(),
    events: Optional[xr.DataArray] = None,
    freq: Optional[float] = None,
    n_dim_time: Optional[str] = "time",
    reference_var: Optional[Union[str, dict]] = None,
    discard_phases_ini: int = 0,
    n_phases: Optional[int] = None,
    discard_phases_end: int = 0,
    include_first_next_last: Optional[bool] = False,
    max_phases: int = 100,
    func_events: Optional[Any] = None,
    **kwargs_func_events,
) -> xr.DataArray:
    if events is None:  # if the events are not detected yet, detect them
        events = detect_events(
            data,
            freq,
            n_dim_time,
            reference_var,
            discard_phases_ini,
            n_phases,
            discard_phases_end,
            include_first_next_last,
            max_phases,
            func_events,
            **kwargs_func_events,
        )

    # PROBANDOOOOO
    cortes_idx = detect_events(
        data=daTodos,
        func_events=detect_onset_detecta_aux,
        **(dict(threshold=60, show=True)),
    )
    dat = data[0, 0, 0].values
    events = cortes_idx[0, 0, 0].values

    def slice_aux_PRUEBAAS(dat, events, max_phases=50, include_first_next_last=True):
        events = np.array([0] + events.tolist() + [len(dat)]).astype(int)
        ind = np.repeat(range(len(events) - 1), np.diff(events))
        # da2=da.assign_coords(time=ind)
        da2 = da.assign_coords(time2=("time", ind))
        # da2 = da2.assign_coords(time2=('time', da.time.values))
        for n, gr in da2.groupby("time2"):
            gr.plot.line(x="time")  # , col="ID")

    def slice_aux(dat, events, max_phases=50, include_first_next_last=True):
        if (
            np.count_nonzero(~np.isnan(dat)) == 0
            or np.count_nonzero(~np.isnan(events)) == 0
        ):
            return np.full((max_phases, len(dat)), np.nan)

        events = events[~np.isnan(events)].astype(int)
        phases = np.full((max_phases, len(dat)), np.nan)
        t = np.split(dat, events)[1:-1]
        try:
            t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
            phases[: t.shape[0], : t.shape[1]] = t
        except:
            pass

        # To include the first value of the next slice as the last of the
        # present. Usefull when graphing cycles
        # TODO: improve vectorizing
        if include_first_next_last:
            for sl in range(len(events) - 1):
                phases[sl, events[sl + 1] - events[sl]] = dat[events[sl + 1]]
        return phases

    da = xr.apply_ufunc(
        slice_aux,
        data,
        events,
        max_phases,
        include_first_next_last,
        input_core_dims=[[n_dim_time], ["n_event"], [], []],
        output_core_dims=[["n_event", n_dim_time]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        # kwargs=args_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .assign_coords(time=np.arange(0, len(da.time)) / data.freq)
        .dropna(dim="n_event", how="all")
        .dropna(dim=n_dim_time, how="all")
        .rename({"n_event": "phase"})
    )
    da.attrs = data.attrs
    try:
        da.time.attrs["units"] = data.time.attrs["units"]
    except:
        pass

    return da


def trim_window(daDatos, daEvents=None, window=None):
    """
    Si se pasa un valor a ventana, debería pasarse sólo un evento.
    Suma la ventana (en segundos) al evento inicial
    """
    # TODO: PROBAR CON DA.PAD

    def corta_ventana(datos, ini, fin):
        # print(datos.shape, ini,fin)
        d2 = np.full(
            datos.shape, np.nan
        )  # rellena con nan al final para que tengan mismo tamaño
        try:
            ini = int(ini)
            fin = int(fin)
            if ini < 0:
                ini = 0
            if fin > len(datos):
                fin = len(datos)
            d2[: fin - ini] = datos[ini:fin]
        except:
            pass
        return d2  # datos[int(ini):int(fin)]

    if window is not None:
        if window > 0:
            daIni = daEvents
            daFin = daEvents + window * daDatos.freq
        else:
            daIni = daEvents + window * daDatos.freq
            daFin = daEvents

    else:
        daIni = daEvents.isel(n_event=0)
        daFin = daEvents.isel(n_event=1)

    daCortado = (
        xr.apply_ufunc(
            corta_ventana,
            daDatos,
            daIni,
            daFin,  # .sel(ID=daDatos.ID, repe=daDatos.repe)
            input_core_dims=[["time"], [], []],
            output_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            vectorize=True,
            # join='outer'
        )
        .assign_coords({"time": daDatos.time})
        .dropna(dim="time", how="all")
    )
    daCortado.attrs = daDatos.attrs

    if not isinstance(daCortado, xr.Dataset):
        daCortado.name = daDatos.name
        daCortado = daCortado.astype(daDatos.dtype)
    else:
        for var in list(daCortado.data_vars):  # ['F', 'v', 's', 'P', 'RFD']:
            daCortado[var].attrs = daDatos[var].attrs

    # daCortado.plot.line(x='time', row='ID', col='axis')
    return daCortado


#################################################################
class SliceTimeSeriesPhases:
    def __init__(
        self,
        data: Optional[xr.DataArray] = xr.DataArray(),
        freq: Optional[float] = None,
        n_dim_time: Optional[str] = "time",
        reference_var: Optional[Union[str, dict]] = None,
        discard_phases_ini: int = 0,
        n_phases: Optional[int] = None,
        discard_phases_end: int = 0,
        include_first_next_last: Optional[bool] = False,
        max_phases: int = 100,
        func_events: Optional[Any] = None,
        **kwargs_func_events,
    ):
        self.data = data
        self.events = None
        self.n_dim_time = n_dim_time
        self.reference_var = reference_var
        self.discard_phases_ini = discard_phases_ini
        self.n_phases = n_phases
        self.discard_phases_end = discard_phases_end
        self.include_first_next_last = include_first_next_last
        self.func_events = func_events
        self.max_phases = max_phases
        self.kwargs_func_events = kwargs_func_events

        if freq == None and not data.isnull().all():
            self.freq = (
                np.round(
                    1 / (self.data[self.n_dim_time][1] - self.data[self.n_dim_time][0]),
                    1,
                )
            ).data
        else:
            self.freq = freq

    def detect_events(self) -> xr.DataArray:
        # TODO: AJUSTAR LA FUNCIÓN PARA QUE ADMITA UMBRALES ESPECÍFICOS DE CADA ENSAYO
        def detect_aux_idx(
            data,
            data_reference_var=None,
            func_events=None,
            max_phases=50,
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            **kwargs_func_events,
        ):
            events = np.full(max_phases, np.nan)
            if (
                np.count_nonzero(~np.isnan(data)) == 0
                or np.count_nonzero(~np.isnan(data_reference_var)) == 0
            ):
                return events
            try:
                evts = func_events(data_reference_var, **kwargs_func_events)
            except:
                return events

            # If necessary, adjust initial an final events
            evts = evts[discard_phases_ini:]

            if n_phases == None:
                evts = evts[: len(evts) - discard_phases_end]
            else:  # if a specific number of phases from the first event is required
                if len(evts) >= n_phases:
                    evts = evts[: n_phases + 1]
                else:  # not enought number of events in the block, trunkated to the end
                    pass
            events[: len(evts)] = evts
            return events

        if self.func_events == None:
            raise Exception("A function to detect the events must be specified")

        da = xr.apply_ufunc(
            detect_aux_idx,
            self.data,
            self.data.sel(self.reference_var),
            self.func_events,
            self.max_phases,
            self.discard_phases_ini,
            self.n_phases,
            self.discard_phases_end,
            input_core_dims=[
                [self.n_dim_time],
                [self.n_dim_time],
                [],
                [],
                [],
                [],
                [],
            ],  # lista con una entrada por cada argumento
            output_core_dims=[["n_event"]],
            exclude_dims=set(("n_event", self.n_dim_time)),
            dataset_fill_value=np.nan,
            vectorize=True,
            dask="parallelized",
            # keep_attrs=True,
            kwargs=self.kwargs_func_events,
        )
        da = (
            da.assign_coords(n_event=range(len(da.n_event)))
            .dropna(dim="n_event", how="all")
            .dropna(dim="n_event", how="all")
        )
        self.events = da
        return da

    def slice_time_series(self, events: Optional[xr.DataArray] = None) -> xr.DataArray:
        if events is not None:  # the events are passed manually
            self.events = events
        elif self.events is None:  # if the events are not detected yet, detect them
            self.detect_events()

        def slice_aux(data, events, max_phases=50, include_first_next_last=True):
            if (
                np.count_nonzero(~np.isnan(data)) == 0
                or np.count_nonzero(~np.isnan(events)) == 0
            ):
                return np.full((max_phases, len(data)), np.nan)

            events = events[~np.isnan(events)].astype(int)
            phases = np.full((max_phases, len(data)), np.nan)
            t = np.split(data, events)[1:-1]
            try:
                t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
                phases[: t.shape[0], : t.shape[1]] = t
            except:
                pass

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing
            if include_first_next_last:
                for sl in range(len(events) - 1):
                    phases[sl, events[sl + 1] - events[sl]] = data[events[sl + 1]]
            return phases

        da = xr.apply_ufunc(
            slice_aux,
            self.data,
            self.events,
            self.max_phases,
            self.include_first_next_last,
            input_core_dims=[[self.n_dim_time], ["n_event"], [], []],
            output_core_dims=[["n_event", self.n_dim_time]],
            exclude_dims=set(("n_event", self.n_dim_time)),
            dataset_fill_value=np.nan,
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
            # kwargs=args_func_events,
        )
        da = (
            da.assign_coords(n_event=range(len(da.n_event)))
            .assign_coords(time=np.arange(0, len(da.time)) / self.freq)
            .dropna(dim="n_event", how="all")
            .dropna(dim=self.n_dim_time, how="all")
            .rename({"n_event": "phase"})
        )
        da.attrs = self.data.attrs
        try:
            da.time.attrs["units"] = self.data.time.attrs["units"]
        except:
            pass

        return da

    # =============================================================================
    # Custom function to adapt from Detecta detect_onset
    # =============================================================================
    def detect_onset_detecta_aux(
        data, event_ini=0, xSD=None, show=False, **args_func_events
    ):
        # If event_ini=1 is passed as an argument, it takes the cut at the end of each window.
        try:
            from detecta import detect_onset
        except ImportError:
            raise Exception(
                "This function needs Detecta to be installed (https://pypi.org/project/detecta/)"
            )

        # try: #detect_onset returns 2 indexes. If not specified, select the first
        #     event_ini=args_func_events['event_ini']
        #     args_func_events.pop('event_ini', None)
        # except:
        #     event_ini=0
        if (
            xSD is not None
        ):  # the threshold is defined by the mean + x times the standard deviation
            if "threshold" in args_func_events:
                args_func_events.pop("threshold", None)
            args_func_events["threshold"] = (
                np.mean(data, where=~np.isnan(data))
                + np.std(data, where=~np.isnan(data)) * xSD
            )
            # print(args_func_events, np.mean(data, where=~np.isnan(data)), np.std(data, where=~np.isnan(data)), xSD)

        events = detect_onset(data, **args_func_events)

        if event_ini == 1:
            events = (
                events[:, event_ini] + 1
            )  # if the end of the window is chosen, 1 is added to start when the threshold has already been exceeded
            events = events[
                :-1
            ]  # removes the last one because it is usually incomplete
        else:
            events = events[
                :, event_ini
            ]  # keeps the first or second value of each data pair
            events = events[1:]  # removes the last one because it is usually incomplete

        if show:
            SliceTimeSeriesPhases.show_events(
                data, events, threshold=args_func_events["threshold"]
            )

        return events

    # =============================================================================
    # Custom function to adapt from scipy.signal find_peaks
    # =============================================================================
    def find_peaks_aux(data, xSD=None, show=False, **args_func_events):
        try:
            from scipy.signal import find_peaks
        except ImportError:
            raise Exception("This function needs scipy.signal to be installed")
        if (
            xSD is not None
        ):  # the threshold is defined by the mean + x times the standar deviation
            if isinstance(xSD, list):
                args_func_events["height"] = [
                    np.mean(data[~np.isnan(data)])
                    + xSD[0] * np.std(data[~np.isnan(data)]),
                    np.mean(data[~np.isnan(data)])
                    + xSD[1] * np.std(data[~np.isnan(data)]),
                ]
            else:
                args_func_events["height"] = np.mean(
                    data[~np.isnan(data)]
                ) + xSD * np.std(
                    data[~np.isnan(data)]
                )  # , where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))

        data = data.copy()

        # Deal with nans
        data[np.isnan(data)] = -np.inf

        events, _ = find_peaks(data, **args_func_events)

        if show:
            SliceTimeSeriesPhases.show_events(
                data, events, threshold=args_func_events["height"]
            )

        return events  # keeps the first value of each data pair

    def show_events(data, events, threshold=None):
        plt.plot(data, c="b")
        plt.plot(events, data[events], "ro")
        if threshold is not None:
            plt.hlines(y=threshold, xmin=0, xmax=len(data), color="C1", ls="--", lw=1)
        plt.show()


# =============================================================================


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # ---- Create a sample
    # =============================================================================

    import numpy as np
    import time

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
    n = 5
    duracion = 5

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

    Pre_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[35, 39],
            rango_amp=[50, 55],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["b"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[32, 36],
            rango_amp=[32, 45],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["b"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_b = xr.concat([Pre_b, Post_b], dim="momento")

    Pre_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[35, 39],
            rango_amp=[10, 15],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["c"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[32, 36],
            rango_amp=[12, 16],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["c"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_c = xr.concat([Pre_c, Post_c], dim="momento")

    # concatena todos los sujetos
    daTodos = xr.concat([var_a, var_b, var_c], dim="n_var")
    daTodos.name = "Angle"
    daTodos.attrs["freq"] = 1 / (
        daTodos.time[1].values - daTodos.time[0].values
    )  # incluimos la frecuencia como atributo
    daTodos.attrs["units"] = "deg"
    daTodos.time.attrs["units"] = "s"

    # Gráficas
    daTodos.plot.line(x="time", col="momento", hue="ID", row="n_var")

    # =============================================================================
    # %% Test the functions
    # =============================================================================

    """
    #Example importing
    sys.path.insert(1, r'F:\Programacion\Python\Mios\Functions')  # add to pythonpath
    from slice_time_series_phases import SliceTimeSeriesPhases as stsp
    """
    from detecta import detect_peaks

    # Busca índices y luego lo corta
    daEventos = detect_events(data=daTodos, func_events=detect_peaks)
    dacuts = slice_time_series(
        daTodos, daEventos
    )  # corta con los índices buscados anteriormente
    dacuts.sel(n_var="a").plot.line(x="time", col="momento", hue="phase", row="ID")

    # Corta directamente
    dacuts = slice_time_series(data=daTodos, func_events=detect_peaks, max_phases=100)
    dacuts.sel(n_var="a").plot.line(x="time", col="momento", hue="phase", row="ID")

    # Especificando una de las variables para hacer todos los cortes
    dacuts = slice_time_series(
        data=daTodos, func_events=detect_peaks, reference_var=dict(n_var="b")
    )
    dacuts.stack(var_momento=("n_var", "momento")).plot.line(
        x="time", col="var_momento", hue="phase", row="ID"
    )

    # Cortar aportando cuts ya buscados o ajustados previamente
    cortes_idx = detect_events(
        data=daTodos,
        func_events=detect_peaks,
        reference_var=dict(n_var="a"),
        max_phases=100,
    )
    cortes_retocados = cortes_idx.isel(n_event=slice(3, 20, 2))

    dacor = slice_time_series(data=daTodos, events=cortes_retocados)
    dacor.isel(ID=slice(None, 6)).sel(n_var="a").plot.line(
        x="time", col="momento", hue="phase", row="ID"
    )

    cortes_idx = detect_events(
        data=daTodos,
        func_events=detect_peaks,
        reference_var=dict(n_var="a"),
        max_phases=100,
    )
    cortes_retocados = cortes_idx.isel(n_event=slice(5, 20))

    dacor = slice_time_series(data=daTodos, events=cortes_retocados)
    dacor.isel(ID=slice(None, 6)).sel(n_var="c").plot.line(
        x="time", col="momento", hue="phase", row="ID"
    )

    daCortado = slice_time_series(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(threshold=60, show=True),
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    daCortado = slice_time_series(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(height=60, distance=10),
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    daCortado = SliceTimeSeriesPhases(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(height=140, distance=1),
    ).slice_time_series()
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    daCortado = slice_time_series(
        data=daTodos,
        func_events=detect_peaks,
        reference_var=dict(momento="pre", n_var="b"),
        # max_phases=100,
        **dict(mph=140),
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    # find_peaks with xSD
    daCortado = slice_time_series(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        # reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(xSD=0.8, distance=1),
        show=True,
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    # onset by xSD
    daCortado = slice_time_series(
        daTodos,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        # reference_var=dict(momento='pre', n_var='b'),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(xSD=-1.2),
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    # Trim data, slice ini and end events
    daEvents = detect_events(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(height=0, distance=10),
    )

    # ---- Detecta onset a partir de detect peaks y derivada
    daCortado = detect_events(
        data=daTodos,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(threshold=60, show=False),
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    from general_processing_functions import integrate_window

    daInteg = integrate_window(daTodos, daOffset=daTodos.isel(time=0))
    daInteg[2, 0].plot.line(x="time")
    daCortado = detect_events(
        data=daInteg,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(height=0, show=True),
    )
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    ventana = xr.concat(
        [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
    )
    # ventana = daEvents.isel(n_event=[5,7]) #xr.concat([daEvents.isel(n_event=5), daEvents.isel(n_event=7)], dim='n_event')

    daTrimed = slice_time_series(daTodos, events=ventana)
    daTrimed.stack(var_momento=("n_var", "momento")).plot.line(
        x="time", col="var_momento", hue="phase", row="ID"
    )

    # =============================================================================
    # %% Test functions to slice (polars, numpy, ...)
    # =============================================================================
    t = time.perf_counter()
    for i in range(10):
        slices = slice_time_series(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="numpy",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
    print(f"{time.perf_counter() - t:.4f} s")

    t = time.perf_counter()
    for i in range(10):
        slices2 = slice_time_series(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="polars",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
    print(f"{time.perf_counter() - t:.4f} s")

    t = time.perf_counter()
    for i in range(10):
        slices22 = slice_time_series(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="polarspiv",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
    print(f"{time.perf_counter() - t:.4f} s")

    t = time.perf_counter()  # Con pandas de momento el más lento
    for i in range(10):
        slices3 = slice_time_series(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="pandas",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
    print(f"{time.perf_counter() - t:.4f} s")

    slices[0, 0, 0, :, -10:-6].plot.line(x="time", hue="phase")
    slices2[0, 0, 0, :, -10:-6].plot.line(x="time", hue="phase")
    slices3[0, 0, 0, :, -10:-6].plot.line(x="time", hue="phase")

    ###################################
    # With the class version
    t = time.perf_counter()
    for i in range(10):
        daCortado = SliceTimeSeriesPhases(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            # reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            # split_version_function="polars",
            **dict(xSD=0.8, distance=1),
            show=False,
        ).slice_time_series()
    print(f"{time.perf_counter() - t:.4f} s")
    daCortado.sel(n_var="b").plot.line(x="time", col="momento", hue="phase", row="ID")

    # %%Performance tests class / functions
    import time

    t = time.perf_counter()
    for i in range(50):
        daCortado = SliceTimeSeriesPhases(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=True,
            **dict(height=60, distance=10),
        ).slice_time_series()
    print("Tiempo con clase=", time.perf_counter() - t)

    t = time.perf_counter()
    for i in range(50):
        daCortado = slice_time_series(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(momento="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=True,
            **dict(height=60, distance=10),
        )
    print("Tiempo con función=", time.perf_counter() - t)

    # %%Performance tests trim
    # Muuucho más rápido con función trim
    import time

    t = time.perf_counter()
    for i in range(30):
        daEvents = detect_events(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=True,
            **dict(height=0, distance=10),
        )
        window = xr.concat(
            [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
        )
        # window = daEvents.isel(n_event=[5,7]) #xr.concat([daEvents.isel(n_event=5), daEvents.isel(n_event=7)], dim='n_event')
        daTrimed = slice_time_series(daTodos, events=window)
    print("Tiempo con clase=", time.perf_counter() - t)

    t = time.perf_counter()
    for i in range(30):
        daEvents = detect_events(
            data=daTodos,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=True,
            **dict(height=0, distance=10),
        )
        window = xr.concat(
            [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
        )
        daTrimed2 = trim_window(daTodos, window)
    print("Tiempo con clase=", time.perf_counter() - t)
