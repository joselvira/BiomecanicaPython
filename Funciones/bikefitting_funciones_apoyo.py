# -*- coding: utf-8 -*-


"""
Created on Fry Jan 12 16:50:46 2024

Funciones comunes para tratar archivos de registros de bikefitting.
Heredado de Nexus_FuncionesApoyo.py

@author: Jose Luis López Elvira
"""

# =============================================================================
# %% INICIA
# =============================================================================


__filename__ = "bikefittinh_funciones_apoyo"
__version__ = "0.2.0"
__company__ = "CIDUMH"
__date__ = "23/04/2024"
__author__ = "Jose L. L. Elvira"

"""
Modificaciones:          
    21/05/2024, v0.2.0
        - Ahora importa funciones útiles desde el package instalable biomdp.
    
    08/05/2024, v0.1.1
        - Corregido cálculo de la mecia y std longitudes restringido a region_of_interest.
    
    23/04/2024, v0.1.0
        - Corregido ajuste a la región de interés del cálculo del vAngBiela.
        - vAngBiela extrapola los primeros y últimos datos de la región de interés.
    
    12/01/2024, v0.0.1
        - Versión inicial basada en Nexus_FuncionesApoyo.py.
                - 
"""

# =============================================================================
# DEFINE OPCIONES DE PROCESADO
# =============================================================================
bCrearGraficas = False  # crea las gráficas de ángulos y posiciones
formatoImagenes = ".pdf"  #'.svg' #'.png'
bEnsembleAvg = (
    True  # puede ser True, False o 'completo' (la media con cada repe de fondo)
)

umbral_onset = 10.0  # Umbral para detectar el onset de músculo activo
delay_EMG = int(-0.650 * 2000)  # Retraso en señal EMG, en fotogramas
filtro_MVC = "todos"  # Para determinar qué grupo de archivos de MVC se usan para calcular los máximos en cada músculo
# Las opciones son:
#'todo' : coge todos los archivos con nombre MVC.
#'auto' : coge el máximo del propio archivo dinámico activo.
# Una cadena de texto común a un grupo de archivos (sprint, 200W, standar, etc.)
bSoloMVC = False  # para que procese solo las MVCs
# bComparaLadosGraf = True #para que compare lados con SPM1D
# =============================================================================
from typing import Optional, Union, Any

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # para guardar gráficas en pdf
from matplotlib.lines import (
    Line2D,
)  # necesario para controlar el formato de las líneas de la leyenda
import seaborn as sns
from pathlib import Path
import time  # para cuantificar tiempos de procesado
import spm1d  # para comparar curvas

import time

# Importa mis funciones necesarias
import biomdp.biomec_xarray_accessor  # accessor biomxr
import biomdp.Nexus_FuncionesApoyo as nfa  # funciones de apoyo con datos del Nexus
import biomdp.slice_time_series_phases as stsp
# from biomec_processing.slice_time_series_phases import SliceTimeSeriesPhases as stsp  # tratamientos para bike fitting Nexus

r"""
import sys
sys.path.append(r"F:\Programacion\Python\Mios\Functions")
sys.path.append(
    "/media/yomismo/Seagate Basic/Programacion/Python/Mios/Functions"
)  # para Linux
import biomec_xarray_processing  # accessor biomxr
import Nexus_FuncionesApoyo as nfa

# from calculaEulerAngles import euler_angles_from_rot_xyz #para calcular el ángulo entre 2 matrices de rotación
from slice_time_series_phases import SliceTimeSeriesPhases as stsp
"""
# from cortar_repes_ciclicas import corta_repes as cts

# from readViconCsv import read_vicon_csv, read_vicon_csv_pl_xr
# from read_vicon_c3d import read_vicon_c3d_xr
# print('Cargadas funciones de apoyo de mi carpeta')


# from psd import psd #funcion para visualizar Power Spectral Density


# =============================================================================
# %% Funciones varias
# =============================================================================


def carga_variables_desde_Nexus(vicon, nomVarsContinuas250) -> xr.DataArray:

    print("Cargando datos...")
    timer = time.time()  # inicia el contador de tiempo

    n_subject = vicon.GetSubjectNames()[0]
    (CarpetaSesion, ArchivoActivo) = vicon.GetTrialName()
    NumFrames = vicon.GetTrialRange()[1]
    frec = vicon.GetFrameRate()
    RegionOfInterest = (
        np.array(vicon.GetTrialRegionOfInterest()) - 1
    )  # corrección para que ajuste a la escala empezando en cero
    NumFramesParcial = RegionOfInterest[1] - RegionOfInterest[0]  # + 1

    file = CarpetaSesion + ArchivoActivo + ".csv"

    # renombrar de nuevo a antiguo (necesario para los joint centers)
    nuevo_antiguo = {
        "HJC_L": "LHJC",
        "HJC_R": "RHJC",
        "KJC_L": "LKJC",
        "KJC_R": "RKJC",
        "AJC_L": "LAJC",
        "AJC_R": "RAJC",
        # "AngBiela_LR": "AngBiela",
        # "vAngBiela_LR": "vAngBiela",
        #'PosPedal_L':'LPosPedal', 'PosPedal_R':'RPosPedal',
    }
    nomVarsContinuas250_adaptado = (
        pd.DataFrame(columns=nomVarsContinuas250)
        .rename(columns=nuevo_antiguo)
        .columns.to_list()
    )

    renombrar_vars = {
        "LHJC": "HJC_L",
        "RHJC": "HJC_R",
        "LKJC": "KJC_L",
        "RKJC": "KJC_R",
        "LAJC": "AJC_L",
        "RAJC": "AJC_R",
        "LPosPedal": "PosPedal_L",
        "RPosPedal": "PosPedal_R",
        "AngBiela": "AngBiela_LR",
        "vAngBiela": "vAngBiela_LR",
    }
    ejes = ["x", "y", "z"]

    cols = pd.MultiIndex.from_product(
        [nomVarsContinuas250_adaptado, ejes], names=["n_var", "axis"]
    )
    dfTodosArchivos = pd.DataFrame(index=np.arange(0, NumFramesParcial), columns=cols)

    for nom in nomVarsContinuas250_adaptado:
        try:
            dfTodosArchivos[nom] = np.array(vicon.GetModelOutput(n_subject, nom)[0])[
                :3, RegionOfInterest[0] : RegionOfInterest[1]
            ].T
        except:
            print(f"No se pudo cargar la variable {nom}")
            pass

    dfTodosArchivos = dfTodosArchivos.rename(columns=renombrar_vars).sort_index(axis=1)

    # Duplica AngBiela para _L, _R
    dfTodosArchivos = (
        pd.concat(
            [
                dfTodosArchivos,
                dfTodosArchivos[["AngBiela_LR"]].rename(
                    columns={"AngBiela_LR": "AngBiela_L"}
                ),
                dfTodosArchivos[["AngBiela_LR"]].rename(
                    columns={"AngBiela_LR": "AngBiela_R"}
                ),
            ],
            axis=1,
        )
        .sort_index(axis=1)
        .drop(
            "AngBiela_LR", axis=1, inplace=False
        )  # elimina para no repetirlo al separar lados
    )

    # PROBANDO QUITAR EL AJUSTE ARTIFICIAL DE ADELANTAR 180º EN LADO R
    # # Ajusta el AngBiela R para ser como el L pero con diferencia de 180º
    # dfTodosArchivos.loc[:, ("AngBiela_R", "x")] = (
    #     dfTodosArchivos.loc[:, ("AngBiela_R", "x")].where(
    #         dfTodosArchivos.loc[:, ("AngBiela_R", "x")] < np.pi,
    #         dfTodosArchivos.loc[:, ("AngBiela_R", "x")] - 2 * np.pi,
    #     )
    #     + np.pi
    # )
    # dfTodosArchivos.loc[:, ("AngBiela_R", "y")] = (
    #     dfTodosArchivos.loc[:, ("AngBiela_R", "x")] - np.pi
    # )  # dfprovis.loc[:, ('AngBiela_R', 'y')].where(dfprovis.loc[:, ('AngBiela_R', 'y')]<0.0, dfprovis.loc[:, ('AngBiela_R', 'y')]-2*np.pi)+np.pi
    # dfTodosArchivos.loc[:, ("AngBiela_R", "z")] = (
    #     dfTodosArchivos.loc[:, ("AngBiela_R", "z")].where(
    #         dfTodosArchivos.loc[:, ("AngBiela_R", "z")] < 0.0,
    #         dfTodosArchivos.loc[:, ("AngBiela_R", "z")] - 2 * np.pi,
    #     )
    #     + np.pi
    # )

    dfTodosArchivos.insert(
        0, "ID", Path(file).parts[-3] + "_" + ArchivoActivo
    )  # ]*len(dfTodosArchivos))
    dfTodosArchivos.insert(
        1, "time", np.arange(0, len(dfTodosArchivos))[0 : len(dfTodosArchivos)] / frec
    )  # la parte final es para asegurarse de que se queda con el tamaño adecuado
    #####################

    # ----------------
    """
    #Duplica la variable usada para hacer los cortes para que funcione el segmenta_xr
    import itertools
    dfprovis = dfTodosArchivos.copy()
    for bloque in itertools.product(['AngBiela_L', 'AngBiela_R', 'AngBiela_LR'], ['x', 'y', 'z']):
        dfprovis.loc[:, bloque] = dfTodosArchivos.loc[:, ('AngBiela_LR', 'y')]
    dfTodosArchivos = dfprovis
    """
    # ----------------

    # El python de Nexus es antiguo su xarray y no tiene str
    # Pasa de df a da
    daTodos = (
        dfTodosArchivos.set_index(["ID", "time"])
        .stack(future_stack=True)
        .to_xarray()
        .to_array()
        .rename({"variable": "n_var"})
    )  # .transpose('Archivo', 'nom_var', 'eje', 'time')

    # Separa trayectorias lados incluyendo LR en L y en R repetido
    daTodos = nfa.separa_trayectorias_lados(daTodos)

    # L = daTodos.sel(n_var=daTodos["n_var"].str.endswith("_L"))
    # R = daTodos.sel(n_var=daTodos["n_var"].str.endswith("_R"))
    # LR = daTodos.sel(n_var=daTodos["n_var"].str.endswith("_LR"))

    # # Incluye los LR en L y R
    # L = xr.concat([L, LR], dim="n_var")
    # R = xr.concat([R, LR], dim="n_var")

    # # Quita las terminaciones después de _
    # L["nom_var"] = ("nom_var", L["nom_var"].str.rstrip(to_strip="_L").data)
    # R["nom_var"] = ("nom_var", R["nom_var"].str.rstrip(to_strip="_R").data)
    # LR["nom_var"] = ("nom_var", LR["nom_var"].str.rstrip(to_strip="_LR").data)

    # daTodos = xr.concat([L, R, LR], pd.Index(["L", "R", "LR"], name="lado")).transpose(
    #     "Archivo", "nom_var", "lado", "eje", "time"
    # )
    daTodos.loc[
        dict(n_var=["HJC", "KJC", "AJC"])
    ] /= 10  # pasa las posiciones de los ejes a cm

    # ???daTodos = nfa.ajusta_angulos(daTodos)

    # Calcula vAngBiela
    if "vAngBiela" not in daTodos.n_var:
        vAngBiela = (
            xr.apply_ufunc(np.unwrap, daTodos.sel(n_var="AngBiela"))
            .differentiate(coord="time")
            .expand_dims(dim=["n_var"])
            .assign_coords(dict(n_var=["vAngBiela"]))
        )
        vAngBiela = np.rad2deg(vAngBiela)  # pasa de radianes a grados
        daTodos = xr.concat(
            [daTodos, vAngBiela], dim="n_var", join="left"
        )  # join left para que guarde el orden de coords lado

    daTodos.name = "Cinem"
    daTodos.attrs["freq"] = float(frec)
    daTodos.attrs["units"] = "deg"
    daTodos.time.attrs["units"] = "s"

    """
    # Pone el df en formato 1 nivel encabezados
    dfTodosArchivos.columns = dfTodosArchivos.columns.map("_".join).str.strip()
    dfTodosArchivos = dfTodosArchivos.rename(
        columns={"Archivo_": "Archivo", "time_": "time"}
    )
    """
    # EL DATAFRAME VA SIN vAngBiela
    print("Cargados los datos en {0:.3f} s \n".format(time.time() - timer))

    return daTodos


def acota_region_of_interest(daData):
    # si no está definida la región, busca inicio y final de datos (no nans)
    region_interest = np.nonzero(daData.isel(ID=0).sel(axis="x").values >= -1.0)[0]

    if region_interest.size:
        region_interest = np.vstack(
            (
                region_interest[np.diff(np.hstack((-np.inf, region_interest))) > 0 + 1],
                region_interest[np.diff(np.hstack((region_interest, np.inf))) > 0 + 1],
            )
        ).T

    return region_interest[0]


def calcula_angulos_desde_trajec(
    daData, modelo_completo=False, verbose=False, region_interest=None
) -> xr.DataArray:
    """
    Calcula ángulos de articulaciones a partir de las trayectorias.
    Paso intermedio de calcular matrices de rotación
    region_of_interest: de momento solo para calcular vAngBiela cuando empieza con nans
    """
    print("Calculando matrices de rotación...")
    timer_procesa = time.perf_counter()

    dsRlG = nfa.calcula_bases(daData, modelo_completo)  # daData=daDatos.isel(ID=0))

    print("\nCalculando ángulos de segmentos...")
    dsAngSegments = nfa.calcula_angulos_segmentos(dsRlG, verbose=verbose)

    print("\nCalculando ángulos articulares...")
    dsAngArtics = nfa.calcula_angulos_articulaciones(
        dsRlG, modelo_completo, verbose=verbose
    )
    # dsAngArtics['AngArtHip_L'].plot.line(x='time', row='ID', hue='axis')

    # Calcula ángulos pedales
    if "Pedal_A" in daData.n_var:
        dsAngPedal = np.arctan2(
            daData.sel(n_var="Pedal_A", axis="z")
            - daData.sel(n_var="Pedal_P", axis="z"),
            daData.sel(n_var="Pedal_A", axis="y")
            - daData.sel(n_var="Pedal_P", axis="y"),
        )  # .to_dataset(name="AngPedal")

    elif "Meta5" in daData.n_var:  # modelo antiguo
        dsAngPedal = np.arctan2(
            daData.sel(n_var="Toe", axis="z") - daData.sel(n_var="TalonSup", axis="z"),
            daData.sel(n_var="Toe", axis="y") - daData.sel(n_var="TalonSup", axis="y"),
        )  # .to_dataset(name="AngPedal")
    L = dsAngPedal.sel(side="L").drop_vars("side")
    L.name = "AngPedal_L"
    R = dsAngPedal.sel(side="R").drop_vars("side")
    R.name = "AngPedal_R"
    dsAngPedal = xr.merge([L, R])
    dsAngPedal = np.rad2deg(dsAngPedal)

    # Calcula ángulo de biela. Primero ejes pedales modelo actual con Pedal_A y Pedal_P, el antiguo con Meta5
    if "Pedal_A" in daData.n_var:
        daEjePedal = (
            daData.sel(n_var=["Pedal_A", "Pedal_P"])
            .mean(dim="n_var")
            .expand_dims({"n_var": ["PosPedal"]}, axis=0)  # Añade dimensión n_var
        )
    elif "Meta5" in daData.n_var:
        daEjePedal = daData.sel(n_var="Meta5").expand_dims(
            {"n_var": ["PosPedal"]}, axis=0
        )  # Añade dimensión n_var

    # TODO: Si hace falta, por aquí seleccionar qué variables se usan para calcular angBiela (modelo_eje_biela='bilateral')
    dsAngBiela = calcula_ang_biela(
        daEje1=daEjePedal.sel(side="R"), daEje2=daEjePedal.sel(side="L")
    ).to_dataset(
        name="AngBiela_LR"
    )  # .expand_dims({"n_var": ["AngBiela"]}, axis=0)  # Añade dimensión n_var
    # dsAngBiela["AngBiela_LR"].isel(ID=0).plot.line(x='time')
    dsAngBiela.attrs["freq"] = daData.freq

    # Calcula velocidad angular biela
    dsvAngBiela = calcula_vAng_biela(dsAngBiela, region_interest)

    daAngles = (
        xr.merge([dsAngSegments, dsAngArtics, dsAngPedal, dsAngBiela, dsvAngBiela])
        .to_array()
        .rename({"variable": "n_var"})
        .transpose(..., "axis", "time")
    )

    if "side" in daData.coords:
        daAngles = nfa.separa_trayectorias_lados(daData=daAngles)

    daAngles.name = "Angles"
    daAngles.attrs["units"] = "deg"
    daAngles.attrs["freq"] = daData.freq

    # daAngles.sel(n_var='AngArtHip').plot.line(x='time', row='ID', col='axis', hue='side')

    print(
        f"Tiempo total en procesar {len(daData.ID)} registros: {time.perf_counter() - timer_procesa:.3f} s."
    )

    return daAngles


def calcula_ang_biela(daEje1, daEje2) -> xr.DataArray:
    """
    Calcula ángulo de biela. Variable unidimensional, en los ejes se guardan distintas versiones,
    con distinto sistema de referencias.
    """
    # Con xarray directamente
    x = (
        np.arctan2(
            daEje1.sel(axis="y") - daEje2.sel(axis="y"),
            daEje1.sel(axis="z") - daEje2.sel(axis="z"),
        )
        + np.pi
    )  # cero con L arriba , PI con R arriba
    y = np.arctan2(
        daEje1.sel(axis="y") - daEje2.sel(axis="y"),
        daEje1.sel(axis="z") - daEje2.sel(axis="z"),
    )  # mínimo con L arriba, paso por cero con R arriba
    z = np.arctan2(
        daEje1.sel(axis="z") - daEje2.sel(axis="z"),
        daEje1.sel(axis="y") - daEje2.sel(axis="y"),
    )  # mínimo con L delante, paso por cero con R delante

    daAngBiela = (
        (
            xr.concat([x, y, z], dim="axis").assign_coords(axis=["x", "y", "z"])
            # .expand_dims({'side':['LR']},
            #                     axis=0) #Añade dimensión
        )
        .squeeze("n_var")
        .drop_vars("n_var")
    )
    # Pasar a grados no funciona bien en Nexus

    """
    x.plot.line(x='time', row='ID')
    daAngBiela.plot.line(x='time', row='ID', col='axis')
    """

    '''
    def calc_ang_biela_aux(eje1, eje2):
        # print(eje1)
        # print(eje1.shape, eje2.shape)
        #eje1 y eje2 son ejes pedales R y L o el de la biela según se requiera
        AngBiela = np.zeros(eje1.shape)
        #Todo en radianes porque en Nexus parece que si se pone en grados lo unwrappea
        
        AngBiela[:,0] = np.arctan2((eje1[:, 1] - eje2[:, 1]), (eje1[:, 2] - eje2[:, 2])) + np.pi  #cero con L arriba , PI con R arriba
        AngBiela[:,1] = np.arctan2((eje1[:, 1] - eje2[:, 1]), (eje1[:, 2] - eje2[:, 2])) #mínimo con L arriba, paso por cero con R arriba
        AngBiela[:,2] = np.arctan2((eje1[:, 2] - eje2[:, 2]), (eje1[:, 1] - eje2[:, 1])) #mínimo con L delante, paso por cero con R delante

        if AngBiela.shape[0]==0:
            AngBiela = np.zeros((NumFrames,3))
            print('No se ha podido crear el angulo AngBiela')

        return AngBiela


        """
        eje1= daEje1[0,0]
        eje2= daEje2[0,0]
        """
        daAngBiela = xr.apply_ufunc(calc_ang_biela_aux, daEje1, daEje2,
                        input_core_dims=[['time', 'axis'], ['time', 'axis']],
                        output_core_dims=[['time', 'axis']],
                        #exclude_dims=set(('axis',)),
                        dask='parallelized',
                        vectorize=True
                        )
        #daAngBiela.plot.line(x='time', row='ID', col='axis')
    '''
    # daAngBiela = daAngBiela.drop_vars('n_var')
    daAngBiela.name = "AngBiela_LR"
    daAngBiela.attrs["units"] = "rad"

    return daAngBiela


def calcula_vAng_biela(dsAngBiela, region_interest) -> xr.Dataset:
    # np.unwrap NO ADMITE NANs por eso hay que acotarlo
    if region_interest is None:
        # si no está definida la región, busca inicio y final de datos (no nans)
        region_interest = acota_region_of_interest(dsAngBiela["AngBiela_LR"])

    """
    # Para vAngBiela cuando empieza con nans
    from detect_onset import detect_onset
    region_interest = detect_onset(dsAngBiela["AngBiela_LR"].isel(ID=0).sel(axis="x"))[
        0
    ]
    """
    dsvAngBiela = xr.full_like(dsAngBiela, np.nan)  # .transpose(..., 'axis', 'time')
    dsvAngBiela.loc[
        dict(
            time=slice(
                region_interest[0] / dsAngBiela.freq,
                (region_interest[1] - 1) / dsAngBiela.freq,
            )
        )
    ] = (
        xr.apply_ufunc(
            np.unwrap,
            dsAngBiela.isel(time=slice(region_interest[0], region_interest[1])),
        ).differentiate(coord="time")
        # xr.apply_ufunc(np.unwrap, dsAngBiela.dropna("time")).differentiate(coord="time")
        # .expand_dims(dim=['n_var'])
        # .assign_coords(dict(n_var=['vAngBiela']))
    )
    dsvAngBiela = dsvAngBiela.rename({"AngBiela_LR": "vAngBiela_LR"})
    # dsvAngBiela['vAngBiela_LR'].sel(axis='x').plot()

    # Ajuste para los primeros y últimos datos con extrapolación
    if region_interest[1] - region_interest[0] > 200: # comprueba si hay suficiente nº de datos
        dsvAngBiela["vAngBiela_LR"].loc[
            dict(
                time=slice(
                    (region_interest[0]) / dsAngBiela.freq,
                    (region_interest[0] + 6) / dsAngBiela.freq,
                )
            )
        ] = (
            dsvAngBiela["vAngBiela_LR"]
            .isel(time=slice(region_interest[0] + 7, region_interest[0] + 100))
            .interp(
                time=np.arange(region_interest[0], region_interest[0] + 7)
                / dsAngBiela.freq,
                method="slinear",
                kwargs={"fill_value": "extrapolate"},
            )
        )
        # RELLENA CON MISMO DATO dsvAngBiela["vAngBiela_LR"].isel(time=region_interest[0] + 6, axis=0)

        #TODO: AJUSTAR MÁRGENES FOTOGRAMAS PARA QUE NO SE SALGA DE LO QUE HAYA
        # try: # fallaría si hay num datos pequeño
            
        dsvAngBiela["vAngBiela_LR"].loc[
            dict(
                time=slice(
                    (region_interest[1] - 10) / dsAngBiela.freq,
                    (region_interest[1] - 1) / dsAngBiela.freq,
                )
            )
        ] = (
            dsvAngBiela["vAngBiela_LR"]
            .isel(time=slice(region_interest[1] - 100, region_interest[1] - 10))
            .interp(
                time=np.arange(region_interest[1] - 10, region_interest[1])
                / dsAngBiela.freq,
                method="slinear",
                kwargs={"fill_value": "extrapolate"},
            )
        )
        # except:
        #     pass
        # RELLENA CON MISMO DATO dsvAngBiela["vAngBiela_LR"].isel(time=region_interest[1] - 10, axis=0)

    # Añade media y SD en ejes y, z
    dsvAngBiela["vAngBiela_LR"].loc[dict(axis="y")] = (
        dsvAngBiela["vAngBiela_LR"]
        .sel(axis="x")
        .isel(time=slice(int(0.5 * dsAngBiela.freq), int(-0.5 * dsAngBiela.freq)))
        .mean("time")
    )
    dsvAngBiela["vAngBiela_LR"].loc[dict(axis="z")] = (
        dsvAngBiela["vAngBiela_LR"]
        .sel(axis="x")
        .isel(time=slice(int(0.5 * dsAngBiela.freq), int(-0.5 * dsAngBiela.freq)))
        .std("time")
    )
    dsvAngBiela = np.rad2deg(dsvAngBiela)  # pasar a grados??
    # dsvAngBiela['vAngBiela_LR'].isel(ID=0).plot.line(x='time')

    return dsvAngBiela


def calcula_variables_posicion(
    daData, modelo_eje_biela="bilateral", verbose=False, region_interest=None
) -> xr.DataArray:
    """
    Calcula variables de marcadores a partir de las trayectorias.
    modelo_eje_biela = 'derecha', 'izquierda' o 'bilateral'. Define cómo se calculará el marcador del eje de la biela, a partir de los dos pies o con uno de ellos
    """

    print("\nCalculando variables de marcadores...")
    timer_procesa = time.perf_counter()

    # Une variable ejes pedales para poder verlos junto a los ángulos

    # ----Copia posiciones ejes articulares
    daEjeArtic = daData.sel(n_var=["HJC", "KJC", "AJC"])

    # ----Calcula eje pedales.Con modelo actual con Pedal_A y Pedal_P, el antiguo con Meta5
    if "Pedal_A" in daData.n_var:
        daEjePedal = (
            daData.sel(n_var=["Pedal_A", "Pedal_P"])
            .mean(dim="n_var")
            .expand_dims({"n_var": ["PosPedal"]}, axis=0)  # Añade dimensión n_var
        )
    elif "Meta5" in daData.n_var:
        daEjePedal = daData.sel(n_var="Meta5").expand_dims(
            {"n_var": ["PosPedal"]}, axis=0
        )  # Añade dimensión n_var

    # daEjePedal.isel(n_var=0).plot.line(x='time', row='ID', col='axis')

    # ----Calcula eje biela
    if modelo_eje_biela == "bilateral":
        # Calcula el punto medio entre lado L y R y promedio a lo largo del tiempo
        daEjeBiela = (
            daEjePedal.sel(side=["L", "R"])
            .mean(dim=["side", "time"])
            .assign_coords(n_var=["EjeBiela"])
            .expand_dims({"side": ["LR"]}, axis=0)  # Añade dimensión
        )

        # daEjeBiela.assign_coords(ID=range(len(daEjeBiela.ID))).isel(n_var=0).plot(x='ID', col='axis', sharey=False)

        # Expande la dimensión tiempo repitiendo
        daEjeBiela = xr.concat([daEjePedal, daEjeBiela], dim="n_var")

        # Repite valor LR en L y en R
        daEjeBiela.loc[dict(side=["L", "R"])] = daEjeBiela.loc[dict(side="LR")]
        # Se queda con lado L y R
        daEjeBiela = daEjeBiela.sel(n_var="EjeBiela", side=["L", "R"])

        # daEjeBiela.plot.line(x='time', row='ID', col='axis')

    # TODO: #NO ADAPTADO A XARRAY
    else:
        raise Exception('Sólo se ha implementado el modelo de biela "bilateral"')
        try:
            # ==============================================================================
            # Calcula el eje del pedalier a partir de 3 puntos del pedal
            # ==============================================================================
            if modeloEjeBiela == "izquierda":
                if EjePedal_L.shape[0] != 0:
                    marcProvis = EjePedal_L
                else:
                    marcProvis = Left_Meta

            elif modeloEjeBiela == "derecha":
                if EjePedal_R.shape[0] != 0:
                    marcProvis = EjePedal_R
                else:
                    marcProvis = Right_Meta

            centro = []
            for fot in range(RegionOfInterest[0] + 100, RegionOfInterest[1] - 100, 100):
                centro.append(
                    define_circle(
                        marcProvis[fot, 1:3],
                        marcProvis[fot + 30, 1:3],
                        marcProvis[fot + 60, 1:3],
                    )
                )
            centro = np.array(centro).mean(
                axis=0
            )  # calcula la media de todos los centros que ha calculado

            paraEjeX = ((EjePedal_L + EjePedal_R) / 2).mean(axis=0)[0]
            ejeBiela = np.full((NumFrames, 3), [paraEjeX, centro[0], centro[1]])
            if ejeBiela.shape[0] == 0:
                ejeBiela = np.zeros((NumFrames, 3))
                print("No se ha podido crear el marcador EjeBiela con", modeloEjeBiela)
            modeledName = "EjeBiela"
            if modeledName not in vicon.GetModelOutputNames(SubjectName):
                vicon.CreateModeledMarker(SubjectName, modeledName)
            vicon.SetModelOutput(SubjectName, modeledName, ejeBiela.T, exists)
        except:
            print(
                'No se ha podido calcular el eje de la biela con el modelo "derecha o izquierda".'
            )

    # ---- Calcula KOPS
    daLengthKops = (
        (daEjeArtic.sel(n_var="KJC") - daEjePedal)
        # .expand_dims("n_var")
        .assign_coords(n_var=["LengthKops"])
        .copy()
        .transpose(..., "axis", "time")  # ("n_var", "side", "ID", "time", "axis")
    )
    # Mete la coordenada importante (y) en eje x y los demás vacíos
    daLengthKops.loc[dict(axis="x")] = daLengthKops.sel(axis="y").values
    daLengthKops.loc[dict(axis=["y", "z"])] = 0.0

    # ---- Calcula longitud muslos, piernas y pies
    daDistSegMuslo = (np.sqrt(((daEjeArtic.sel(n_var="HJC") - daEjeArtic.sel(n_var="KJC"))**2).sum('axis'))
                .expand_dims({"n_var":["LengthMuslo"], 'axis':['x', 'y', 'z']})                
                .transpose(..., "axis", "time")
                .copy()
                 )
    # Sustituye ceros por nan para calcular media y std
    daDistSegMuslo = daDistSegMuslo.where(daDistSegMuslo != 0.0)
    daDistSegMuslo.loc[dict(axis="y")] = daDistSegMuslo.sel(axis="x").mean('time')
    daDistSegMuslo.loc[dict(axis="z")] = daDistSegMuslo.sel(axis="x").std('time')
    
    daDistSegPierna = (np.sqrt(((daEjeArtic.sel(n_var="KJC") - daEjeArtic.sel(n_var="AJC"))**2).sum('axis'))
                .expand_dims({"n_var":["LengthPierna"], 'axis':['x', 'y', 'z']})                
                .transpose(..., "axis", "time")
                .copy()
                 )
    daDistSegPierna = daDistSegPierna.where(daDistSegPierna != 0.0)
    daDistSegPierna.loc[dict(axis="y")] = daDistSegPierna.sel(axis="x").mean('time')
    daDistSegPierna.loc[dict(axis="z")] = daDistSegPierna.sel(axis="x").std('time')
    
    # Elige el marcador extremo del pie según el modelo
    if 'Meta' in daData.n_var.values:
        punta_pie = daData.sel(n_var="Meta")
    elif 'Meta5' in daData.n_var.values:
        punta_pie = daData.sel(n_var=["Meta5", 'Meta1']).mean('n_var')
        
    daDistSegPie = (np.sqrt(((punta_pie - daData.sel(n_var="TalonSup"))**2).sum('axis'))
                .expand_dims({"n_var":["LengthPie"], 'axis':['x', 'y', 'z']})                
                .transpose(..., "axis", "time")
                .copy()
                 )
    daDistSegPie = daDistSegPie.where(daDistSegPie != 0.0)    
    daDistSegPie.loc[dict(axis="y")] = daDistSegPie.sel(axis="x").mean('time')
    daDistSegPie.loc[dict(axis="z")] = daDistSegPie.sel(axis="x").std('time')
    

    # ---- Concatena resultados
    daPos = xr.concat(
        [daEjeArtic, daEjePedal, daEjeBiela, daLengthKops, daDistSegMuslo, daDistSegPierna, daDistSegPie], dim="n_var", join="left"
    )  # join left para que guarde el orden de coords lado
    # daEjePedal.isel(ID=0, n_var=0, axis=0).plot.line(x='time')

    print(
        f"Procesados {len(daData.ID)} registros en {time.perf_counter() - timer_procesa:.3f} s."
    )

    return daPos


def busca_eventos_biela(daData: xr.DataArray, region_interest:list=None, show=False) -> xr.DataArray:
    """busca posiciones de biela"""

    if region_interest is None:
        # si no está definida la región, busca inicio y final de datos (no nans)
        region_interest = acota_region_of_interest(
            daData.sel(n_var="AngBiela", side="L")
        )

    var_ang = daData.sel(
        n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle", "AngBiela"]
    ).isel(ID=0, time=slice(region_interest[0], region_interest[1]))

    # Vertical
    daProvis_L = (
        var_ang.sel(side="L").biomxr.detect_events(
            func_events=stsp.detect_onset_detecta_aux,
            reference_var=dict(n_var="AngBiela", axis="y"),
            # discard_phases_ini=0,
            discard_phases_end=1,
            # n_phases=12,
            # include_first_next_last=True,
            **dict(
                threshold=0.0,
                n_above=2,
                event_ini=1,
            ),
            show=show,
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    )
    daProvis_R = (
        var_ang.sel(side="R").biomxr.detect_events(
            func_events=stsp.detect_onset_detecta_aux,
            reference_var=dict(n_var="AngBiela", axis="y"),
            # discard_phases_ini=0,
            discard_phases_end=1,
            # n_phases=12,
            # include_first_next_last=True,
            **dict(
                threshold=0.0,
                n_above=2,
                event_ini=0,
                show=show,
            ),
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    )
    daIndPedalSup = (
        xr.concat([daProvis_L, daProvis_R], dim="side").sel(
            n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"], axis="x"
        )
        + region_interest[0]
    )
    daIndPedalSup.name = "sup"
    # El inferior es igual cambiado side
    daIndPedalInf = (
        xr.concat([daProvis_R, daProvis_L], dim="side")
        .sel(n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"], axis="x")
        .assign_coords(side=["L", "R"])
    ) + region_interest[0]
    daIndPedalInf.name = "inf"

    # Horizontal
    daProvis_L = (
        var_ang.sel(side="L").biomxr.detect_events(
            func_events=stsp.detect_onset_detecta_aux,
            reference_var=dict(n_var="AngBiela", axis="z"),
            # discard_phases_ini=0,
            discard_phases_end=1,
            **dict(
                threshold=0.0,
                n_above=2,
                event_ini=0,
                show=show,
            ),
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    )
    daProvis_R = (
        var_ang.sel(side="R").biomxr.detect_events(
            func_events=stsp.detect_onset_detecta_aux,
            reference_var=dict(n_var="AngBiela", axis="z"),
            # discard_phases_ini=0,
            discard_phases_end=1,
            **dict(
                threshold=0.0,
                n_above=2,
                event_ini=1,
                show=show,
            ),
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    )
    daIndPedalAnt = (
        xr.concat([daProvis_L, daProvis_R], dim="side").sel(
            n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"], axis="x"
        )
        + region_interest[0]
    )
    daIndPedalAnt.name = "ant"
    # El post es igual invirtiendo side
    daIndPedalPost = (
        xr.concat([daProvis_R, daProvis_L], dim="side")
        .sel(n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"], axis="x")
        .assign_coords(side=["L", "R"])
    ) + region_interest[0]
    daIndPedalPost.name = "post"

    # Unifica todos en un mismo ds
    dsEvtPosPedal = xr.merge(
        [
            daIndPedalSup,
            daIndPedalInf,
            daIndPedalAnt,
            daIndPedalPost,
        ],
    )
    """
    daIndGlobal = xr.concat(
        [
            daIndPedalSup,
            daIndPedalInf,
            daIndPedalAnt,
            daIndPedalPost,
        ],
        dim="criterio",
    ).assign_coords(
        criterio=[
            "pedal_sup",
            "pedal_inf",
            "pedal_ant",
            "pedal_post",
        ]
    )
    """
    return dsEvtPosPedal

def busca_eventos_max_min(daData: xr.DataArray, evt_pedal: Optional[Union[int, xr.DataArray]]=None, mean_pedal_freq:int=None, region_interest:list=None, show=False) -> xr.DataArray:
    # Si no viene una frecuencia media, la calcula
    if mean_pedal_freq is None:
        if evt_pedal is None:
            mean_pedal_freq = 90
        else:    
            mean_pedal_freq = (
            daData.freq
            / (evt_pedal.dropna(dim="n_event").astype(int).sel(n_var="AngArtKnee", side="L").diff("n_event"))
            * 60
            ).mean().data
        
    if region_interest is None:
        # si no está definida la región, busca inicio y final de datos (no nans)
        region_interest = acota_region_of_interest(
            daData.sel(n_var="AngBiela", side="L")
        )
        
    
    var_ang = daData.sel(
        n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"], axis="x"
    ).isel(ID=0, time=slice(region_interest[0], region_interest[1]))

    # ---- Busca picos máximos de extensión de cada articulación (invierte)    
    daIndMaxExt = (        
        (-var_ang).biomxr.detect_events(
            func_events=stsp.find_peaks_aux,  # bfa.stsp.detect_peaks,
            # reference_var=dict(n_var="AngBiela", axis="x"),
            # discard_phases_ini=0,
            # discard_phases_end=0,
            **dict(
                xSD=0.1,
                distance = (1 / (mean_pedal_freq / 60) / 1.6) * daData.freq,
                show=show,
            ),
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    ) + region_interest[0]
    daIndMaxExt.name = 'max_ext'
    """
    daIndMaxExt = (
        t=time.perf_counter()
        for i in range(500):
            var_ang.biomxr.detect_events(
                func_events=detect_peaks,  # bfa.stsp.detect_peaks,
                # reference_var=dict(n_var="AngBiela", axis="x"),
                # discard_phases_ini=0,
                # discard_phases_end=0,
                **dict(
                    valley=True,
                    mph=80,
                    mpd=(1 / (mean_pedal_freq / 60) / 1.6) * daData.freq,
                    show=False,
                ),
            )
        print(time.perf_counter() - t)
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    ) + region_interest[0]
    """
    
    # ---- Busca picos máximos de flexión de cada articulación    
    daIndMaxFlex = (
        var_ang.biomxr.detect_events(
            func_events=stsp.find_peaks_aux,  # bfa.stsp.detect_peaks,
            # reference_var=dict(n_var="AngBiela", axis="x"),
            # discard_phases_ini=0,
            # discard_phases_end=0,
            **dict(
                xSD=0.1,
                distance = (1 / (mean_pedal_freq / 60) / 1.6) * daData.freq,
                show=show,
            ),
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    ) + region_interest[0]
    daIndMaxFlex.name = 'max_flex'
    """
    daIndMaxFlex = (
        var_ang.biomxr.detect_events(
            func_events=detect_peaks,  # bfa.stsp.detect_peaks,
            # reference_var=dict(n_var="AngBiela", axis="x"),
            # discard_phases_ini=0,
            # discard_phases_end=0,
            **dict(
                mph=-80,
                mpd=(1 / (mean_pedal_freq / 60) / 1.6) * daData.freq,
                show=True,
            ),
        )
        # .isel(ID=0)
        # .sel(n_var="AngArtKnee", axis="x")
    ) + region_interest[0]
    """
    
    return xr.merge([daIndMaxExt, daIndMaxFlex])


def discretiza_ciclos(data, events, n_var, side, num_reps_excluir=1) -> xr.DataArray:
    """Repite el mismo valor de ángulo en todo el ciclo.
    Lo mantiene durante el siguiente ciclo.
    La función se usa con xr.apply_ufunc
    """

    events = events[~np.isnan(events)].astype(int)
    discret = np.full(data.shape, np.nan)

    # print(n_var, side, events.shape, data.shape)

    try:
        # transfiere los índices a datos discretos
        discr_data = data[events, 0]

        discret = np.full(data.shape, np.nan)

        # Repite valor de cada ciclo el nº de veces que corresponde
        rep = np.diff(events)
        discret[events[0] : events[-1], 0] = np.repeat(discr_data[:-1], rep)

        """
        for i in range(len(events) - 1):
            discret[events[i] : events[i + 1], 0] = discr_data[i]
            # daReturn.loc[dict(axis="x")][idx[i] : idx[i + 1]] = discr_data[i]
        """
        # ajusta el primer y el último tramo
        discret[0 : events[0], 0] = discr_data[0]
        discret[events[-1] : len(data), 0] = discr_data[-1]

        # incluye promedio y sd de las repeticiones centrales (quita num_reps_excluir primeras y num_reps_excluir últimas)
        discret[:, 1] = discr_data[num_reps_excluir:-num_reps_excluir].mean()
        discret[:, 2] = discr_data[num_reps_excluir:-num_reps_excluir].std()
    except:
        print(f"No se ha podido calcular el ángulo {n_var} lado {side}")

    return discret


# =============================================================================
# %% Funciones complementarias SPM1D
# =============================================================================
def calcula_spm1d(df):
    # Compara lados con spm1d
    # Primero selecciona el df con la variable concreta y el lado concreto
    dfProvis_L = df.query(
        "side==@df.side.unique()[0]"
    )  # con unique para evitar problemas nomenclatura L Izq, R Der
    dfProvis_R = df.query("side==@df.side.unique()[1]")

    # Después ordena el df poniendo en columnas cada repetición y lo pasa a numpy traspuesto para el spm1d
    YL = (
        dfProvis_L.pivot(index="AngBielaInRepe", columns="phase", values=["value"])
        .to_numpy()
        .T
    )
    YR = (
        dfProvis_R.pivot(index="AngBielaInRepe", columns="phase", values=["value"])
        .to_numpy()
        .T
    )
    # plt.plot(YL.T)
    # plt.plot(YR.T)

    try:  # si alguna variable es cero, se peta
        t = spm1d.stats.ttest_paired(
            YL, YR
        )  # YL[repes[0]:repes[1]], YR[repes[0]:repes[1]])
        ti = t.inference(alpha=0.05, two_tailed=True)
    except:
        ti = None

    # ti.plot(ax=ax[1])
    return ti


# Post hoc
def plot_clusters(spmi, y=0, ax=None, print_p=True) -> None:
    if spmi == None:  # .h0reject:
        # print('No hay clusters')
        return

    ax = ax if ax else plt

    for cluster in spmi.clusters:
        ax.axvspan(
            cluster.endpoints[0], cluster.endpoints[1], facecolor="gray", alpha=0.3
        )
        if print_p:
            ax.text(
                x=np.mean(cluster.endpoints),
                y=spmi.zstar - 1.5,
                s="p = {0:.3f}".format(cluster.P),
                horizontalalignment="center",
            )


def get_region_of_interest(spmi):
    region_of_interest = np.zeros(spmi.z.shape[0])
    for cluster in spmi.clusters:
        region_of_interest[
            int(round(cluster.endpoints[0])) : int(round(cluster.endpoints[1]))
        ] = 1
    return region_of_interest.astype(bool)


# =============================================================================


# =============================================================================
# %% Segmenta por repeticiones, versión cinem y EMG
# =============================================================================


def segmenta_ModeloBikefitting_xr_cinem(
    daData, num_cortes=None, show=False
) -> xr.DataArray:
    # from detecta import detect_peaks
    # from cortar_repes_ciclicas import detect_onset_aux

    print(f"Segmentando {len(daData.ID)} archivos.")
    # Es necesario separar lado L y R porque usan criterios distintos de corte
    ###CORTES A PARTIR DE ANG BIELA
    print("Cortando lado L...")
    # daData.isel(ID=0).sel(side="R", n_var='AngBiela').plot.line(x='time')

    daL = stsp.slice_time_series(daData.sel(side="L"),
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela", axis="y"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=1, show=show),
    )
    """COMPROBAR, FALLA CON DATAARRAY NON NANS
    daData.sel(side="L").biomxr.slice_time_series(
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela", axis="y"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=1, show=show),
    )
    """
    print("Cortando lado R...")
    daR = stsp.slice_time_series(daData.sel(side="R"),
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela", axis="y"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=0, show=show),
    )
    
    """daData.sel(side="R").biomxr.slice_time_series(
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela", axis="y"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=0, show=show),
    )"""
    """
    print('Cortando lados R y LR...')
    daR = stsp(data=daData.sel(side=['R', 'LR']), func_events=stsp.detect_onset_detecta_aux,
              reference_var=dict(n_var='AngBiela', side='LR', axis='y'),
              discard_phases_ini=1, discard_phases_end=0, n_phases=num_cortes,
              include_first_next_last=True, 
              **dict(threshold=0.0, n_above=2, event_ini=0, show=show)
              ).slice_time_series()
    """
    # daL = cts(daData.sel(side='L'), func_cortes=cts.detect_onset_aux, var_referencia=dict(n_var='AngBiela', axis='y'),
    #           descarta_corte_ini=1, descarta_corte_fin=0, num_cortes=num_cortes,
    #           **dict(threshold=0.0, n_above=2, corte_ini=1, show=show)).corta_repes()
    # daR = cts(daData.sel(side=['R', 'LR']), func_cortes=cts.detect_onset_aux, var_referencia=dict(n_var='AngBiela', side='LR', axis='y'),
    #           descarta_corte_ini=1, num_cortes=num_cortes, descarta_corte_fin=0, **dict(threshold=0.0, n_above=2, corte_ini=0, show=show)).corta_repes()

    # daL = corta_repes_xr(daData.sel(side='L'), func_cortes=detect_onset_aux, var_referencia=dict(n_var='AngBiela', axis='y'), descarta_rep_ini=1, descarta_rep_fin=0, num_repes=num_repes, **dict(threshold=0.0, n_above=2, corte_ini=1, show=show))
    # daR = corta_repes_xr(daData.sel(side=['R', 'LR']), func_cortes=detect_onset_aux, var_referencia=dict(n_var='AngBiela', side='LR', axis='y'), descarta_rep_ini=1, num_repes=num_repes, descarta_rep_fin=0, **dict(threshold=0.0, n_above=2, corte_ini=0, show=show))

    # #Mete variables LR en lado R (antiguo)
    # daLR = daR.sel(side='LR')
    # daR = daR.sel(side='R')

    # daL.isel(ID=0).sel(n_var=['AngArtHip', 'AngArtKnee', 'AngArtAnkle']).plot.line(x='time', col='axis', row='n_var', sharey=False)
    # daR.isel(ID=0).sel(n_var=['AngArtHip', 'AngArtKnee', 'AngArtAnkle']).plot.line(x='time', col='axis', row='n_var', sharey=False)
    # daLR.isel(ID=0).dropna(dim='n_var', how='all').plot.line(x='time', col='axis', row='n_var', sharey=False)

    """
    ####CORTES A PARTIR DE DIFERENCIA PEDALES
    #Incorpora variable distancia entre pedales
    dPedales = ((daData.sel(n_var='PosPedal', side='R') - daData.sel(n_var='PosPedal', side='L'))
                .expand_dims(dim=['n_var', 'side'])
                .assign_coords(dict(n_var=['dPedales'], side=['LR']))
              )
    daData = xr.concat([daData, dPedales], dim='n_var', join='left') #join left para que guarde el orden de coords lado
    
    daL = (corta_repes_xr(daData.sel(side=['L', 'LR']), func_cortes=detect_peaks, var_referencia=dict(n_var='dPedales', side='LR', axis='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=True, mph=-100, mpd=50, show=show))
           .sel(side='L')
           )
    daR = corta_repes_xr(daData.sel(side=['R', 'LR']), func_cortes=detect_peaks, var_referencia=dict(n_var='dPedales', side='LR', axis='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=False, mph=100, mpd=50, show=show))#.sel(side='R')
    """

    """    
    daSegment = (xr.concat([daL, daR, daLR], pd.Index(['L','R', 'LR'], name='side'))
                 .transpose('ID', 'n_var', 'side', 'axis', 'phase', 'time')
                 #.isel(n_var=slice(0,-1))#quita la variable creada distancia pedales
                 )
    """

    daSegment = (
        xr.concat([daL, daR], pd.Index(["L", "R"], name="side"))
        # .transpose("ID", "n_var", "side", "axis", "phase", "time")
        # .isel(n_var=slice(0,-1))#quita la variable creada distancia pedales
    )

    return daSegment


# #Versión con dataframe
# def ApilaFactores_Segmenta_ModeloBikefitting_cinem(dfArchivo, show=False):
#     from detect_peaks import detect_peaks #from detecta import detect_peaks
#     from cortar_repes_ciclicas import detect_onset_aux

#     #Se crean variables de apoyo con los nombres de cada ámbito
#     nomVarsContinuas_coord = nomVarsContinuas250_coords
#     nomVarsContinuas_L_coord = [i for i in nomVarsContinuas250_coords if '_L' in i and '_LR' not in i]
#     nomVarsContinuas_R_coord = [i for i in nomVarsContinuas250_coords if '_R' in i]
#     nomVarsContinuas_LR_coord = [i for i in nomVarsContinuas250_coords if '_LR' in i and 'AngBiela_LR_y' not in i] # en versiones modernas de Pandas no hace falta quitar AngBiela_LR_y, pero en la antigua sí


#     #Es necesario separar lado L y R porque usan criterios distintos de corte
#     #Corta izquierdo
#     dfL = dfArchivo.drop(dfArchivo.filter(regex='|'.join(['_R_', '_LR_'])).columns, axis=1).assign(**{'AngBiela_LR_y' : dfArchivo['AngBiela_LR_y'], 'vAngBiela_LR_x' : dfArchivo['vAngBiela_LR_x']}) #añade las bielas al final
#     dfL = corta_repes(dfL, func_cortes=detect_peaks, frec=Frec,  col_factores='ID', col_referencia='AngBiela_LR_y', col_variables=nomVarsContinuas_L_coord+['vAngBiela_LR_x'], descarta_rep_ini=1, descarta_rep_fin=0, incluye_primero_siguiente=True, **dict(valley=True, show=show))

#     #Corta derecho y LR a la vez
#     dfR = dfArchivo.drop(dfArchivo.filter(regex='|'.join(['_L_'])).columns, axis=1) #al derecho no hace falta añadir bielas
#     dfR = corta_repes(dfArchivo, func_cortes=detect_onset_aux, frec=Frec,  col_factores='ID', col_referencia='AngBiela_LR_y', col_variables=nomVarsContinuas_R_coord+nomVarsContinuas_LR_coord+['vAngBiela_LR_x'], descarta_rep_ini=1, descarta_rep_fin=0, **dict(threshold=0.0, corte_ini=0, n_above=2, show=show))
#     dfR = dfR.loc[:,~dfR.columns.duplicated()] #quita un vAngBiela duplicado
#     dfLR = dfR

#     dfR = dfR.drop(dfR.filter(regex='_LR_').columns, axis=1).assign(**{'AngBiela_LR_y' : dfR['AngBiela_LR_y'], 'vAngBiela_LR_x' : dfR['vAngBiela_LR_x']}) #añade AngBiela y vAngBiela
#     dfLR = dfLR.drop(list(dfLR.filter(regex='_R_').columns), axis=1)
#     #dfLR = dfLR.loc[:,~dfLR.columns.duplicated()] #quita un vAngBiela duplicado


#     #sns.relplot(data=dfL, x='time_repe', y='AngArtKnee_L_x',  units='repe', estimator=None, hue='repe',  kind='line')

#     #Lo pone en formato long y añade columna con AngBiela en grados
#     vars_factores=['ID', 'repe', 'AngBiela_LR_y', 'time', 'time_repe']
#     dfFactor = pd.concat([pd.melt(dfL, id_vars=vars_factores, value_vars=nomVarsContinuas_L_coord, var_name='NomVarOrig', value_name='value'),
#                           pd.melt(dfR, id_vars=vars_factores, value_vars=nomVarsContinuas_R_coord, var_name='NomVarOrig', value_name='value'),
#                           #pd.melt(dfLR, id_vars=dfLR.columns[:4], value_vars=dfLR.columns[-4:], var_name='NomVarOrig', value_name='value')
#                           pd.melt(dfLR, id_vars=vars_factores, value_vars=nomVarsContinuas_LR_coord, var_name='NomVarOrig', value_name='value')
#                          ])
#     dfFactor = dfFactor.assign(**{'AngBielaInRepe':np.rad2deg(dfFactor['AngBiela_LR_y']+np.pi),
#                             'n_var':dfFactor['NomVarOrig'].str.split('_', expand=True)[0],
#                             'side':dfFactor['NomVarOrig'].str.split('_', expand=True)[1],
#                             'axis':dfFactor['NomVarOrig'].str.split('_', expand=True)[2]
#                             }).reindex(columns=['ID', 'n_var', 'side', 'axis', 'repe', 'AngBiela_LR_y', 'AngBielaInRepe', 'time', 'time_repe', 'value'])

#     #dfFactor = dfFactor.reindex(columns=['ID', 'n_var', 'side', 'axis', 'repe', 'AngBiela_y', 'AngBielaInRepe', 'time', 'time_repe'])

#     #sns.relplot(data=dfFactor.query('axis=="x"'), x='time_repe', y='value', col='side', row='n_var', units='repe', estimator=None, hue='repe',  kind='line', facet_kws={'sharey': False})


#     return dfFactor

# =============================================================================


# --------------------------------------------


def segmenta_ModeloBikefitting_xr_EMG(
    daDatos, num_cortes=12, show=False
) -> xr.DataArray:
    # from detecta import detect_peaks
    # from cortar_repes_ciclicas import detect_onset_aux
    print(f"Segmentando {len(daDatos.ID)} archivos.")
    daL = stsp(
        data=daDatos.sel(side="L"),
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=1, show=show),
    ).slice_time_series()
    daR = stsp(
        data=daDatos.sel(side="R"),
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=0, show=show),
    ).slice_time_series()

    # daL = cts(daDatos.sel(side='L'), func_cortes=cts.detect_onset_aux,
    #           var_referencia=dict(n_var='AngBiela'),
    #           descarta_corte_ini=1, num_cortes=num_cortes, descarta_corte_fin=0,
    #           **dict(threshold=0.0, n_above=2, corte_ini=1, show=show)).corta_repes()
    # daR = cts(daDatos.sel(side='R'), func_cortes=cts.detect_onset_aux,
    #           var_referencia=dict(n_var='AngBiela'),
    #           descarta_corte_ini=1, num_cortes=num_cortes, descarta_corte_fin=0,
    #           **dict(threshold=0.0, n_above=2, corte_ini=0, show=show)).corta_repes()

    # daL = corta_repes_xr(daDatos.sel(side='L'), func_cortes=detect_onset_aux, var_referencia=dict(n_var='AngBiela'), descarta_corte_ini=1, descarta_corte_fin=0, num_cortes=num_cortes, **dict(threshold=0.0, n_above=2, corte_ini=1, show=show))
    # daR = corta_repes_xr(daDatos.sel(side='R'), func_cortes=detect_onset_aux, var_referencia=dict(n_var='AngBiela'), descarta_corte_ini=1, descarta_corte_fin=0, num_cortes=num_cortes, **dict(threshold=0.0, n_above=2, corte_ini=0, show=show))

    # daL.isel(ID=0).plot.line(x='time', hue='corte', row='n_var', sharey=False)

    """
    #Con PosPedal
    daL = corta_repes_xr(daDatos.sel(side='L'), func_cortes=detect_peaks, var_referencia=dict(n_var='dPedal_z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=True, mph=-100, mpd=50, show=show))
    daR = corta_repes_xr(daDatos.sel(side='R'), func_cortes=detect_peaks, var_referencia=dict(n_var='dPedal_z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=False, mph=100, mpd=50, show=show))#.sel(side='R')
    """

    daSegment = (
        xr.concat([daL, daR], pd.Index(["L", "R"], name="side")).transpose(
            "ID", "n_var", "side", "phase", "time"
        )
        # .isel(n_var=slice(0,-2))#quita las variables cinemáticas
    )

    # daSegment.isel(ID=0).plot.line(x='time', hue='repe', row='n_var', col='side', sharey=False)

    return daSegment


# =============================================================================
# %% Función para normalizar a 360 datos cada ciclo. Se pasa después de la función Segmenta
# =============================================================================
def normaliza_t_aux(
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


def normaliza_biela_360_xr(
    daData, base_norm_horiz="time", show=False
) -> xr.DataArray:  # recibe da de daTodos. Versión con numpy
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

    daNorm = xr.apply_ufunc(
        normaliza_t_aux,
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
# %% Calcular datos análisis
# =============================================================================
def calcula_descrip_antropo_global(daData, daDataNorm):
    """Calcula variables antropométricas global (en todos los IDs)"""
    # TODO: calcular ángulo inclinación eje cadera-rodilla-tobillo frontal a 90º biela
    """
    for n, _ in daData.groupby("ID"):
        calcula_descrip_antropo(dagb=daData.sel(ID=n), daNorm=daDataNorm)
    """
    if 'LengthMuslo' in daData.n_var:
        muslo = (daDataNorm.sel(n_var="LengthMuslo", AngBielaInRepe=180, axis='x')
                 .drop_vars('axis')
                 .mean('phase')
        )
    else:
        # Distancias anatómicas, se mide en posición 180º de biela    
        HJC = daDataNorm.sel(side=["L", "R"], n_var="HJC", AngBielaInRepe=180)
        KJC = daDataNorm.sel(side=["L", "R"], n_var="KJC", AngBielaInRepe=180)
        AJC = daDataNorm.sel(side=["L", "R"], n_var="AJC", AngBielaInRepe=180)
        muslo = np.sqrt(
            (HJC.sel(axis="x") - KJC.sel(axis="x")) ** 2
            + (HJC.sel(axis="y") - KJC.sel(axis="y")) ** 2
            + (HJC.sel(axis="z") - KJC.sel(axis="z")) ** 2
        ).mean(dim="phase").drop_vars(["AngBielaInRepe", 'AngBielaInRepe_rad'])
    muslo.name='muslo'
    
    if 'LengthPierna' in daData.n_var:
        pierna = (daDataNorm.sel(n_var="LengthPierna", AngBielaInRepe=180, axis='x')
                 .drop_vars('axis')
                 .mean('phase')
        )
    else:
        pierna = np.sqrt(
            (KJC.sel(axis="x") - AJC.sel(axis="x")) ** 2
            + (KJC.sel(axis="y") - AJC.sel(axis="y")) ** 2
            + (KJC.sel(axis="z") - AJC.sel(axis="z")) ** 2
        ).mean(dim="phase").drop_vars(["AngBielaInRepe", 'AngBielaInRepe_rad'])
    pierna.name='pierna'
    
    if 'LengthPie' in daData.n_var:
        pie = (daDataNorm.sel(n_var="LengthPie", AngBielaInRepe=180, axis='x')
                 .drop_vars('axis')
                 .mean('phase')
        )
    else:
        print('No se ha podido calcular la longitud del pie')
        pie = xr.full_like((daDataNorm.isel(n_var=0).sel(AngBielaInRepe=180, axis='x')
                 .drop_vars('axis')
                 .mean('phase')
        ), np.nan)
    pie.name='pie'
    """
    muslo2=np.sqrt((
        (daDataNorm.sel(n_var=["KJC", "HJC"], AngBielaInRepe=180).mean(dim="phase")
         .diff('n_var'))**2)
        .sum('axis')
        .isel(n_var=0)
    )
    muslo2-muslo
    """
    # np.sqrt((HJC.sel(axis='x')-KJC.sel(axis='x'))**2 + (HJC.sel(axis='y')-KJC.sel(axis='y'))**2 + (HJC.sel(axis='z')-KJC.sel(axis='z'))**2).plot.line(x='AngBielaInRepe', hue='phase', col='side')
    # np.sqrt((KJC.sel(axis='x')-AJC.sel(axis='x'))**2 + (KJC.sel(axis='y')-AJC.sel(axis='y'))**2 + (KJC.sel(axis='z')-AJC.sel(axis='z'))**2).plot.line(x='AngBielaInRepe', hue='repe', col='side')

    """
    #Distancias anatómicas
    HJC = daData.sel(side=['L','R'], n_var='HJC')
    KJC = daData.sel(side=['L','R'], n_var='KJC')
    AJC = daData.sel(side=['L','R'], n_var='AJC')
    #np.sqrt((HJC.sel(axis='x')-KJC.sel(axis='x'))**2 + (HJC.sel(axis='y')-KJC.sel(axis='y'))**2 + (HJC.sel(axis='z')-KJC.sel(axis='z'))**2).plot.line(x='time', hue='side')
    muslo = np.sqrt((HJC.sel(axis='x')-KJC.sel(axis='x'))**2 + (HJC.sel(axis='y')-KJC.sel(axis='y'))**2 + (HJC.sel(axis='z')-KJC.sel(axis='z'))**2).mean(dim='time')
    pierna = np.sqrt((KJC.sel(axis='x')-AJC.sel(axis='x'))**2 + (KJC.sel(axis='y')-AJC.sel(axis='y'))**2 + (KJC.sel(axis='z')-AJC.sel(axis='z'))**2).mean(dim='time')
    """
       
    # dfAntropo = (
    #     xr.concat([muslo, pierna], pd.Index(["muslo", "pierna"], name="segmento"))
    #     # .transpose('ID', 'segmento', 'side')
    #     .to_dataframe()
    #     .reset_index()
    #     .drop(columns=["axis"])
    #     .reindex(columns=["ID", "segmento", "side", muslo.name])
    # )

    # Calcula ancho caderas y pedales y lo añade
    ancho_caderas = daData.sel(n_var="HJC", axis="x").diff('side').mean('time').drop_vars(["n_var"])
    ancho_caderas.name='ancho_caderas'
    
    # (daData.sel(n_var='HJC', side='R', axis='x') - daData.sel(n_var='HJC', side='L', axis='x')).plot.line(x='time')
    # (daDataNorm.isel(ID=0).sel(n_var='HJC', side='R', axis='x') - daDataNorm.isel(ID=0).sel(n_var='HJC', side='L', axis='x')).plot.line(x='AngBielaInRepe', hue='repe')

    LONG_CLUSTER_PEDALES = 100  # TODO: ¿HACERLO CON MARCADOR METAS?
    ancho_pedales = daData.sel(n_var="PosPedal", axis="x").diff('side').mean('time').drop_vars(["n_var"])-(LONG_CLUSTER_PEDALES * 2)
    ancho_pedales.name='ancho_pedales'
        
    # segments = xr.concat([muslo, pierna], pd.Index(["muslo", "pierna"], name="n_var"))
    # anchos =(xr.concat([ancho_caderas, ancho_pedales], pd.Index(["ancho_caderas", "ancho_pedales"], name="n_var"))
    # .assign_coords(side=['LR'])
    # )
    
    dfAntropo1 = (
        xr.concat([muslo, pierna, pie], pd.Index(["muslo", "pierna", 'pie'], name="n_var"))
        # .transpose('ID', 'segmento', 'side')
        .to_dataframe('value')
        .reset_index()
        # .drop(columns=["axis"])
        # .reindex(columns=["ID", "particip", "n_var", "side", muslo.name])
        # .rename(columns={muslo.name:'value'})
    )    
    dfAntropo2 = (xr.concat([ancho_caderas, ancho_pedales], pd.Index(["ancho_caderas", "ancho_pedales"], name="n_var"))
    .assign_coords(side=['LR'])
    .to_dataframe('value').reset_index()
    # .reindex(columns=["ID", "particip", "n_var", "side", ancho_caderas.name])
    # .rename(columns={ancho_caderas.name:'value'})
    )
    
    dfAntropo = (pd.concat([dfAntropo1, dfAntropo2], axis=0).reset_index(drop=True)
            # .reindex(columns=['n_var', 'ID', "particip", 'side', 'value'])
    )
        
    # OBTIENE DATOS DISCRETOS SEGÚN POSICIÓN BIELA DISCRETA
    vars_group = [n for n in daDataNorm.dims if n not in ['phase', 'AngBielaInRepe']]
    dfResumen_discreto = (daDataNorm.sel(n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle", "AngSegPELVIS"], AngBielaInRepe=[0, 90, 180])
        .to_dataframe(name="value")
        .groupby(vars_group)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    
    # DIFERENCIA ENTRE LADOS
    rmse = ((daDataNorm.sel(n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"], axis="x").mean(dim="phase")
              .diff('side'))**2).isel(side=0).drop_vars('side').mean(dim="AngBielaInRepe") ** 0.5
    """
    L = daDataNorm.sel(
        side="L",
        n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"],
        axis="x",
    ).mean(dim="phase")
    R = daDataNorm.sel(
        side="R",
        n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"],
        axis="x",
    ).mean(dim="phase")
    rmse = (((L - R) ** 2).mean(dim="AngBielaInRepe")) ** 0.5
    """
    dfCompLados = (
        rmse.to_dataframe()
        .reset_index()
        .drop(columns=["axis"])
        # .reindex(columns=["ID", "particip", "n_var", daData.name])
        .rename(columns={"Cinem": "RMSE"})
    )

    return dfResumen_discreto, dfAntropo, dfCompLados


def calcula_descrip_antropo(dagb, daNorm):
    """Calcula variables antropométricas individual (en un ID)"""
    # TODO: calcular ángulo inclinación eje cadera-rodilla-tobillo frontal a 90º biela
    daNorm = daNorm.sel(ID=dagb.ID.data)
    # Distancias anatómicas, se mide en posición 180º de biela
    HJC = daNorm.sel(side=["L", "R"], n_var="HJC", AngBielaInRepe=180
    )
    KJC = daNorm.sel(side=["L", "R"], n_var="KJC", AngBielaInRepe=180
    )
    AJC = daNorm.sel(side=["L", "R"], n_var="AJC", AngBielaInRepe=180
    )
    muslo = np.sqrt(
        (HJC.sel(axis="x") - KJC.sel(axis="x")) ** 2
        + (HJC.sel(axis="y") - KJC.sel(axis="y")) ** 2
        + (HJC.sel(axis="z") - KJC.sel(axis="z")) ** 2
    ).mean(dim="phase")
    pierna = np.sqrt(
        (KJC.sel(axis="x") - AJC.sel(axis="x")) ** 2
        + (KJC.sel(axis="y") - AJC.sel(axis="y")) ** 2
        + (KJC.sel(axis="z") - AJC.sel(axis="z")) ** 2
    ).mean(dim="phase")
    # np.sqrt((HJC.sel(axis='x')-KJC.sel(axis='x'))**2 + (HJC.sel(axis='y')-KJC.sel(axis='y'))**2 + (HJC.sel(axis='z')-KJC.sel(axis='z'))**2).plot.line(x='AngBielaInRepe', hue='repe', col='side')
    # np.sqrt((KJC.sel(axis='x')-AJC.sel(axis='x'))**2 + (KJC.sel(axis='y')-AJC.sel(axis='y'))**2 + (KJC.sel(axis='z')-AJC.sel(axis='z'))**2).plot.line(x='AngBielaInRepe', hue='repe', col='side')

    """
    #Distancias anatómicas
    HJC = dagb.sel(side=['L','R'], n_var='HJC')
    KJC = dagb.sel(side=['L','R'], n_var='KJC')
    AJC = dagb.sel(side=['L','R'], n_var='AJC')
    #np.sqrt((HJC.sel(axis='x')-KJC.sel(axis='x'))**2 + (HJC.sel(axis='y')-KJC.sel(axis='y'))**2 + (HJC.sel(axis='z')-KJC.sel(axis='z'))**2).plot.line(x='time', hue='side')
    muslo = np.sqrt((HJC.sel(axis='x')-KJC.sel(axis='x'))**2 + (HJC.sel(axis='y')-KJC.sel(axis='y'))**2 + (HJC.sel(axis='z')-KJC.sel(axis='z'))**2).mean(dim='time')
    pierna = np.sqrt((KJC.sel(axis='x')-AJC.sel(axis='x'))**2 + (KJC.sel(axis='y')-AJC.sel(axis='y'))**2 + (KJC.sel(axis='z')-AJC.sel(axis='z'))**2).mean(dim='time')
    """

    # Calcula ancho caderas y pedales y lo añade
    ancho_caderas = dagb.sel(n_var="HJC", axis="x").diff('side').mean('time').isel(side=0)
    """
    ancho_caderas = abs(
        dagb.sel(n_var="HJC", side="R", axis="x").data
        - dagb.sel(n_var="HJC", side="L", axis="x").data
    ).mean()
    """
    # (dagb.sel(n_var='HJC', side='R', axis='x') - dagb.sel(n_var='HJC', side='L', axis='x')).plot.line(x='time')
    # (daNorm.isel(ID=0).sel(n_var='HJC', side='R', axis='x') - daNorm.isel(ID=0).sel(n_var='HJC', side='L', axis='x')).plot.line(x='AngBielaInRepe', hue='repe')

    LONG_CLUSTER_PEDALES = 100  # TODO: ¿HACERLO CON MARCADOR METAS?
    ancho_pedales = dagb.sel(n_var="PosPedal", axis="x").diff('side').mean('time').isel(side=0)-(LONG_CLUSTER_PEDALES * 2)
    """
    ancho_pedales = (
        -LONG_CLUSTER_PEDALES * 2
        + abs(
            dagb.sel(n_var="PosPedal", side="R", axis="x")
            - dagb.sel(n_var="PosPedal", side="L", axis="x")
        )
        .mean()
        .data
        / 10
    )
    """
    # (dagb.sel(n_var='PosPedal', side='R', axis='x') - dagb.sel(n_var='PosPedal', side='L', axis='x')).plot.line(x='time')
    
    dfAntropo = (
        xr.concat([muslo, pierna], pd.Index(["muslo", "pierna"], name="segmento"))
        # .transpose('ID', 'segmento', 'side')
        .to_dataframe()
        .reset_index()
        .drop(columns=["axis"])
        .reindex(columns=["ID", "segmento", "side", dagb.name])
    )
    
    dfAntropo.loc[dfAntropo.shape[0]] = [
        dfAntropo.iloc[0, 0],
        "ancho_caderas",
        "LR",
        ancho_caderas.values,
    ]
    dfAntropo.loc[dfAntropo.shape[0]] = [
        dfAntropo.iloc[0, 0],
        "ancho_pedales",
        "LR",
        ancho_pedales.values,
    ]
    
    # Presenta núm. repes de cada archivo
    # dfFactorTodosNorm.groupby('ID')['repe'].max()

    # OBTIENE DATOS DISCRETOS SEGÚN POSICIÓN BIELA DISCRETA
    resumen_discreto = (daNorm.sel(n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle", "AngSegPELVIS"], AngBielaInRepe=[0, 90, 180])
    .to_dataframe(name="value")
    .groupby(["ID", "n_var", "AngBielaInRepe", "axis", "side"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    
    # Ángulos en posiciones de biela
    """dfNorm = (
        daNorm.to_dataframe(name="value").dropna()['value'].reset_index()
    )
    resumen_discreto = (
        dfNorm.query("AngBielaInRepe==[0, 90, 180]")
        .query('n_var==["AngArtHip", "AngArtKnee", "AngArtAnkle", "AngSegPELVIS"]')
        .groupby(["n_var", "AngBielaInRepe", "axis", "side"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )"""
    # Nombre todo junto con guiones
    # resumen90=dfFactorTodosNorm.query('AngBielaInRepe==90').groupby(['ID', 'n_var', 'side', 'axis'])['value'].agg(['mean', 'std'])
    # resumen90.index = resumen90.index.map('_'.join).str.strip() #junta las categorías del index por si hace falta

    # Con xarray por si acaso
    # resumen_da = daTodosCortNorm.sel(n_var=['AngArtHip', 'AngArtKnee', 'AngArtAnkle', 'AngSegPELVIS'], AngBielaInRepe=[0,90,180]).mean(dim='repe').to_dataframe(name='resumen')

    # DIFERENCIA ENTRE LADOS
    L = daNorm.sel(
        side="L",
        n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"],
        axis="x",
    ).mean(dim="phase")
    R = daNorm.sel(
        side="R",
        n_var=["AngArtHip", "AngArtKnee", "AngArtAnkle"],
        axis="x",
    ).mean(dim="phase")
    # L.sel(axis='x').plot.line(x='AngBielaInRepe', hue='n_var')
    # np.sqrt((L-R)**2).sel(axis='x').plot.line(x='AngBielaInRepe', hue='n_var')
    rmse = (((L - R) ** 2).mean(dim="AngBielaInRepe")) ** 0.5
    dfCompLados = (
        rmse.to_dataframe()
        .reset_index()
        .drop(columns=["axis"])
        .reindex(columns=["ID", "n_var", dagb.name])
        .rename(columns={"Cinem": "RMSE"})
    )

    return resumen_discreto, dfAntropo, dfCompLados


# =============================================================================
# %% GRÁFICAS
# =============================================================================
# dfLateral = dfFactorTodosNorm.groupby('ID').get_group('01_Carrillo_WU')

# Función para ensemble averages a mano
from matplotlib.path import Path as matPath
from matplotlib.patches import PathPatch


def draw_error_band(ax, x, y, err, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    x = x.values
    y = y.values
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]], [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), matPath.LINETO)
    codes[0] = codes[len(xp)] = matPath.MOVETO
    path = matPath(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


def RealizaGraficas_cinem(
    daGraf,
    repes=[0, 10],
    nomvars=["AngArtKnee", "AngArtHip", "AngArtAnkle"],
    ejes=["x", "y"],
    tipo_graf=["lados_lin", "lados_circ", "coordinacion", "planoXY"],
    compara_lados_graf=False,
    ensemble_avg=True,
    carpeta_guardar=False,
) -> None:  # por mejorar: que se puedan descartar repeticiones intermedias
    import itertools

    """
    Nomenclatura:
        '_A_': Gráfica Ángulo de articulación en escala grados biela.
        '_AP_': Gráfica Ángulo en coordenadas polares.
        '_APV_': Gráfica Ángulo en coordenadas polares con velocidad biela como color.
        '_AA_': Gráfica Ángulo/Ángulo
        '_I_': Gráfica individual (no compara L y R)
        '_IP_': Gráfica individual en coordenadas polares (no compara L y R)
        '_PP_': Gráfica posición/posición (plano) eje articular
    """
    # FALTA QUE SELECCIONE REPES SUELTAS

    # Si viene como xarray lo pasa a dataframe
    if isinstance(daGraf, xr.DataArray):
        dfGraf = daGraf.to_dataframe(name="value").dropna().reset_index()

    # dfLateral.loc[:, 'side'] = dfLateral.loc[:, 'side'].replace({'L':'I', 'R':'D'})

    numreps = dfGraf["phase"].max()
    if repes == None:  # si no se indica nada, muestra todas las repeticiones
        repes = [0, numreps]
    else:  # comprueba si tiene tantas repeticiones como se le pide
        if repes[1] > numreps:
            repes[1] = numreps
    rango = np.arange(
        repes[0], repes[1]
    ).tolist()  # rango de repeticiones para todas las gráficas
    dfGraf = dfGraf.query("phase==@rango")

    # ---------------------------------------
    # Crea carpeta donde guardará las gráficas
    if carpeta_guardar:
        carpeta_output = Path(carpeta_guardar) / "Figs"
        if not carpeta_output.exists():
            carpeta_output.mkdir()

    if ensemble_avg == True or ensemble_avg == "completo":
        estim = "mean"
        ci = "sd"
        unit = None
        alpha = 1
        # g = sns.lineplot(data=dfGraf.query('n_var == @nomvar & eje ==@eje & repe==@rango'), x='AngBielaInRepe', y='value', hue='side', ci='sd', lw=1, palette=['r', 'lime'], ax=ax)
    else:
        estim = None
        ci = ("ci", 95)
        unit = "phase"
        alpha = 0.5

    # Función genérica para guardar todas las gráficas creadas
    def guarda_grafica(nom="A", carpeta_guardar=None):
        if carpeta_guardar is None:
            print("No se ha especificado la carpeta de guardado")
            return

        # print(carpeta_guardar)
        # print(formatoImagenes)
        ruta_fig = carpeta_guardar.joinpath(dfGraf["ID"].iloc[0] + nom).with_suffix(
            formatoImagenes
        )
        # ruta_fig = CarpetaSesion + 'Figs\\' + ArchivoActivo+ '_A_' + nfa.traduce_variables(nomvar) + '_' + eje + formatoImagenes
        if formatoImagenes == ".pdf":
            with PdfPages(ruta_fig) as pdf_pages:
                pdf_pages.savefig(fig)
        else:
            fig.savefig(ruta_fig, dpi=300)

    rc = {"axes.titlesize": "large"}  # títulos de cada gráfica
    with sns.plotting_context(context="paper", rc=rc):
        # Gráficas ÁNGULO / BIELA repetición a repetición con variables por separado
        if "lados_lin" in tipo_graf:
            for nomvar, eje in itertools.product(nomvars, ejes):
                # print(nomvar,eje)
                titulo = "{0:s} {1:s}".format(
                    nfa.traduce_variables(nomvar), nfa.traduce_variables(eje)
                )

                # fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(8,6))#para incluir gráfica spm{t}
                fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

                # Compara lados con spm1d
                if compara_lados_graf:
                    ti = calcula_spm1d(dfGraf.query("n_var==@nomvar & axis==@eje"))

                    # Marca regiones con diferencia L y R
                    plot_clusters(ti, y=5, ax=ax, print_p=False)
                if ensemble_avg == "completo":
                    sns.lineplot(
                        data=dfGraf.query("n_var==@nomvar & axis==@eje"),
                        x="AngBielaInRepe",
                        y="value",
                        hue="side",
                        estimator=None,
                        errorbar=ci,
                        units="phase",
                        lw=0.25,
                        palette=["r", "lime"],
                        alpha=0.3,
                        zorder=2,
                        ax=ax,
                    )  #'darkgrey', 'darkgrey'
                g = sns.lineplot(
                    data=dfGraf.query("n_var==@nomvar & axis==@eje"),
                    x="AngBielaInRepe",
                    y="value",
                    hue="side",
                    estimator=estim,
                    errorbar=ci,
                    units=unit,
                    lw=1,
                    palette=["r", "lime"],
                    alpha=alpha,
                    zorder=3,
                    ax=ax,
                )
                # Ajusta formato gráfica
                # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))

                if nomvar == "AngArtKnee":
                    ax.axhspan(
                        25, 35, alpha=0.3, color="green"
                    )  # crea un rectángulo horizontal delimitado a mano
                    ax.text(
                        5,
                        31,
                        "Rango recomendado",
                        c="green",
                        alpha=0.6,
                        fontsize=5,
                        transform=ax.transData,
                    )
                    ax.text(
                        5,
                        27,
                        "máxima extensión",
                        c="green",
                        alpha=0.6,
                        fontsize=5,
                        transform=ax.transData,
                    )

                g.set_title(
                    titulo
                )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set(xlim=(0, 360), xticks=np.linspace(0, 360, 5))

                # Excepción para velocidad angular biela
                if "vAngBiela" in nomvar:
                    ylabel = "Velocidad angular (grados/s)"
                    # HACER MEDIA DE LOS DOS LADOS O SÓLO DE UNO?
                    media_vAng = dfGraf.query(
                        "n_var==@nomvar & axis==@eje"  # & side=="R"'
                    )["value"].mean()
                    ax.axhline(
                        y=media_vAng,
                        ls="-.",
                        lw=0.5,
                        dash_capstyle="round",
                        c="k",
                        alpha=0.5,
                    )
                    ax.text(
                        5,
                        media_vAng,
                        "Frecuencia media: {0:.1f} rpm".format(media_vAng * 60 / (360)),
                        va="bottom",
                        c="k",
                        alpha=0.6,
                        fontsize=5,
                        transform=ax.transData,
                    )

                else:
                    ylabel = "Ángulo (grados)"
                g.set(xlabel="Ángulo de biela (grados)", ylabel=ylabel)  # ($^\circ$)

                g.xaxis.grid(
                    True,
                    linestyle="dashed",
                    dashes=(5, 5),
                    dash_capstyle="round",
                    zorder=1,
                )

                # En la leyenda sustituye por en español
                custom_lines = [
                    Line2D(
                        [0], [0], color="r", lw=2, solid_capstyle="round", label="Izq"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="lime",
                        lw=2,
                        solid_capstyle="round",
                        label="Der",
                    ),
                ]
                g.legend(
                    handles=custom_lines, loc="best"
                )  # bbox_to_anchor=(.95,.95))#, bbox_transform=ax_ang.transAxes)

                plt.tight_layout()

                if carpeta_guardar:
                    guarda_grafica(
                        f"_A_{nfa.traduce_variables(nomvar)}_{eje}",
                        carpeta_guardar=carpeta_output,
                    )

                # plt.show()
        # ---------------------------------------

        # Gráfica ÁNGULO / BIELA Radial
        if "lados_circ" in tipo_graf:
            for nomvar, eje in itertools.product(nomvars, ejes):
                # print(nomvar,eje)
                titulo = "{0:s} {1:s}".format(
                    nfa.traduce_variables(nomvar), nfa.traduce_variables(eje)
                )

                fig, ax = plt.subplots(
                    figsize=(3, 3), subplot_kw=dict(projection="polar"), dpi=300
                )
                if ensemble_avg == "completo":
                    sns.lineplot(
                        data=dfGraf.query("n_var == @nomvar & axis ==@eje"),
                        x="AngBielaInRepe_rad",
                        y="value",
                        hue="side",
                        estimator=None,
                        errorbar=ci,
                        units="phase",
                        lw=0.25,
                        palette=["r", "lime"],
                        alpha=0.3,
                        ax=ax,
                    )  #'darkgrey', 'darkgrey'
                g = sns.lineplot(
                    data=dfGraf.query("n_var == @nomvar & axis ==@eje"),
                    x="AngBielaInRepe_rad",
                    y="value",
                    hue="side",
                    estimator=estim,
                    errorbar=ci,
                    units=unit,
                    lw=1,
                    palette=["r", "lime"],
                    alpha=alpha,
                    ax=ax,
                )

                # Pone el valor 0º arriba y giro en sentido horario
                g.set(theta_zero_location="N", theta_direction=-1)
                g.set(xlabel="", ylabel="")

                g.set_title(
                    titulo
                )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))

                # En la leyenda sustituye por en español
                custom_lines = [
                    Line2D(
                        [0], [0], color="r", lw=2, solid_capstyle="round", label="Izq"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="lime",
                        lw=2,
                        solid_capstyle="round",
                        label="Der",
                    ),
                ]
                g.legend(
                    handles=custom_lines, bbox_to_anchor=(0.0, 0.95)
                )  # , bbox_transform=ax_ang.transAxes)

                plt.tight_layout()
                
                if carpeta_guardar:
                    guarda_grafica(
                        f"_AP_{nfa.traduce_variables(nomvar)}_{eje}",
                        carpeta_guardar=carpeta_output,
                    )

                # plt.show()
        # ---------------------------------------

        # Gráfica ÁNGULO/ÁNGULO
        if "coordinacion" in tipo_graf:
            for eje in ejes:
                for parnomvar in itertools.combinations(nomvars, 2):
                    # print(parnomvar[0])
                    # rango=np.arange(repes[0],repes[1]).tolist()

                    # Adapta el dataframe para que tenga las dos variables en columnas
                    df = dfGraf.query("n_var==@parnomvar[0] & axis ==@eje")[
                        ["ID", "repe", "side", "axis", "AngBielaInRepe"]
                    ].reset_index(drop=True)
                    df[parnomvar[0]] = dfGraf.query(
                        "n_var==@parnomvar[0] & axis==@eje"
                    )["value"].reset_index(drop=True)
                    df[parnomvar[1]] = dfGraf.query(
                        "n_var==@parnomvar[1] & axis==@eje"
                    )["value"].reset_index(drop=True)

                    df_ = df
                    y = None
                    if ensemble_avg:
                        y = parnomvar[1]
                        df_ = df.groupby(["side", "AngBielaInRepe"])[
                            [parnomvar[0], parnomvar[1]]
                        ].mean()

                    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
                    # Dibuja líneas (media o repe a repe según ensemble_avg)
                    g = sns.lineplot(
                        data=df_,
                        x=parnomvar[1],
                        y=parnomvar[0],
                        units=unit,
                        hue="side",
                        estimator=None,
                        palette=["r", "lime"],
                        sort=False,
                        alpha=alpha,
                        ax=ax,
                    )

                    # Dibuja la posición en puntos críticos de cada ciclo
                    for posbiela in [0, 90, 180]:
                        posx = (
                            df.query('side=="L" & AngBielaInRepe==@posbiela')
                            .loc[:, parnomvar[1]]
                            .mean()
                        )
                        posy = (
                            df.query('side=="L" & AngBielaInRepe==@posbiela')
                            .loc[:, parnomvar[0]]
                            .mean()
                        )
                        ax.plot(
                            posx,
                            posy,
                            c="firebrick",
                            mfc="r",
                            marker="o",
                            ms=13,
                            alpha=0.7,
                            zorder=2,
                        )
                        ax.text(
                            posx,
                            posy,
                            str(posbiela),
                            c="firebrick",
                            fontsize=7,
                            fontweight="bold",
                            transform=ax.transData,
                            ha="center",
                            va="center",
                            zorder=3,
                        )

                        posx = (
                            df.query('side=="R" & AngBielaInRepe==@posbiela')
                            .loc[:, parnomvar[1]]
                            .mean()
                        )
                        posy = (
                            df.query('side=="R" & AngBielaInRepe==@posbiela')
                            .loc[:, parnomvar[0]]
                            .mean()
                        )
                        ax.plot(
                            posx,
                            posy,
                            c="limegreen",
                            mfc="lime",
                            marker="o",
                            ms=13,
                            alpha=0.7,
                            zorder=2,
                        )
                        ax.text(
                            posx,
                            posy,
                            str(posbiela),
                            c="limegreen",
                            fontsize=7,
                            fontweight="bold",
                            transform=ax.transData,
                            ha="center",
                            va="center",
                            zorder=3,
                        )

                    """
                    #Dibuja inicios de cada ciclo            
                    ax.plot(df.query('side=="L" & AngBielaInRepe==0').loc[:,parnomvar[1]].mean(), 
                            df.query('side=="L" & AngBielaInRepe==0').loc[:,parnomvar[0]].mean(), 
                            c='firebrick', mfc='r', marker='o', ms=10, alpha=0.5, zorder=2)
                    ax.plot(df.query('side=="R" & AngBielaInRepe==0').loc[:,parnomvar[1]].mean(), 
                            df.query('side=="R" & AngBielaInRepe==0').loc[:,parnomvar[0]].mean(), 
                            c='limegreen', mfc='lime', marker='o', ms=10, alpha=0.5, zorder=2)
                    """

                    # sombrea la zona de desviación típica si ensemble_avg no es False
                    if ensemble_avg:
                        # for nomvar, ax in g.axes_dict.items():
                        # print(nomvar)
                        err = np.sqrt(
                            df.query('side=="L"')
                            .groupby("AngBielaInRepe")
                            .std()[parnomvar[0]]
                            ** 2
                            + df.query('side=="L" & phase==@rango')
                            .groupby("AngBielaInRepe")
                            .std()[parnomvar[1]]
                            ** 2
                        )
                        # err = (df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                        draw_error_band(
                            ax,
                            df_.query('side=="L"')[parnomvar[1]],
                            df_.query('side=="L"')[parnomvar[0]],
                            err=err,
                            facecolor="r",
                            edgecolor="none",
                            alpha=0.2,
                            zorder=0,
                        )
                        err = np.sqrt(
                            df.query('side=="R"')
                            .groupby("AngBielaInRepe")
                            .std()[parnomvar[0]]
                            ** 2
                            + df.query('side=="R" & phase==@rango')
                            .groupby("AngBielaInRepe")
                            .std()[parnomvar[1]]
                            ** 2
                        )
                        # err = (df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                        draw_error_band(
                            ax,
                            df_.query('side=="R"')[parnomvar[1]],
                            df_.query('side=="R"')[parnomvar[0]],
                            err=err,
                            facecolor="lime",
                            edgecolor="none",
                            alpha=0.2,
                            zorder=0,
                        )

                        if ensemble_avg == "completo":
                            # print('dibuja repes sueltas en' + nomvar)
                            for rep in range(dfGraf.phase.max()):
                                ax.plot(
                                    df.query('side=="L" & phase==@rep').loc[
                                        :, parnomvar[1]
                                    ],
                                    df.query('side=="L" & phase==@rep').loc[
                                        :, parnomvar[0]
                                    ],
                                    c="r",
                                    lw=0.25,
                                    alpha=0.3,
                                    zorder=0,
                                )
                                ax.plot(
                                    df.query('side=="R" & phase==@rep').loc[
                                        :, parnomvar[1]
                                    ],
                                    df.query('side=="R" & phase==@rep').loc[
                                        :, parnomvar[0]
                                    ],
                                    c="limegreen",
                                    lw=0.25,
                                    alpha=0.3,
                                    zorder=0,
                                )

                    # Ajusta formato gráfica
                    # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                    g.set_title(
                        "Coordinación articular, vista {0:s}".format(
                            nfa.traduce_variables(eje)
                        )
                    )  #' {0:s}/{1:s} {2:s}'.format(nfa.traduce_variables(parnomvar[0]), nfa.traduce_variables(parnomvar[1]), nfa.traduce_variables(eje))) #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))

                    g.set(
                        xlabel=f"{nfa.traduce_variables(parnomvar[1])} (grados)",
                        ylabel=f"{nfa.traduce_variables(parnomvar[0])} (grados)",
                    )

                    # g.axis('equal') #hace los dos ejes proporcionales

                    # En la leyenda sustituye por en español
                    custom_lines = [
                        Line2D(
                            [0],
                            [0],
                            color="r",
                            lw=2,
                            solid_capstyle="round",
                            label="Izq",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="lime",
                            lw=2,
                            solid_capstyle="round",
                            label="Der",
                        ),
                    ]
                    g.legend(
                        handles=custom_lines, loc="best"
                    )  # bbox_to_anchor=(.95,.95))#, bbox_transform=ax_ang.transAxes)

                    plt.tight_layout()

                    if carpeta_guardar:
                        guarda_grafica(
                            f"_AA_{nfa.traduce_variables(parnomvar[0])}_{nfa.traduce_variables(parnomvar[1])}_{eje}",
                            carpeta_guardar=carpeta_output,
                        )

                    # plt.show()
            # ---------------------------------------

        # Gráficas individuales de una en una
        if "indiv_lin" in tipo_graf:
            for nomvar, eje in itertools.product(nomvars, ejes):
                # print(nomvar, eje)
                # nom=dfFactorTodosNorm.loc[dfFactorTodosNorm['side'].str.contains('LR'), 'n_var'].unique()#nombres de variables con LR
                if nomvar == "AngSegPELVIS" and eje == "y":
                    titulo = "Basculación lateral pelvis"
                else:
                    titulo = "{0:s} {1:s}".format(
                        nfa.traduce_variables(nomvar), nfa.traduce_variables(eje)
                    )

                fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
                if ensemble_avg == "completo":
                    sns.lineplot(
                        data=dfGraf.query("n_var == @nomvar & axis ==@eje"),
                        x="AngBielaInRepe",
                        y="value",
                        estimator=None,
                        errorbar=ci,
                        units="phase",
                        lw=0.25,
                        color="orange",
                        alpha=0.3,
                        zorder=1,
                        ax=ax,
                    )  #'lightgrey'
                g = sns.lineplot(
                    data=dfGraf.query("n_var == @nomvar & axis ==@eje"),
                    x="AngBielaInRepe",
                    y="value",
                    estimator=estim,
                    errorbar=ci,
                    units=unit,
                    lw=1,
                    color="orange",
                    alpha=alpha,
                    zorder=3,
                    ax=ax,
                )

                # Ajusta formato gráfica
                # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set_title(
                    titulo
                )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set(xlim=(0, 360), xticks=np.linspace(0, 360, 5))
                g.set(
                    xlabel="Ángulo de biela (grados)", ylabel="Ángulo (grados)"
                )  # ($^\circ$)

                g.xaxis.grid(
                    True,
                    linestyle="dashed",
                    dashes=(5, 5),
                    dash_capstyle="round",
                    zorder=1,
                )

                media = dfGraf.query("n_var == @nomvar & axis ==@eje")["value"].mean()
                ax.axhline(
                    y=media,
                    ls="-.",
                    lw=0.5,
                    dash_capstyle="round",
                    c="orange",
                    alpha=0.5,
                )
                ax.text(
                    3,
                    media,
                    "Posición central",
                    va="bottom",
                    c="orange",
                    alpha=0.6,
                    fontsize=6,
                    transform=ax.transData,
                )

                plt.tight_layout()

                if carpeta_guardar:
                    guarda_grafica(
                        f"_I_{nfa.traduce_variables(nomvar)}_{eje}",
                        carpeta_guardar=carpeta_output,
                    )

                # plt.show()
        # ---------------------------------------

        # Gráficas individuales de una en una circular
        if "indiv_circ" in tipo_graf:
            for nomvar, eje in itertools.product(nomvars, ejes):
                # print(nomvar, eje)
                # nom=dfFactorTodosNorm.loc[dfFactorTodosNorm['side'].str.contains('LR'), 'n_var'].unique()#nombres de variables con LR

                titulo = "{0:s} {1:s}".format(
                    nfa.traduce_variables(nomvar), nfa.traduce_variables(eje)
                )

                fig, ax = plt.subplots(
                    figsize=(3, 3), subplot_kw=dict(projection="polar"), dpi=300
                )
                if ensemble_avg == "completo":
                    sns.lineplot(
                        data=dfGraf.query("n_var == @nomvar & axis ==@eje"),
                        x="AngBielaInRepe_rad",
                        y="value",
                        estimator=None,
                        errorbar=95,
                        units="phase",
                        lw=0.25,
                        color="orange",
                        alpha=0.3,
                        ax=ax,
                    )  #'lightgrey'
                g = sns.lineplot(
                    data=dfGraf.query("n_var == @nomvar & axis ==@eje"),
                    x="AngBielaInRepe_rad",
                    y="value",
                    estimator=estim,
                    errorbar=ci,
                    units=unit,
                    lw=1,
                    color="orange",
                    alpha=alpha,
                    ax=ax,
                )

                # Ajusta formato gráfica
                # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))

                # Pone el valor 0º arriba y giro en sentido horario
                g.set(theta_zero_location="N", theta_direction=-1)

                g.set_title(
                    titulo
                )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set(xlabel="", ylabel="")

                plt.tight_layout()

                if carpeta_guardar:
                    guarda_grafica(
                        f"_IP_{nfa.traduce_variables(nomvar)}_{eje}",
                        carpeta_guardar=carpeta_output,
                    )

                # plt.show()
        # ---------------------------------------

        # Gráfica POSICIÓN / POSICIÓN ejes articulares
        if "planoXY" in tipo_graf:
            for nomvar in nomvars:
                for parejes in itertools.combinations(ejes, 2):
                    # print(nomvar, parejes)

                    # Adapta el dataframe para que tenga las dos variables en columnas
                    df = dfGraf.query("n_var==@nomvar & axis==@parejes[0]")[
                        ["ID", "phase", "side", "axis", "AngBielaInRepe"]
                    ].reset_index(drop=True)
                    var1 = nfa.traduce_variables("Eje " + parejes[0])
                    var2 = nfa.traduce_variables("Eje " + parejes[1])
                    df[var1] = dfGraf.query("n_var==@nomvar & axis==@parejes[0]")[
                        "value"
                    ].reset_index(drop=True)
                    df[var2] = dfGraf.query("n_var==@nomvar & axis==@parejes[1]")[
                        "value"
                    ].reset_index(drop=True)

                    if parejes[0] == "x":
                        try:  # solución provisional
                            offsetX = (
                                (
                                    daGraf.isel(ID=0).sel(
                                        n_var="PosPedal", axis="x", side="L"
                                    )
                                    + daGraf.isel(ID=0).sel(
                                        n_var="PosPedal", axis="x", side="R"
                                    )
                                )
                                / 2
                            ).mean(
                                dim=["AngBielaInRepe", "phase"]
                            ).data / 10  # lo pasa a cm
                        except:
                            offsetX = (
                                (
                                    daGraf.sel(n_var="PosPedal", axis="x", side="L")
                                    + daGraf.sel(n_var="PosPedal", axis="x", side="R")
                                )
                                / 2
                            ).mean(
                                dim=["AngBielaInRepe", "phase"]
                            ).data / 10  # lo pasa a cm

                        df[
                            var1
                        ] -= offsetX  # var1.query('side=="Der"').min()['var1'] #solo para ajustar el cero del eje x

                    # En la visión frontal invierte el eje x para que se vea de frente
                    if parejes == ("x", "z"):
                        print("")
                        # offsetX = ((daGraf.isel(ID=0).sel(n_var='PosPedal', axis='x', side='L') + daGraf.isel(ID=0).sel(n_var='PosPedal', axis='x', side='R')) / 2).mean(dim=['AngBielaInRepe', 'repe']).data
                        df[var1] = -df[var1]
                        # var1['var1']+=offsetX #var1.query('side=="Der"').min()['var1'] #solo para ajustar el cero del eje x
                    # En la visión superior invierte el eje x para que las coordenadas sean positivas
                    if parejes == ("x", "y"):
                        print("")
                        # var1['var1']-=var1.iloc[0,var1.columns.get_loc('var1')] #solo para ajustar el cero del eje

                    """
                    #En la visión frontal invierte el eje x para que se vea de frente                
                    if parejes==('x','z'):
                        df[var1]=-df[var1]
                        df[var1]-=df.query('side=="R"').min()[var1] #solo para ajustar el cero del eje x
                    #En la visión superior invierte el eje x para que las coordenadas sean positivas
                    if parejes==('x','y'):
                        df[var1]-=df.iloc[0,df.columns.get_loc(var1)] #solo para ajustar el cero del eje 
                    """

                    df_ = df
                    y = None
                    if ensemble_avg:
                        y = var2
                        df_ = df.groupby(["side", "AngBielaInRepe"])[
                            [var1, var2]
                        ].mean()

                    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
                    # Dibuja líneas (media o repe a repe según ensemble_avg)
                    g = sns.lineplot(
                        data=df_,
                        x=var1,
                        y=var2,
                        units=unit,
                        hue="side",
                        estimator=None,
                        palette=["r", "lime"],
                        sort=False,
                        alpha=alpha,
                        ax=ax,
                    )

                    # Dibuja la posición en puntos críticos de cada ciclo
                    for posbiela in [0, 90, 180]:
                        posx = (
                            df.query('side=="L" & AngBielaInRepe==@posbiela')
                            .loc[:, var1]
                            .mean()
                        )
                        posy = (
                            df.query('side=="L" & AngBielaInRepe==@posbiela')
                            .loc[:, var2]
                            .mean()
                        )
                        ax.plot(
                            posx,
                            posy,
                            c="firebrick",
                            mfc="r",
                            marker="o",
                            ms=13,
                            alpha=0.7,
                            zorder=2,
                        )
                        ax.text(
                            posx,
                            posy,
                            str(posbiela),
                            c="firebrick",
                            fontsize="x-small",
                            fontweight="bold",
                            transform=ax.transData,
                            horizontalalignment="center",
                            verticalalignment="center",
                            zorder=3,
                        )

                        posx = (
                            df.query('side=="R" & AngBielaInRepe==@posbiela')
                            .loc[:, var1]
                            .mean()
                        )
                        posy = (
                            df.query('side=="R" & AngBielaInRepe==@posbiela')
                            .loc[:, var2]
                            .mean()
                        )
                        ax.plot(
                            posx,
                            posy,
                            c="limegreen",
                            mfc="lime",
                            marker="o",
                            ms=13,
                            alpha=0.7,
                            zorder=2,
                        )
                        ax.text(
                            posx,
                            posy,
                            str(posbiela),
                            c="limegreen",
                            fontsize="x-small",
                            fontweight="bold",
                            transform=ax.transData,
                            horizontalalignment="center",
                            verticalalignment="center",
                            zorder=3,
                        )

                    # Línea eje central
                    ax.axvline(
                        x=0,
                        c="grey",
                        ls="--",
                        dashes=(5, 5),
                        dash_capstyle="round",
                        alpha=0.5,
                    )

                    # sombrea la zona de desviación típica si ensemble_avg no es False
                    if ensemble_avg:
                        # for nomvar, ax in g.axes_dict.items():
                        # print(nomvar)
                        err = np.sqrt(
                            df.query('side=="L"').groupby("AngBielaInRepe")[var1].std()
                            ** 2
                            + df.query('side=="L" & phase==@rango')
                            .groupby("AngBielaInRepe")[var2]
                            .std()
                            ** 2
                        )
                        # err = (df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                        draw_error_band(
                            ax,
                            df_.query('side=="L"')[var1],
                            df_.query('side=="L"')[var2],
                            err=err,
                            facecolor="r",
                            edgecolor="none",
                            alpha=0.2,
                            zorder=0,
                        )
                        err = np.sqrt(
                            df.query('side=="R"').groupby("AngBielaInRepe")[var1].std()
                            ** 2
                            + df.query('side=="R" & phase==@rango')
                            .groupby("AngBielaInRepe")[var2]
                            .std()
                            ** 2
                        )
                        # err = (df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                        draw_error_band(
                            ax,
                            df_.query('side=="R"')[var1],
                            df_.query('side=="R"')[var2],
                            err=err,
                            facecolor="lime",
                            edgecolor="none",
                            alpha=0.2,
                            zorder=0,
                        )

                        if ensemble_avg == "completo":
                            # print('dibuja repes sueltas en' + nomvar)
                            for rep in range(dfGraf.phase.max()):
                                ax.plot(
                                    df.query('side=="L" & phase==@rep').loc[:, var1],
                                    df.query('side=="L" & phase==@rep').loc[:, var2],
                                    c="r",
                                    lw=0.25,
                                    alpha=0.3,
                                    zorder=0,
                                )
                                ax.plot(
                                    df.query('side=="R" & phase==@rep').loc[:, var1],
                                    df.query('side=="R" & phase==@rep').loc[:, var2],
                                    c="limegreen",
                                    lw=0.25,
                                    alpha=0.3,
                                    zorder=0,
                                )

                    # En vista frontal rodilla coloca rango posición caderas-tobillos
                    if nomvar == "KJC" and parejes == ("x", "z"):
                        try:  # solución provisional para cuando se llama individualmente desde Nexus o desde el tratamiento global
                            eje_medio = (
                                -daGraf.isel(ID=0)
                                .sel(n_var=["HJC", "AJC"], axis="x")
                                .mean(dim=["AngBielaInRepe", "phase"])
                            )
                        except:
                            eje_medio = -daGraf.sel(
                                n_var=["HJC", "AJC"], axis="x"
                            ).mean(dim=["AngBielaInRepe", "phase"])
                        # Lado L
                        ax.axvspan(
                            eje_medio.sel(n_var="HJC", side="L") + offsetX,
                            eje_medio.sel(n_var="AJC", side="L") + offsetX,
                            alpha=0.3,
                            color="green",
                        )  # crea un rectángulo horizontal delimitado a mano
                        ax.text(
                            eje_medio.sel(n_var="AJC", side="L") + offsetX,
                            ax.get_ylim()[1],
                            "Eje tobillo",
                            ha="left",
                            va="top",
                            c="green",
                            alpha=0.6,
                            fontsize=6,
                            rotation="vertical",
                            transform=ax.transData,
                            zorder=5,
                        )
                        # ax.text(eje_medio.sel(n_var='HJC', side='L')+offsetX, daGraf.isel(ID=0).sel(n_var=['KJC'], axis='z', side='L').max(dim=['AngBielaInRepe', 'repe']), 'Eje cadera', ha='right', va='top', c='green', alpha=0.6, fontsize=6, rotation='vertical', transform=ax.transData, zorder=5)
                        ax.text(
                            eje_medio.sel(n_var="HJC", side="L") + offsetX,
                            ax.get_ylim()[1],
                            "Eje cadera",
                            ha="right",
                            va="top",
                            c="green",
                            alpha=0.6,
                            fontsize=6,
                            rotation="vertical",
                            transform=ax.transData,
                            zorder=5,
                        )

                        # Lado R
                        ax.axvspan(
                            eje_medio.sel(n_var="HJC", side="R") + offsetX,
                            eje_medio.sel(n_var="AJC", side="R") + offsetX,
                            alpha=0.3,
                            color="green",
                        )  # crea un rectángulo horizontal delimitado a mano
                        ax.text(
                            eje_medio.sel(n_var="AJC", side="R") + offsetX,
                            ax.get_ylim()[1],
                            "Eje tobillo",
                            ha="right",
                            va="top",
                            c="green",
                            alpha=0.6,
                            fontsize=6,
                            rotation="vertical",
                            transform=ax.transData,
                            zorder=5,
                        )
                        ax.text(
                            eje_medio.sel(n_var="HJC", side="R") + offsetX,
                            ax.get_ylim()[1],
                            "Eje cadera",
                            ha="left",
                            va="top",
                            c="green",
                            alpha=0.6,
                            fontsize=6,
                            rotation="vertical",
                            transform=ax.transData,
                            zorder=5,
                        )

                    # Ajusta formato gráfica
                    # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                    g.set_title(
                        "Vista {0:s} {1:s}".format(
                            nfa.traduce_variables([parejes[0], parejes[1]]),
                            nfa.traduce_variables(nomvar),
                        )
                    )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                    g.set(xlabel=var1 + " (cm)", ylabel=var2 + " (cm)")  # ($^\circ$)

                    # En la leyenda sustituye por en español
                    custom_lines = [
                        Line2D(
                            [0],
                            [0],
                            color="r",
                            lw=2,
                            solid_capstyle="round",
                            label="Izq",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="lime",
                            lw=2,
                            solid_capstyle="round",
                            label="Der",
                        ),
                    ]
                    g.legend(
                        handles=custom_lines, loc="best"
                    )  # bbox_to_anchor=(.95,.95))#, bbox_transform=ax_ang.transAxes)

                    ax.axis("equal")  # hace los dos ejes proporcionales
                    plt.tight_layout()

                    if carpeta_guardar:
                        guarda_grafica(
                            f"_P_{nfa.traduce_variables(nomvar)}_{nfa.traduce_variables([parejes[0], parejes[1]])}",
                            carpeta_guardar=carpeta_output,
                        )

                    # plt.show()

            # ---------------------------------------

    # ---------------------------------------


def RealizaGraficas_triples_cinem(
    daGraf,
    repes=[0, 10],
    tipo_graf=["lados_lin"],
    compara_lados_graf=False,
    ensemble_avg=False,
    carpeta_guardar=False,
) -> None:  # por mejorar: que se puedan descartar repeticiones intermedias
    import itertools

    """
    Nomenclatura:
        '_A_': Gráfica Ángulo de articulación en escala grados biela.
        '_AP_': Gráfica Ángulo en coordenadas polares.
        '_APV_': Gráfica Ángulo en coordenadas polares con velocidad biela como color.
        '_AA_': Gráfica Ángulo/Ángulo
        '_I_': Gráfica individual (no compara L y R)
        '_IP_': Gráfica individual en coordenadas polares (no compara L y R)
        '_PP_': Gráfica posición/posición (plano) eje articular
        '_PPM_': Gráfica posición/posición (plano) ejes varias articulaciones
        '_A3D_': Gráfica Ángulo de misma variable en 3 ejes
    """
    # TODO: FALTA QUE SELECCIONE REPES SUELTAS

    # Si viene como xarray lo pasa a dataframe
    if isinstance(daGraf, xr.DataArray):
        dfGraf = daGraf.to_dataframe(name="value").dropna().reset_index()
        if dfGraf.shape[0] == 0:
            return

    # Cambia los nombres de las variables
    dfGraf["n_var"] = dfGraf["n_var"].replace(
        {
            dfGraf.n_var.unique()[i]: nfa.traduce_variables(dfGraf.n_var.unique()[i])
            for i in range(len(dfGraf.n_var.unique()))
        }
    )
    dfGraf["side"] = dfGraf["side"].replace({"L": "Izq", "R": "Der"})

    # dfLateral.loc[:, 'side'] = dfLateral.loc[:, 'side'].replace({'L':'I', 'R':'D'})

    numreps = dfGraf["phase"].max()
    if repes == None:  # si no se indica nada, muestra todas las repeticiones
        repes = [0, numreps]
    else:  # comprueba si tiene tantas repeticiones como se le pide
        if repes[1] > numreps:
            repes[1] = numreps
    rango = np.arange(
        repes[0], repes[1]
    ).tolist()  # rango de repeticiones para todas las gráficas
    dfGraf = dfGraf.query("phase==@rango").copy()

    # ---------------------------------------
    # Crea carpeta donde guardará las gráficas
    if carpeta_guardar:
        carpeta_output = Path(carpeta_guardar) / "Figs"
        if not carpeta_output.exists():
            carpeta_output.mkdir()

    if ensemble_avg:
        estim = "mean"
        ci = "sd"
        unit = None
        alpha = 1
    else:
        estim = None
        ci = ("ci", 95)
        unit = "phase"
        alpha = 0.5

    # Función genérica para guardar todas las gráficas creadas
    def guarda_grafica(nom="A", carpeta_guardar=None):
        if carpeta_guardar is None:
            print("No se ha especificado la carpeta de guardado")
            return

        ruta_fig = carpeta_guardar.joinpath(dfGraf["ID"].iloc[0] + nom).with_suffix(
            formatoImagenes
        )
        if formatoImagenes == ".pdf":
            with PdfPages(ruta_fig) as pdf_pages:
                pdf_pages.savefig(g.fig)
        else:
            g.fig.savefig(ruta_fig, dpi=300)

    rc = {"axes.labelsize": "x-large"}  # títulos de cada gráfica
    with sns.plotting_context(context="paper", rc=rc):
        # Gráficas ÁNGULO / BIELA repetición a repetición con variables por separado
        if "lados_lin" in tipo_graf:
            g = sns.relplot(
                data=dfGraf,
                x="AngBielaInRepe",
                y="value",
                col="n_var",
                hue="side",
                estimator=estim,
                errorbar=ci,
                units=unit,
                lw=1,
                palette=["r", "lime"],
                alpha=alpha,
                facet_kws={"sharey": False, "legend_out": False, "despine": False},
                solid_capstyle="round",
                kind="line",
                legend=True,
                aspect=1.2,
                height=3,
                zorder=2,
            )
            if ensemble_avg == "completo":
                # Dibuja todas las repeticiones
                g.map_dataframe(
                    sns.lineplot,
                    x="AngBielaInRepe",
                    y="value",
                    hue="side",
                    estimator=None,
                    errorbar=ci,
                    units="phase",
                    lw=0.25,
                    palette=["r", "lime"],
                    alpha=0.3,
                )

            (
                g.set_axis_labels(
                    "Ángulo de biela (grados)", "Ángulo (grados)"
                ).set_titles(  # , size='large') #pone etiquetas en ejes (las del x todas iguales)
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                # .tight_layout(w_pad=2) #separación entre gráficas
            )

            g.set(xlim=(0, 360), xticks=np.linspace(0, 360, 5))
            # g.axes[0,0].xaxis.grid(True, linestyle='dashed', dashes=(5, 5), dash_capstyle='round')
            # Configura leyenda
            # g._legend.texts[0].set_text('Izq')
            # g._legend.texts[1].set_text('Der')
            sns.move_legend(
                g,
                loc="upper right",
                bbox_to_anchor=(0.99, 1.02),
                title="",
                frameon=True,
            )
            # TODO: PONER ESTO CON MAP
            # TODO: INCLUIR DIFERENCIA MEDIA EN CADA GRÁFICA
            for nomvar, ax in g.axes_dict.items():
                # ax.set_xticks(np.linspace(0, 360, 5))
                # ax.set_xlim(0, 360)
                ax.xaxis.grid(
                    True,
                    linestyle="dashed",
                    dashes=(5, 5),
                    dash_capstyle="round",
                    alpha=0.8,
                    zorder=1,
                )
                # ax.set_xlabel('Ángulo de biela (grados)')

                if compara_lados_graf:
                    ti = calcula_spm1d(dfGraf.query("n_var==@nomvar"))
                    # Marca regiones con diferencia L y R
                    plot_clusters(ti, y=5, ax=ax, print_p=False)

            # plt.tight_layout()

            if carpeta_guardar:
                guarda_grafica(
                    nom=f"_A_Triple_{dfGraf.axis.unique()[0]}",
                    carpeta_guardar=carpeta_output,
                )

            # plt.show()

            # ---------------------------------------

        # Gráficas ÁNGULO / BIELA Radial
        if "lados_circ" in tipo_graf:

            g = sns.FacetGrid(
                dfGraf,
                col="n_var",
                hue="side",
                subplot_kws=dict(projection="polar"),
                height=4,
                legend_out=False,
                sharex=False,
                sharey=False,
                despine=False,
                palette=["r", "lime"],
            )

            if ensemble_avg == "completo":
                # Dibuja todas las repeticiones
                g.map_dataframe(
                    sns.lineplot,
                    x="AngBielaInRepe_rad",
                    y="value",
                    estimator=None,
                    errorbar=None,
                    units="phase",
                    lw=0.25,
                    palette=["r", "lime"],
                    alpha=0.3,
                )
            # Dibuja gráfica principal (después que línea a línea para que mantenga las características de la leyenda)
            g.map_dataframe(
                sns.lineplot,
                x="AngBielaInRepe_rad",
                y="value",
                estimator=estim,
                errorbar=ci,
                units=unit,
                lw=1,
                alpha=alpha,
            )

            g.add_legend()
            (
                g.set_axis_labels(
                    "", ""
                )  # pone etiquetas en ejes (las del x todas iguales)
                .set_titles(
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                .tight_layout(w_pad=0)  # separación entre gráficas
            )

            # Pone el valor 0º arriba y giro en sentido horario
            g.set(theta_zero_location="N", theta_direction=-1)

            # Ajusta leyenda
            sns.move_legend(
                g,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.95),
                title="",
                frameon=True,
            )
            """#En la leyenda sustituye por en español
            custom_lines = [Line2D([0], [0], color='r', lw=2, solid_capstyle='round', label='Izq'),
                            Line2D([0], [0], color='lime', lw=2, solid_capstyle='round', label='Der'),
                           ]
            g.axes[0,-1].legend(handles=custom_lines, bbox_to_anchor=(.95,1.0))#, bbox_transform=ax_ang.transAxes)
            """

            # plt.tight_layout()

            if carpeta_guardar:
                guarda_grafica(
                    nom=f"_AP_Triple_{dfGraf.axis.unique()[0]}",
                    carpeta_guardar=carpeta_output,
                )

            # plt.show()
        # ---------------------------------------

        # Gráficas coordinación Ángulo/Ángulo
        if "coordinacion" in tipo_graf:

            # Crea un dataframe con las comparaciones pareadas y con columna con nº comparación
            df = []
            for comp, parnomvar in enumerate(
                itertools.combinations(dfGraf.n_var.unique(), 2)
            ):
                # print(comp,parnomvar)
                df.append(
                    pd.concat(
                        [
                            dfGraf.rename(columns={"value": "var1"})
                            .query("n_var==@parnomvar[0]")
                            .reset_index(drop=True),
                            dfGraf.rename(columns={"value": "var2"})
                            .query("n_var==@parnomvar[1]")["var2"]
                            .reset_index(drop=True),
                        ],
                        axis=1,
                    ).assign(**{"comparacion": parnomvar[0] + "/" + parnomvar[1]})
                )
            df = pd.concat(df).sort_values(
                by=["n_var", "side"]
            )  # ordena para que mantenga el orden tras otras operaciones

            """
            g = sns.FacetGrid(df, col='comparacion', hue='side', #col_order=['Izq', 'Der'], hue_order=orden_musculos,
                       height=4, legend_out=False,
                       despine=False, palette=['r', 'lime'])
            
            def lineas(x,y, **kwargs):
                plt.plot(x,y, **kwargs)
                
            g.map_dataframe(lineas, x='var2', y='var1', solid_capstyle='round', lw=0.25, alpha=0.3)
            """

            df_ = df
            y = None
            if ensemble_avg:
                y = "var2"
                df_ = df.groupby(["comparacion", "side", "AngBielaInRepe"])[
                    ["var1", "var2"]
                ].mean()

            # Dibuja líneas (media o repe a repe según ensemble_avg)
            g = sns.relplot(
                data=df_,
                x="var2",
                y="var1",
                col="comparacion",
                hue="side",  # estimator=estim, ci=ci, units=unit,
                lw=1,
                palette=["lime", "r"],
                alpha=alpha,
                facet_kws={
                    "sharex": False,
                    "sharey": False,
                    "legend_out": False,
                    "despine": False,
                },
                sort=False,
                kind="line",
                legend=True,
                height=4,
                zorder=1,
            )

            # g = sns.relplot(data=df, x='var2', y='var1', col='comparacion', hue='side', estimator=estim, ci=ci, units=unit,
            #                 lw=1, palette=['r', 'lime'], alpha=0.8, facet_kws={'sharex': False, 'sharey': False, 'legend_out':False, 'despine':False}, sort=False, kind='line',
            #                 legend=True, height=4)#, zorder=1)
            (
                g.set_titles(
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                # .set_axis_labels('Ángulo de biela (grados)', 'Ángulo (grados)', size='large') #pone etiquetas en ejes (las del x todas iguales)
                # .tight_layout(w_pad=2) #separación entre gráficas
            )

            sns.move_legend(
                g,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.915),
                title="",
                frameon=True,
            )

            # Ajusta gráfica a gráfica
            for nomvar, ax in g.axes_dict.items():
                # print(nomvar)
                ax.set_xlabel(
                    f'{nomvar.split("/")[1]} (grados)'
                )  # , size='large') #($^\circ$)
                ax.set_ylabel(f'{nomvar.split("/")[0]} (grados)')  # , size='large')

                # ax.axis('equal') #hace los dos ejes proporcionales

                # Dibuja inicios de cada ciclo
                for posbiela in [0, 90, 180]:
                    posx = (
                        df.query(
                            'side=="Izq" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var2"]
                        .mean()
                    )
                    posy = (
                        df.query(
                            'side=="Izq" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var1"]
                        .mean()
                    )
                    ax.plot(
                        posx,
                        posy,
                        c="firebrick",
                        mfc="r",
                        marker="o",
                        ms=14,
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.text(
                        posx,
                        posy,
                        str(posbiela),
                        c="firebrick",
                        fontsize=8,
                        fontweight="bold",
                        transform=ax.transData,
                        horizontalalignment="center",
                        verticalalignment="center",
                        zorder=3,
                    )

                    posx = (
                        df.query(
                            'side=="Der" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var2"]
                        .mean()
                    )
                    posy = (
                        df.query(
                            'side=="Der" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var1"]
                        .mean()
                    )
                    ax.plot(
                        posx,
                        posy,
                        c="limegreen",
                        mfc="lime",
                        marker="o",
                        ms=14,
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.text(
                        posx,
                        posy,
                        str(posbiela),
                        c="limegreen",
                        fontsize=8,
                        fontweight="bold",
                        transform=ax.transData,
                        horizontalalignment="center",
                        verticalalignment="center",
                        zorder=3,
                    )

                """
                #Dibuja inicios de cada ciclo            
                ax.plot(df.query('side=="Izq" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var2'].mean(), 
                        df.query('side=="Izq" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var1'].mean(), 
                        c='firebrick', mfc='r', marker='o', ms=10, alpha=0.5, zorder=2)
                ax.plot(df.query('side=="Der" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var2'].mean(), 
                        df.query('side=="Der" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var1'].mean(), 
                        c='limegreen', mfc='lime', marker='o', ms=10, alpha=0.5, zorder=2)
                """

                # sombrea la zona de desviación típica si ensemble_avg no es False
                if ensemble_avg:
                    # for nomvar, ax in g.axes_dict.items():
                    # print(nomvar)
                    # Izq
                    v1 = (
                        df.query('comparacion==@nomvar & side=="Izq"')[
                            ["AngBielaInRepe", "var1"]
                        ]
                        .groupby("AngBielaInRepe")
                        .std()
                        ** 2
                    ).iloc[:, 0]
                    v2 = (
                        df.query('comparacion==@nomvar & side=="Izq"')[
                            ["AngBielaInRepe", "var2"]
                        ]
                        .groupby("AngBielaInRepe")
                        .std()
                        ** 2
                    ).iloc[:, 0]
                    err = np.sqrt(v1 + v2)
                    # err = (df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                    draw_error_band(
                        ax,
                        df_.query('comparacion==@nomvar & side=="Izq"').var2,
                        df_.query('comparacion==@nomvar & side=="Izq"').var1,
                        err=err,
                        facecolor="r",
                        edgecolor="none",
                        alpha=0.2,
                        zorder=0,
                    )

                    # Der
                    v1 = (
                        df.query('comparacion==@nomvar & side=="Der"')[
                            ["AngBielaInRepe", "var1"]
                        ]
                        .groupby("AngBielaInRepe")
                        .std()
                        ** 2
                    ).iloc[:, 0]
                    v2 = (
                        df.query('comparacion==@nomvar & side=="Der"')[
                            ["AngBielaInRepe", "var2"]
                        ]
                        .groupby("AngBielaInRepe")
                        .std()
                        ** 2
                    ).iloc[:, 0]
                    err = np.sqrt(v1 + v2)
                    # err = (df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                    draw_error_band(
                        ax,
                        df_.query('comparacion==@nomvar & side=="Der"').var2,
                        df_.query('comparacion==@nomvar & side=="Der"').var1,
                        err=err,
                        facecolor="lime",
                        edgecolor="none",
                        alpha=0.2,
                        zorder=0,
                    )

                    if ensemble_avg == "completo":
                        # print('dibuja repes sueltas en' + nomvar)
                        for rep in range(dfGraf.phase.max()):
                            ax.plot(
                                df.query(
                                    'side=="Izq" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var2"],
                                df.query(
                                    'side=="Izq" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var1"],
                                c="r",
                                lw=0.25,
                                alpha=0.3,
                                zorder=0,
                            )
                            ax.plot(
                                df.query(
                                    'side=="Der" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var2"],
                                df.query(
                                    'side=="Der" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var1"],
                                c="limegreen",
                                lw=0.25,
                                alpha=0.3,
                                zorder=0,
                            )

                # ax.plot(df.query('side=="Izq" & repe==@rango & comparacion==@nomvar').loc[:,'var2'].iloc[0], df.query('side=="Izq" & comparacion==@nomvar').loc[:,'var1'].iloc[0], c='r', marker='o', alpha=0.8, zorder=2)
                # ax.plot(df.query('side=="Der" & comparacion==@nomvar').loc[:,'var2'].iloc[0], df.query('side=="Der" & comparacion==@nomvar').loc[:,'var1'].iloc[0], c='lime', marker='o', alpha=0.8, zorder=2)

            plt.tight_layout()

            if carpeta_guardar:
                guarda_grafica(
                    nom=f"_AA_Triple_{dfGraf.axis.unique()[0]}",
                    carpeta_guardar=carpeta_output,
                )

            # plt.show()
        # ---------------------------------------

        # Gráfica POSICIÓN / POSICIÓN ejes articulares
        if "planoXY" in tipo_graf:
            # Crea un dataframe con las comparaciones pareadas y con columna con nº comparación
            df = []
            for comp, parejes in enumerate(
                itertools.combinations(dfGraf.axis.unique(), 2)
            ):
                # print(comp,parejes)
                # var1 = dfGraf.rename(columns = {'value':'var1'}).query('axis==@parejes[0] & n_var != "desconocido"') #desconocido son los datos de pospedal
                var1 = dfGraf.rename(columns={"value": "var1"}).query(
                    "axis==@parejes[0]"
                )

                # if parejes[0]=='x':
                #     offsetX = ((daGraf.isel(ID=0).sel(n_var='PosPedal', axis='x', side='L') + daGraf.isel(ID=0).sel(n_var='PosPedal', axis='x', side='R')) / 2).mean(dim=['AngBielaInRepe', 'phase']).data / 10 #lo pasa a cm
                #     var1['var1']-=offsetX #var1.query('side=="Der"').min()['var1'] #solo para ajustar el cero del eje x

                # En la visión frontal invierte el axis x para que se vea de frente
                if parejes == ("x", "z"):
                    var1["var1"] = -var1["var1"]
                    var1["var1"] -= var1.query('side=="Der"').min()[
                        "var1"
                    ]  # solo para ajustar el cero del axis x
                # En la visión superior invierte el axis x para que las coordenadas sean positivas
                if parejes == ("x", "y"):
                    var1["var1"] -= var1.iloc[
                        0, var1.columns.get_loc("var1")
                    ]  # solo para ajustar el cero del axis

                var2 = dfGraf.rename(columns={"value": "var2"}).query(
                    "axis==@parejes[1]"
                )["var2"]

                df.append(
                    pd.concat(
                        [var1.reset_index(drop=True), var2.reset_index(drop=True)],
                        axis=1,
                    ).assign(**{"comparacion": parejes[0] + "/" + parejes[1]})
                )
            df = pd.concat(df).sort_values(
                by=["n_var", "side"]
            )  # ordena para que mantenga el orden tras otras operaciones
            df["comparacion"] = df["comparacion"].replace(
                {"x/y": "Vista cenital", "x/z": "Vista frontal", "y/z": "Vista lateral"}
            )
            # df['axis'] = df['comparacion'].replace({'x/y':'Superior', 'x/z':'frontal', 'y/z':'lateral'})

            df_ = df
            y = None
            if ensemble_avg:
                y = "var2"
                df_ = df.groupby(["comparacion", "side", "AngBielaInRepe"])[
                    ["var1", "var2"]
                ].mean()

            # Dibuja líneas (media o repe a repe según ensemble_avg)
            g = sns.relplot(
                data=df_,
                x="var1",
                y="var2",
                col="comparacion",
                hue="side",  # estimator=estim, ci=ci, units=unit,
                lw=1,
                palette=["lime", "r"],
                alpha=alpha,
                facet_kws={
                    "sharex": False,
                    "sharey": False,
                    "legend_out": False,
                    "despine": False,
                },
                sort=False,
                kind="line",
                legend=True,
                height=4,
                zorder=1,
            )

            # g = sns.relplot(data=df, x='var2', y='var1', col='comparacion', hue='side', estimator=estim, ci=ci, units=unit,
            #                 lw=1, palette=['r', 'lime'], alpha=0.8, facet_kws={'sharex': False, 'sharey': False, 'legend_out':False, 'despine':False}, sort=False, kind='line',
            #                 legend=True, height=4)#, zorder=1)
            (
                g.set_titles(
                    "{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                # .set_axis_labels('Ángulo de biela (grados)', 'Ángulo (grados)', size='large') #pone etiquetas en ejes (las del x todas iguales)
                # .tight_layout(w_pad=2) #separación entre gráficas
            )

            sns.move_legend(
                g,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.915),
                title="",
                frameon=True,
            )

            # Ajusta gráfica a gráfica
            for nomvar, ax in g.axes_dict.items():
                # print(nomvar)
                if nomvar == "Vista cenital":
                    xlabel = "mediolateral"
                    ylabel = "anteroposterior"
                    ax.axvline(
                        x=0,
                        c="grey",
                        ls="--",
                        dashes=(5, 5),
                        dash_capstyle="round",
                        alpha=0.5,
                    )
                elif nomvar == "Vista frontal":
                    xlabel = "mediolateral"
                    ylabel = "vertical"
                    ax.axvline(
                        x=0,
                        c="grey",
                        ls="--",
                        dashes=(5, 5),
                        dash_capstyle="round",
                        alpha=0.5,
                    )
                if nomvar == "Vista lateral":
                    xlabel = "anteroposterior"
                    ylabel = "vertical"

                ax.set_xlabel(f"{xlabel} (cm)")  # , size='large') #($^\circ$)
                ax.set_ylabel(f"{ylabel} (cm)")  # , size='large')

                # ax.axis('equal') #hace los dos ejes proporcionales

                # Dibuja inicios de cada ciclo
                """
                ax.plot(df.query('side=="Izq" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var1'].mean(), 
                            df.query('side=="Izq" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var2'].mean(), 
                            c='firebrick', mfc='r', marker='o', ms=10, alpha=0.5, zorder=2)
                ax.plot(df.query('side=="Der" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var1'].mean(), 
                        df.query('side=="Der" & comparacion==@nomvar & AngBielaInRepe==0').loc[:,'var2'].mean(), 
                        c='limegreen', mfc='lime', marker='o', ms=10, alpha=0.5, zorder=2)
                """
                # Dibuja la posición en puntos críticos de cada ciclo
                for posbiela in [0, 90, 180]:
                    posx = (
                        df.query(
                            'side=="Izq" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var1"]
                        .mean()
                    )
                    posy = (
                        df.query(
                            'side=="Izq" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var2"]
                        .mean()
                    )
                    ax.plot(
                        posx,
                        posy,
                        c="firebrick",
                        mfc="r",
                        marker="o",
                        ms=14,
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.text(
                        posx,
                        posy,
                        str(posbiela),
                        c="firebrick",
                        fontsize=8,
                        fontweight="bold",
                        transform=ax.transData,
                        horizontalalignment="center",
                        verticalalignment="center",
                        zorder=3,
                    )

                    posx = (
                        df.query(
                            'side=="Der" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var1"]
                        .mean()
                    )
                    posy = (
                        df.query(
                            'side=="Der" & comparacion==@nomvar & AngBielaInRepe==@posbiela'
                        )
                        .loc[:, "var2"]
                        .mean()
                    )
                    ax.plot(
                        posx,
                        posy,
                        c="limegreen",
                        mfc="lime",
                        marker="o",
                        ms=14,
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.text(
                        posx,
                        posy,
                        str(posbiela),
                        c="limegreen",
                        fontsize=8,
                        fontweight="bold",
                        transform=ax.transData,
                        horizontalalignment="center",
                        verticalalignment="center",
                        zorder=3,
                    )

                # sombrea la zona de desviación típica si ensemble_avg no es False
                if ensemble_avg:
                    # for nomvar, ax in g.axes_dict.items():
                    # print(nomvar)
                    err = np.sqrt(
                        df.query('comparacion==@nomvar & side=="Izq"')
                        .groupby("AngBielaInRepe")
                        .std()
                        .var1
                        ** 2
                        + df.query('comparacion==@nomvar & side=="Izq" & phase==@rango')
                        .groupby("AngBielaInRepe")
                        .std()
                        .var2
                        ** 2
                    )
                    # err = (df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                    draw_error_band(
                        ax,
                        df_.query('comparacion==@nomvar & side=="Izq"').var1,
                        df_.query('comparacion==@nomvar & side=="Izq"').var2,
                        err=err,
                        facecolor="r",
                        edgecolor="none",
                        alpha=0.2,
                        zorder=0,
                    )
                    err = np.sqrt(
                        df.query('comparacion==@nomvar & side=="Der"')
                        .groupby("AngBielaInRepe")
                        .std()
                        .var1
                        ** 2
                        + df.query('comparacion==@nomvar & side=="Der" & phase==@rango')
                        .groupby("AngBielaInRepe")
                        .std()
                        .var2
                        ** 2
                    )
                    # err = (df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                    draw_error_band(
                        ax,
                        df_.query('comparacion==@nomvar & side=="Der"').var1,
                        df_.query('comparacion==@nomvar & side=="Der"').var2,
                        err=err,
                        facecolor="lime",
                        edgecolor="none",
                        alpha=0.2,
                        zorder=0,
                    )

                    if ensemble_avg == "completo":
                        # print('dibuja repes sueltas en' + nomvar)
                        for rep in range(dfGraf.phase.max()):
                            ax.plot(
                                df.query(
                                    'side=="Izq" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var1"],
                                df.query(
                                    'side=="Izq" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var2"],
                                c="r",
                                lw=0.25,
                                alpha=0.3,
                                zorder=0,
                            )
                            ax.plot(
                                df.query(
                                    'side=="Der" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var1"],
                                df.query(
                                    'side=="Der" & comparacion==@nomvar & phase==@rep'
                                ).loc[:, "var2"],
                                c="limegreen",
                                lw=0.25,
                                alpha=0.3,
                                zorder=0,
                            )

                # ax.plot(df.query('side=="Izq" & repe==@rango & comparacion==@nomvar').loc[:,'var2'].iloc[0], df.query('side=="Izq" & comparacion==@nomvar').loc[:,'var1'].iloc[0], c='r', marker='o', alpha=0.8, zorder=2)
                # ax.plot(df.query('side=="Der" & comparacion==@nomvar').loc[:,'var2'].iloc[0], df.query('side=="Der" & comparacion==@nomvar').loc[:,'var1'].iloc[0], c='lime', marker='o', alpha=0.8, zorder=2)

            plt.tight_layout()

            if carpeta_guardar:
                guarda_grafica(nom=f"_PP_{dfGraf["n_var"][0]}_Triple")

            # plt.show()
            # --------------------------------

        # Gráfica POSICIÓN / POSICIÓN con los tres ejes articulares simultáneamente
        if "planoXYMultiple" in tipo_graf:
            # Crea un dataframe con las comparaciones pareadas y con columna con nº comparación
            dfm = []
            for comp, parejes in enumerate(
                itertools.combinations(dfGraf.axis.unique(), 2)
            ):
                # print(comp,parejes)
                var1 = dfGraf.rename(columns={"value": "var1"}).query(
                    'axis==@parejes[0] & n_var != "desconocido"'
                )  # desconocido son los datos de pospedal

                if parejes[0] == "x":
                    offsetX = (
                        (
                            daGraf.isel(ID=0).sel(n_var="PosPedal", axis="x", side="L")
                            + daGraf.isel(ID=0).sel(
                                n_var="PosPedal", axis="x", side="R"
                            )
                        )
                        / 2
                    ).mean(
                        dim=["AngBielaInRepe", "phase"]
                    ).data / 10  # lo pasa a cm
                    var1[
                        "var1"
                    ] -= offsetX  # var1.query('side=="Der"').min()['var1'] #solo para ajustar el cero del eje x

                # En la visión frontal invierte el eje x para que se vea de frente
                if parejes == ("x", "z"):
                    print("")
                    # offsetX = ((daGraf.isel(ID=0).sel(n_var='PosPedal', axis='x', side='L') + daGraf.isel(ID=0).sel(n_var='PosPedal', axis='x', side='R')) / 2).mean(dim=['AngBielaInRepe', 'repe']).data
                    var1["var1"] = -var1["var1"]
                    # var1['var1']+=offsetX #var1.query('side=="Der"').min()['var1'] #solo para ajustar el cero del eje x
                # En la visión superior invierte el eje x para que las coordenadas sean positivas
                if parejes == ("x", "y"):
                    print("")
                    # var1['var1']-=var1.iloc[0,var1.columns.get_loc('var1')] #solo para ajustar el cero del eje

                var2 = dfGraf.rename(columns={"value": "var2"}).query(
                    "axis==@parejes[1]"
                )["var2"]

                dfm.append(
                    pd.concat(
                        [var1.reset_index(drop=True), var2.reset_index(drop=True)],
                        axis=1,
                    ).assign(**{"comparacion": parejes[0] + "/" + parejes[1]})
                )
            dfm = pd.concat(dfm).sort_values(
                by=["n_var", "side"]
            )  # ordena para que mantenga el orden tras otras operaciones
            dfm["comparacion"] = dfm["comparacion"].replace(
                {"x/y": "Vista cenital", "x/z": "Vista frontal", "y/z": "Vista lateral"}
            )
            # df['axis'] = df['comparacion'].replace({'x/y':'Superior', 'x/z':'frontal', 'y/z':'lateral'})

            # Función para dibujar gráficas variable a variable
            def draw_muscle(x, y, **kwargs):
                # print(kwargs['data'].query('n_var=="Eje Rodilla"'))#.keys(), kwargs['data'].items())
                for n, gb in kwargs["data"].groupby("n_var"):
                    plt.plot(gb["var1"], gb["var2"], c=kwargs["color"], lw=kwargs["lw"])

            g = sns.FacetGrid(
                dfm,
                col="comparacion",
                hue="side",  # col_order=['Izq', 'Der'],
                legend_out=False,
                sharex=False,
                sharey=False,
                height=4,
                despine=False,
                palette=["r", "lime"],
            )

            g.map_dataframe(draw_muscle, x="var1", y="var2", lw=0.5)

            g.add_legend()
            (
                g.set_titles(
                    "{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                # .set_axis_labels('Ángulo de biela (grados)', 'Ángulo (grados)', size='large') #pone etiquetas en ejes (las del x todas iguales)
                .tight_layout(w_pad=0)  # separación entre gráficas
            )

            sns.move_legend(
                g,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.915),
                title="",
                frameon=True,
            )

            # Ajusta gráfica a gráfica
            for nomvar, ax in g.axes_dict.items():
                # print(nomvar)
                if nomvar == "Vista cenital":
                    xlabel = "mediolateral"
                    ylabel = "anteroposterior"
                    ax.axvline(
                        x=0,
                        c="grey",
                        ls="--",
                        dashes=(5, 5),
                        lw=1,
                        dash_capstyle="round",
                        alpha=0.5,
                    )
                elif nomvar == "Vista frontal":
                    xlabel = "mediolateral"
                    ylabel = "vertical"
                    ax.axvline(
                        x=0,
                        c="grey",
                        ls="--",
                        dashes=(5, 5),
                        lw=1,
                        dash_capstyle="round",
                        alpha=0.5,
                    )
                if nomvar == "Vista lateral":
                    xlabel = "anteroposterior"
                    ylabel = "vertical"

                # ax.axis('equal') #hace los dos ejes proporcionales

                ax.set_xlabel(f"{xlabel} (cm)")  # , size='large') #($^\circ$)
                ax.set_ylabel(f"{ylabel} (cm)")  # , size='large')

                ax.axis("equal")  # hace los dos ejes proporcionales
                # ax.set_aspect('equal', 'box')

                # plt.tight_layout()

            if carpeta_guardar:
                guarda_grafica(nom="_PPM_Multiple_Triple_")

            # plt.show()
            # --------------------------------

        # Gráficas ÁNGULO SEGMENTOS / BIELA repetición a repetición con variables por separado
        if "3D_lin" in tipo_graf:
            dfGraf["axis"].replace(
                {
                    "x": "Vista " + nfa.traduce_variables("x"),
                    "y": "Vista " + nfa.traduce_variables("y"),
                    "z": "Vista " + nfa.traduce_variables("z"),
                },
                inplace=True,
            )  # {'x':u'Basculación ant./post.', 'y':u'Basculación lat.', 'z':u'Rotación'}, inplace=True)

            g = sns.relplot(
                data=dfGraf,
                x="AngBielaInRepe",
                y="value",
                col="axis",
                estimator=estim,
                errorbar=ci,
                units=unit,
                lw=1,
                color="orange",
                alpha=alpha,
                facet_kws={"sharey": False, "legend_out": False, "despine": False},
                solid_capstyle="round",
                kind="line",
                legend=True,
                aspect=1.3,
                height=3,
                zorder=1,
            )

            (
                g.set_axis_labels(
                    "Ángulo de biela (grados)", "Ángulo (grados)"
                ).set_titles(  # , size='large') #pone etiquetas en ejes (las del x todas iguales)
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                # .tight_layout(w_pad=2) #separación entre gráficas
            )
            g.set(xlim=(0, 360), xticks=np.linspace(0, 360, 5))
            # g.axes[0,0].xaxis.grid(True, linestyle='dashed', dashes=(5, 5), dash_capstyle='round')
            # Configura leyenda
            # g._legend.texts[0].set_text('Izq')
            # g._legend.texts[1].set_text('Der')
            # sns.move_legend(g, loc='upper right', bbox_to_anchor=(1.0, 1.01),  title='', frameon=True)

            for nomvar, ax in g.axes_dict.items():
                # ax.set_xticks(np.linspace(0, 360, 5))
                # ax.set_xlim(0, 360)
                ax.xaxis.grid(
                    True,
                    linestyle="dashed",
                    dashes=(5, 5),
                    dash_capstyle="round",
                    alpha=0.8,
                    zorder=0,
                )
                # ax.set_xlabel('Ángulo de biela (grados)')

            if ensemble_avg == "completo":
                # Dibuja todas las repeticiones porque con relplot no puede superponer
                for eje, ax in g.axes_dict.items():
                    for rep in range(dfGraf.phase.max()):
                        ax.plot(
                            range(361),
                            dfGraf.query("axis==@eje & phase==@rep")["value"],
                            c="orange",
                            lw=0.25,
                            alpha=0.3,
                            zorder=0,
                        )

            # plt.tight_layout()

            if carpeta_guardar:
                guarda_grafica(nom=f"_A3D_{dfGraf["n_var"][0]}_Triple")
            # ---------------------------------------

        # ---------------------------------------


def RealizaGraficas_EMG(
    daGraf,
    repes=[0, 10],
    nomvars=["REC", "BIC"],
    tipo_graf=["lados_lin"],
    otros_datos=None,
    extra_nom_archivo="",
    compara_lados_graf=False,
    ensemble_avg=False,
    carpeta_guardar=False,
) -> None:  # por mejorar: que se puedan descartar repeticiones intermedias
    # import itertools
    """
    Nomenclatura:
        '_E_': Gráfica EMG comparando lados
        '_EP_': Gráfica EMG comparando lados en coordenadas polares
        '_EE_': Gráfica coordinación EMG/ENG
        '_EON_' : Gráfica polar con onset músculos y lados separados
        '_EMP_' : gráfica EMG polar con músculos juntos y lados separados
        '_EG_' : gráficas comparando lados grupales
        '_EPG_' : gráficas comparando lados grupales en coordenadas polares

    """
    # FALTA QUE SELECCIONE REPES SUELTAS

    # Si viene como xarray lo pasa a dataframe
    if isinstance(daGraf, xr.DataArray):
        dfGraf = daGraf.to_dataframe(name="value").dropna().reset_index()

    # Cambia los nombres de las variables
    # dfGraf['n_var'] = dfGraf['n_var'].replace({dfGraf.n_var.unique()[i]:nfa.traduce_variables(dfGraf.n_var.unique()[i]) for i in range(len(dfGraf.n_var.unique()))})
    dfGraf["side"] = dfGraf["side"].replace({"L": "Izq", "R": "Der"})

    numreps = dfGraf["phase"].max()
    if repes == None:  # si no se indica nada, muestra todas las repeticiones
        repes = [0, numreps]
    else:  # comprueba si tiene tantas repeticiones como se le pide
        if repes[1] > numreps:
            repes[1] = numreps
    rango = np.arange(
        repes[0], repes[1]
    ).tolist()  # rango de repeticiones para todas las gráficas
    dfGraf = dfGraf.query("phase==@rango").copy()

    # ---------------------------------------
    # Crea carpeta donde guardará las gráficas
    if carpeta_guardar:
        carpeta_output = Path(carpeta_guardar) / "Figs"
        if not carpeta_output.exists():
            carpeta_output.mkdir()

    if ensemble_avg == True or ensemble_avg == "completo":
        estim = "mean"
        ci = "sd"
        unit = None
        alpha = 1
        # g = sns.lineplot(data=dfGraf.query('n_var == @nomvar & axis ==@eje & repe==@rango'), x='AngBielaInRepe', y='value', hue='side', ci='sd', lw=1, palette=['r', 'lime'], ax=ax)
    else:
        estim = None
        ci = 95
        unit = "phase"
        alpha = 0.5

    # Función genérica para guardar todas las gráficas creadas
    def guarda_grafica(nom="A", carpeta_guardar=None):
        if carpeta_guardar is None:
            print("No se ha especificado la carpeta de guardado")
            return

        ruta_fig = carpeta_guardar.joinpath(dfGraf["ID"].iloc[0] + nom).with_suffix(
            formatoImagenes
        )
        # ruta_fig = CarpetaSesion + 'Figs\\' + ArchivoActivo+ '_A_' + nfa.traduce_variables(nomvar) + '_' + eje + formatoImagenes
        if formatoImagenes == ".pdf":
            with PdfPages(ruta_fig) as pdf_pages:
                pdf_pages.savefig(fig)
        else:
            fig.savefig(ruta_fig, dpi=300)

    rc = {"axes.titlesize": "large"}  # títulos de cada gráfica
    with sns.plotting_context(context="paper", rc=rc):
        # Gráficas ÁNGULO / BIELA repetición a repetición con variables por separado
        if "lados_lin" in tipo_graf:
            for nomvar in nomvars:
                # print(nomvar)
                titulo = nfa.traduce_variables(nomvar)

                # fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(8,6))#para incluir gráfica spm{t}
                fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

                # Compara lados con spm1d
                if compara_lados_graf:
                    ti = calcula_spm1d(dfGraf.query("n_var==@nomvar"))

                    # Marca regiones con diferencia L y R
                    if ti != None:
                        plot_clusters(ti, y=5, ax=ax, print_p=False)

                if ensemble_avg == "completo":
                    sns.lineplot(
                        data=dfGraf.query("n_var==@nomvar"),
                        x="AngBielaInRepe",
                        y="value",
                        hue="side",
                        estimator=None,
                        errorbar=ci,
                        units="phase",
                        lw=0.25,
                        palette=["r", "lime"],
                        alpha=0.3,
                        ax=ax,
                    )  #'darkgrey', 'darkgrey'
                g = sns.lineplot(
                    data=dfGraf.query("n_var==@nomvar"),
                    x="AngBielaInRepe",
                    y="value",
                    hue="side",
                    estimator=estim,
                    errorbar=ci,
                    units=unit,
                    lw=1,
                    palette=["r", "lime"],
                    alpha=alpha,
                    ax=ax,
                )
                # Ajusta formato gráfica
                # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set_title(
                    titulo
                )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set(xlim=(0, 360), xticks=np.linspace(0, 360, 5))
                g.set(
                    xlabel="Ángulo de biela (grados)", ylabel="EMG (%MVC)"
                )  # ($^\circ$)

                g.xaxis.grid(True, linestyle="dashed", dash_capstyle="round")

                # En la leyenda sustituye por en español
                custom_lines = [
                    Line2D(
                        [0], [0], color="r", lw=2, solid_capstyle="round", label="Izq"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="lime",
                        lw=2,
                        solid_capstyle="round",
                        label="Der",
                    ),
                ]
                g.legend(
                    handles=custom_lines, loc="best"
                )  # bbox_to_anchor=(.95,.95))#, bbox_transform=ax_ang.transAxes)

                # ---------------------------------------

                plt.tight_layout()

                if carpeta_guardar:
                    guarda_grafica(nom=f"_E_{nfa.traduce_variables(nomvar)}")

                # plt.show()

        # Gráfica ÁNGULO / BIELA Radial
        if "lados_circ" in tipo_graf:
            for nomvar in nomvars:
                # print(nomvar,eje)
                titulo = nfa.traduce_variables(nomvar)

                fig, ax = plt.subplots(
                    figsize=(3, 3), subplot_kw=dict(projection="polar"), dpi=300
                )
                if ensemble_avg == "completo":
                    sns.lineplot(
                        data=dfGraf.query("n_var==@nomvar"),
                        x="AngBielaInRepe_rad",
                        y="value",
                        hue="side",
                        estimator=None,
                        errorbar=ci,
                        units="phase",
                        lw=0.25,
                        palette=["r", "lime"],
                        alpha=0.3,
                        ax=ax,
                    )  #'darkgrey', 'darkgrey'
                g = sns.lineplot(
                    data=dfGraf.query("n_var==@nomvar"),
                    x="AngBielaInRepe_rad",
                    y="value",
                    hue="side",
                    estimator=estim,
                    errorbar=ci,
                    units=unit,
                    lw=1,
                    palette=["r", "lime"],
                    alpha=alpha,
                    ax=ax,
                )

                # Pone el valor 0º arriba y giro en sentido horario
                g.set(theta_zero_location="N", theta_direction=-1, xlabel="", ylabel="")

                g.set_title(
                    titulo
                )  #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))

                # En la leyenda sustituye por en español
                custom_lines = [
                    Line2D(
                        [0], [0], color="r", lw=2, solid_capstyle="round", label="Izq"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="lime",
                        lw=2,
                        solid_capstyle="round",
                        label="Der",
                    ),
                ]
                g.legend(
                    handles=custom_lines, bbox_to_anchor=(0.0, 1.2)
                )  # , bbox_transform=ax_ang.transAxes)

                plt.tight_layout()

                if carpeta_guardar:
                    guarda_grafica(nom=f"_EP_{nfa.traduce_variables(nomvar)}")
                    
                # ruta_fig = carpeta_output.joinpath(dfGraf['ID'].iloc[0] + '_AP_' + nfa.traduce_variables(nomvar) + '_' + eje).with_suffix(formatoImagenes)
                # #CarpetaSesion + 'Figs\\' + ArchivoActivo+ '_AP_' + nfa.traduce_variables(nomvar) + '_' + eje + formatoImagenes
                # if formatoImagenes=='.pdf':
                #     with PdfPages(ruta_fig) as pdf_pages:
                #         pdf_pages.savefig(fig)
                # else:
                #     fig.savefig(ruta_fig, dpi=300)

                # plt.show()
        # ---------------------------------------

        # Gráficas coordinación entre pares de músculos
        if "coordinacion" in tipo_graf:
            for parnomvar in nomvars:
                # print(parnomvar)
                # rango=np.arange(repes[0],repes[1]).tolist()

                # Adapta el dataframe para que tenga las dos variables en columnas
                df = dfGraf.query("n_var==@parnomvar[0]")[
                    ["ID", "phase", "side", "AngBielaInRepe"]
                ].reset_index(drop=True)
                df[parnomvar[0]] = dfGraf.query("n_var==@parnomvar[0]")[
                    "value"
                ].reset_index(drop=True)
                df[parnomvar[1]] = dfGraf.query("n_var==@parnomvar[1]")[
                    "value"
                ].reset_index(drop=True)

                df_ = df
                y = None
                if ensemble_avg:
                    y = parnomvar[1]
                    df_ = df.groupby(["side", "AngBielaInRepe"])[
                        [parnomvar[0], parnomvar[1]]
                    ].mean()

                fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
                # Dibuja líneas (media o repe a repe según ensemble_avg)
                g = sns.lineplot(
                    data=df_,
                    x=parnomvar[1],
                    y=parnomvar[0],
                    units=unit,
                    hue="side",
                    estimator=None,
                    palette=["r", "lime"],
                    sort=False,
                    alpha=alpha,
                    ax=ax,
                )

                # Dibuja inicios de cada ciclo
                ax.plot(
                    df.query('side=="Izq" & AngBielaInRepe==0')
                    .loc[:, parnomvar[1]]
                    .mean(),
                    df.query('side=="Izq" & AngBielaInRepe==0')
                    .loc[:, parnomvar[0]]
                    .mean(),
                    c="firebrick",
                    mfc="r",
                    marker="o",
                    ms=10,
                    alpha=0.5,
                    zorder=2,
                )
                ax.plot(
                    df.query('side=="Der" & AngBielaInRepe==0')
                    .loc[:, parnomvar[1]]
                    .mean(),
                    df.query('side=="Der" & AngBielaInRepe==0')
                    .loc[:, parnomvar[0]]
                    .mean(),
                    c="limegreen",
                    mfc="lime",
                    marker="o",
                    ms=10,
                    alpha=0.5,
                    zorder=2,
                )

                # sombrea la zona de desviación típica si ensemble_avg no es False
                if ensemble_avg:
                    # for nomvar, ax in g.axes_dict.items():
                    # print(nomvar)
                    err = np.sqrt(
                        df.query('side=="Izq"')
                        .groupby("AngBielaInRepe")
                        .std()[parnomvar[0]]
                        ** 2
                        + df.query('side=="Izq"')
                        .groupby("AngBielaInRepe")
                        .std()[parnomvar[1]]
                        ** 2
                    )
                    # err = (df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Izq" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                    draw_error_band(
                        ax,
                        df_.query('side=="Izq"')[parnomvar[1]],
                        df_.query('side=="Izq"')[parnomvar[0]],
                        err=err,
                        facecolor="r",
                        edgecolor="none",
                        alpha=0.2,
                        zorder=0,
                    )
                    err = np.sqrt(
                        df.query('side=="Der"')
                        .groupby("AngBielaInRepe")
                        .std()[parnomvar[0]]
                        ** 2
                        + df.query('side=="Der"')
                        .groupby("AngBielaInRepe")
                        .std()[parnomvar[1]]
                        ** 2
                    )
                    # err = (df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var1 + df.query('comparacion==@nomvar & side=="Der" & repe==@rango').groupby('AngBielaInRepe').std().var2)/2
                    draw_error_band(
                        ax,
                        df_.query('side=="Der"')[parnomvar[1]],
                        df_.query('side=="Der"')[parnomvar[0]],
                        err=err,
                        facecolor="lime",
                        edgecolor="none",
                        alpha=0.2,
                        zorder=0,
                    )

                    if ensemble_avg == "completo":
                        # print('dibuja repes sueltas en' + nomvar)
                        for rep in range(dfGraf.phase.max()):
                            ax.plot(
                                df.query('side=="Izq" & phase==@rep').loc[
                                    :, parnomvar[1]
                                ],
                                df.query('side=="Izq" & phase==@rep').loc[
                                    :, parnomvar[0]
                                ],
                                c="r",
                                lw=0.25,
                                alpha=0.3,
                                zorder=0,
                            )
                            ax.plot(
                                df.query('side=="Der" & phase==@rep').loc[
                                    :, parnomvar[1]
                                ],
                                df.query('side=="Der" & phase==@rep').loc[
                                    :, parnomvar[0]
                                ],
                                c="limegreen",
                                lw=0.25,
                                alpha=0.3,
                                zorder=0,
                            )

                # Ajusta formato gráfica
                # g.figure.suptitle('{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))
                g.set_title(
                    "Coordinación muscular"
                )  # {0:s}/{1:s}'.format(nfa.traduce_variables(parnomvar[0]), nfa.traduce_variables(parnomvar[1]))) #'{0:s} ({1:s})'.format(dfLateral['ID'].iloc[0], nomvar))

                g.set(
                    xlabel=f"{nfa.traduce_variables(parnomvar[1])} (%MVC)",
                    ylabel=f"{nfa.traduce_variables(parnomvar[0])} (%MVC)",
                )

                # g.axis('equal') #hace los dos ejes proporcionales

                # En la leyenda sustituye por en español
                custom_lines = [
                    Line2D(
                        [0], [0], color="r", lw=2, solid_capstyle="round", label="Izq"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="lime",
                        lw=2,
                        solid_capstyle="round",
                        label="Der",
                    ),
                ]
                g.legend(
                    handles=custom_lines, loc="best"
                )  # bbox_to_anchor=(.95,.95))#, bbox_transform=ax_ang.transAxes)

                plt.tight_layout()

                if carpeta_guardar:
                    guarda_grafica(nom=f"_EE_{nfa.traduce_variables(parnomvar[0])}_{nfa.traduce_variables(parnomvar[1])}")

                # plt.show()

        # ---------------------------------------

        # Gráficas onset músculos (por separado lados)
        if "onset_circ" in tipo_graf:
            orden_musculos = [
                "GAS",
                "TIB",
                "REC",
                "VME",
                "BIC",
                "GLU",
            ]  # ['GLU', 'BIC', 'VME', 'REC', 'TIB', 'GAS']
            titulo = "Activación muscular"
            # dfGraf = daGraf.to_dataframe(name='value').dropna().reset_index()
            # dfGraf['side'] = dfGraf['side'].replace({'L':'Izq', 'R':'Der'})

            """
            dfGraf['id_var'] = dfGraf['n_var'].astype('category').cat.codes
            
            plt.plot(np.where(dfGraf['value']>=umbral_onset, dfGraf['n_var'].astype('category').cat.codes, np.nan))
            
            dfGraf['value_cat'] = np.where(dfGraf['value']>=umbral_onset, dfGraf['n_var'].astype('category').cat.codes, np.nan)
            dfGraf.plot(x='AngBielaInRepe', y='value_cat')
            
            dfGraf['id_var2'] = dfGraf['n_var'].astype('category').sort_values().cat.codes #rename_categories([1,2,3,4,5,6]).codes
            dfGraf['value_cat'] = np.where(dfGraf['value']>=20.0, dfGraf['n_var'].astype('category').sort_values().cat.codes, np.nan)
            dfGraf.query('n_var=="TIB" & side=="L"').plot(x='AngBielaInRepe', y='value_cat')
            """

            # reprocesa EMG dinámica
            # _, dadinamic, _ = carga_preprocesa_csv_EMG([(carpeta_guardar / daGraf.ID.data[0].split('_')[-1]).with_suffix('.csv')], nomBloque='Devices')
            daDinamic = otros_datos[0]
            da_tkeo = procesaEMG(daDinamic, fc_band=[30, 300], fclow=10, btkeo=True)
            # da_tkeo.plot.line(x='time', row='n_var', col='side')

            # dakinem = carga_preprocesa_csv_EMG([(carpeta_guardar / daGraf.ID.data[0].split('_')[-1]).with_suffix('.csv')], nomBloque='Model Outputs')
            dakinem = otros_datos[1]
            """
            from filtrar_Butter import filtrar_Butter, filtrar_Butter_bandpass
            daEMG_proces = filtrar_Butter_bandpass(daDinamic, fr=daDinamic.frec, fclow=30, fchigh=300)
            da_tkeo = xr.apply_ufunc(tkeo, daEMG_proces,
                           input_core_dims=[['time']],
                           output_core_dims=[['time']],
                           vectorize=True)            
            da_tkeo = abs(da_tkeo)
            da_tkeo = filtrar_Butter(da_tkeo , fr=daDinamic.frec, fc=10, kind='low') 
            #da_tkeo.isel(ID=0).sel(n_var='VME').plot.line(x='time', col='side')
            """
            # TODO: AQUÍ AJUSTAR LA CARPETA PARTICULAR CUANDO TIENE QUE GUARDAR EN CARPETA GLOBAL
            # Carga MVC sin procesar. OJO, anteriormente guardaba el archivo .nc ya procesado
            ruta = list(carpeta_guardar.parent.glob("**/*.nc"))
            ruta = [
                x
                for x in ruta
                if "PreprocesadoMVC_" + dfGraf["ID"].unique()[0].split("_")[1] + ".nc"
                in x.name
            ][
                0
            ]  # .parent
            da_tkeoMVC = xr.load_dataarray(ruta)
            da_tkeoMVC = procesaEMG(da_tkeoMVC, fc_band=[30, 300], fclow=10, btkeo=True)
            da_tkeoMVC = limpia_MVC(da_tkeoMVC)
            # Selecciona los archivos para la normalización
            if filtro_MVC.lower() == "todos":
                da_normalizar = da_tkeoMVC
            # elif filtro_MVC.lower()=='auto':
            #     da_normalizar = cargaVariablesNexus(nomBloque='Devices')[1]
            #     da_normalizar = procesaEMG(da_normalizar, fc_band=[30, 300], fclow=10, tkeo=True)
            # filtro_MVC = filtro_MVC+ArchivoActivo#para que escriba el nombre del archivo activo
            else:
                da_normalizar = da_tkeoMVC.sel(
                    ID=da_tkeoMVC.ID.str.contains(filtro_MVC, case=False)
                )
            if len(da_normalizar.ID) == 0:
                print(
                    '\nNo se han encontrado archivos con el filtro de normalización "{0}"'.format(
                        filtro_MVC
                    )
                )
                sys.exit(0)

            # Busca máximo
            daMVCtkeo_Max = da_normalizar.max(dim=["time", "ID"])
            daNomMVCMaxtkeo = da_normalizar.ID[
                da_normalizar.max(dim="time").argmax(dim="ID")
            ]

            da_tkeo = da_tkeo / (daMVCtkeo_Max / 100)
            da_tkeo = xr.concat([da_tkeo, dakinem], dim="n_var")
            da_tkeo.attrs["units"] = "%MVC"
            da_tkeo.time.attrs["units"] = daDinamic.time.attrs["units"]

            da_tkeo = segmenta_ModeloBikefitting_xr_EMG(da_tkeo).sel(
                n_var=orden_musculos
            )  # se queda solo con los músculos por si hay alguna variable cinemática de segmentar
            # da_tkeo.isel(ID=0).sel(n_var='VME').plot.line(x='time', col='side')
            da_tkeo = NormalizaBiela360_xr(da_tkeo)  # , base_norm_horiz=biela')
            # da_tkeo.isel(ID=0).plot.line(x='AngBielaInRepe', col='side', row='n_var', sharey=False)
            # xr.where(da_tkeo<1, da_tkeo, np.nan).isel(ID=0).plot.line(x='AngBielaInRepe', col='side', row='n_var', sharey=False)

            # Ajusta offset glúteos
            nv = "GLU"
            for l in ["L", "R"]:
                resta = da_tkeo.loc[
                    dict(n_var=nv, side=l, AngBielaInRepe=slice(180, 220))
                ].mean(dim=["AngBielaInRepe", "phase"])
                da_tkeo.loc[dict(n_var=nv, side=l)] -= resta

            """
            da_tkeo = xr.apply_ufunc(tkeo, daGraf,
                           input_core_dims=[['AngBielaInRepe']],
                           output_core_dims=[['AngBielaInRepe']],
                           vectorize=True)
            da_tkeo.isel(ID=0).sel(n_var='BIC').plot.line(x='AngBielaInRepe', col='side')
            """

            # Calcula la media de las repes
            # df = dfGraf.groupby(['ID', 'n_var', 'side', 'AngBielaInRepe']).mean().reset_index()
            df = (
                da_tkeo.assign_coords(side=["Izq", "Der"])
                .to_dataframe(name="value")
                .dropna()
                .reset_index()
                .groupby(["ID", "n_var", "side", "AngBielaInRepe"])
                .mean()
                .reset_index()
            )
            umbral_onset = 1
            # Asigna a cada músculo un valor arbitrario
            df["value_cat"] = (
                df["n_var"]
                .replace(dict(zip(orden_musculos, range(0, len(orden_musculos)))))
                .astype("int32")
            )
            df["value"] = df["value_cat"].where(df["value"] >= umbral_onset, np.nan)

            # Da valores automáticamente según el número de categoría que le asigna
            # df['value'] = np.where(df['value']>=umbral_onset, df['n_var'].astype('category').sort_values(ascending=False).cat.codes, np.nan)
            # df['cosa']=df['n_var'].astype('category').sort_values().cat.codes

            """
            #Prepara para gráfica con dataarrays
            daGraf2 = xr.where(daGraf.mean(dim='repe') >= umbral_onset, 1, np.nan)
            escala= xr.full_like(daGraf2, np.nan).isel(AngBielaInRepe=0)
            escala.data = np.repeat(np.arange(1, len(escala.n_var)+1), 2).reshape(1,6,2)
            
            daGraf2 = (daGraf2*escala).assign_coords(dict(AngBielaInRepe_rad = ('AngBielaInRepe', np.deg2rad(np.arange(361))))) #Coords en radianes
            
            df = daGraf2.to_dataframe(name='value').reset_index()#.dropna()
            """

            def draw_muscle(x, y, **kwargs):
                plt.plot(x, y, **kwargs)

            g = sns.FacetGrid(
                df,
                col="side",
                hue="n_var",
                col_order=["Izq", "Der"],
                hue_order=orden_musculos,
                subplot_kws=dict(projection="polar"),
                height=4,
                legend_out=False,
                despine=False,
            )  # , palette=['r', 'lime'])

            g.map_dataframe(
                draw_muscle,
                x="AngBielaInRepe_rad",
                y="value",
                lw=10,
                solid_capstyle="round",
            )
            g.add_legend()
            (
                g.set_axis_labels(
                    "", ""
                )  # pone etiquetas en ejes (las del x todas iguales)
                .set_titles(
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                .set(yticks=[])
                .set(ylim=(-3, len(dfGraf["n_var"].unique())))
                .tight_layout(w_pad=0)  # separación entre gráficas
            )

            # Ajusta leyenda
            sns.move_legend(g, loc="upper right", title="", frameon=True)

            # Reduce el grosor de las líneas de la leyenda
            leg = g._legend
            for legobj in leg.legendHandles:
                legobj.set_linewidth(4.0)

            """                        
            custom_lines = [Line2D([0], [0], lw=3, c='C0', solid_capstyle='round', label='GAS'),
                            Line2D([0], [0], lw=3, c='C1', solid_capstyle='round', label='TIB'),
                            Line2D([0], [0], lw=3, c='C2', solid_capstyle='round', label='REC'),
                            Line2D([0], [0], lw=3, c='C3', solid_capstyle='round', label='VME'),
                            Line2D([0], [0], lw=3, c='C4', solid_capstyle='round', label='BIC'),
                            Line2D([0], [0], lw=3, c='C5', solid_capstyle='round', label='GLU'),
                           ]
            #g.axes[0,-1].legend(handles=custom_lines, bbox_to_anchor=(.95,.95)))#, bbox_transform=ax_ang.transAxes)
            sns.move_legend(g, handles=custom_lines, loc='upper right',  title='', frameon=True)
            """

            for nomvar, ax in g.axes_dict.items():
                # ax.set_xticks(np.linspace(0, 360, 5))
                # Pone el valor 0º arriba
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)

            # plt.tight_layout()
            fig = g.fig

            if carpeta_guardar:
                guarda_grafica(nom="_EON_OnsetMusculatura")
        # ------------------------------------------------

        # Gráficas juntando músculos (por separado lados)
        if "musc_circ" in tipo_graf:
            titulo = ""
            # TODO SEGUIR CON ESTO

            orden_musculos = [
                "GAS",
                "TIB",
                "REC",
                "VME",
                "BIC",
                "GLU",
            ]  # ['GLU', 'BIC', 'VME', 'REC', 'TIB', 'GAS']
            titulo = "Activación muscular"
            # dfGraf = daGraf.to_dataframe(name='value').dropna().reset_index()
            # dfGraf['side'] = dfGraf['side'].replace({'L':'Izq', 'R':'Der'})

            # Calcula la media de las repes
            # df = dfGraf.query('repe==@rango').groupby(['ID', 'n_var', 'side', 'AngBielaInRepe']).mean().reset_index()
            df = dfGraf.query("phase==@rango")

            g = sns.FacetGrid(
                df,
                col="side",
                hue="n_var",
                col_order=["Izq", "Der"],
                hue_order=orden_musculos,
                subplot_kws=dict(projection="polar"),
                height=4,
                legend_out=False,
                despine=False,
            )  # , palette=['r', 'lime'])

            # Draw a scatterplot onto each axes in the grid
            g.map_dataframe(
                sns.lineplot,
                x="AngBielaInRepe_rad",
                y="value",
                estimator=estim,
                errorbar=ci,
                units=unit,
                lw=1,
                solid_capstyle="round",
            )
            g.add_legend()
            (
                g.set_axis_labels(
                    "", ""
                )  # pone etiquetas en ejes (las del x todas iguales)
                .set_titles(
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                .set(yticks=[])
                # .set(ylim=(-3, len(dfGraf['n_var'].unique())))
                .tight_layout(w_pad=0)  # separación entre gráficas
            )

            # Ajusta leyenda
            sns.move_legend(g, loc="upper right", title="", frameon=True)

            # Reduce el grosor de las líneas de la leyenda
            leg = g._legend
            for legobj in leg.legendHandles:
                legobj.set_linewidth(4.0)

            """                        
            custom_lines = [Line2D([0], [0], lw=3, c='C0', solid_capstyle='round', label='GAS'),
                            Line2D([0], [0], lw=3, c='C1', solid_capstyle='round', label='TIB'),
                            Line2D([0], [0], lw=3, c='C2', solid_capstyle='round', label='REC'),
                            Line2D([0], [0], lw=3, c='C3', solid_capstyle='round', label='VME'),
                            Line2D([0], [0], lw=3, c='C4', solid_capstyle='round', label='BIC'),
                            Line2D([0], [0], lw=3, c='C5', solid_capstyle='round', label='GLU'),
                           ]
            #g.axes[0,-1].legend(handles=custom_lines, bbox_to_anchor=(.95,.95)))#, bbox_transform=ax_ang.transAxes)
            sns.move_legend(g, handles=custom_lines, loc='upper right',  title='', frameon=True)
            """
            g.set(theta_zero_location="N", theta_direction=-1)

            # for nomvar, ax in g.axes_dict.items():
            #     #ax.set_xticks(np.linspace(0, 360, 5))
            #     #Pone el valor 0º arriba
            #     ax.set_theta_zero_location('N')
            #     ax.set_theta_direction(-1)

            # plt.tight_layout()
            fig = g.fig

            if carpeta_guardar:
                guarda_grafica(nom="_EMP_OnsetMusculatura")

            # plt.show()

            # -------------------------------------------------

        # Gráficas onset músculos todos agrupados (por separado lados)
        if "lados_lin_grupal" in tipo_graf:
            orden_musculos = [
                "GLU",
                "REC",
                "VME",
                "BIC",
                "GAS",
                "TIB",
            ]  # ['GLU', 'BIC', 'VME', 'REC', 'TIB', 'GAS']
            titulo = "Activación muscular"
            # dfGraf = daGraf.to_dataframe(name='value').dropna().reset_index()
            # dfGraf['side'] = dfGraf['side'].replace({'L':'Izq', 'R':'Der'})

            g = sns.relplot(
                data=dfGraf,
                x="AngBielaInRepe",
                y="value",
                col="n_var",
                col_wrap=3,
                col_order=orden_musculos,
                hue="side",
                estimator=estim,
                errorbar=ci,
                units=unit,
                lw=1,
                palette=["r", "lime"],
                alpha=alpha,
                facet_kws={"sharey": True, "legend_out": False, "despine": False},
                solid_capstyle="round",
                kind="line",
                legend=True,
                aspect=1.2,
                height=3,
                zorder=2,
            )

            if ensemble_avg == "completo":
                # Dibuja todas las repeticiones
                g.map_dataframe(
                    sns.lineplot,
                    x="AngBielaInRepe",
                    y="value",
                    hue="side",
                    estimator=None,
                    errorbar=ci,
                    units="phase",
                    lw=0.25,
                    palette=["r", "lime"],
                    alpha=0.3,
                )

            (
                g.set_titles(
                    col_template="{col_name}"
                ).set_axis_labels(  # , fontweight='bold') #títulos de cada gráfica
                    xlabel="Ángulo de biela (grados)", ylabel="EMG (%MVC)"
                )  # pone etiquetas en ejes (las del x todas iguales)
                # .set(yticks=[])
                # .tight_layout(w_pad=2) #separación entre gráficas
                .set(xlim=(0, 360), xticks=np.linspace(0, 360, 5))
                # .set(ylim=(0, 100))
                .set(
                    xlabel="Ángulo de biela (grados)", ylabel="EMG (%MVC)"
                )  # ($^\circ$)
            )

            # Ajusta leyenda
            # g.add_legend()
            sns.move_legend(
                g, loc="upper right", bbox_to_anchor=(1.0, 1.01), title="", frameon=True
            )  # , bbox_to_anchor=(1.0, 1.01)

            for nomvar, ax in g.axes_dict.items():
                ax.set_title(nfa.traduce_variables(nomvar))
                ax.xaxis.grid(
                    True, linestyle="dashed", dash_capstyle="round", alpha=0.8, zorder=1
                )
                if compara_lados_graf:
                    ti = calcula_spm1d(dfGraf.query("n_var==@nomvar"))
                    # Marca regiones con diferencia L y R
                    plot_clusters(ti, y=5, ax=ax, print_p=False)

            # plt.tight_layout()
            fig = g.fig

            if carpeta_guardar:
                guarda_grafica(nom=f"_EG_Grupo{extra_nom_archivo}")
        # ------------------------------------------------

        # Gráficas onset músculos todos agrupados (por separado lados)
        if "lados_circ_grupal" in tipo_graf:
            orden_musculos = [
                "GLU",
                "REC",
                "VME",
                "BIC",
                "GAS",
                "TIB",
            ]  # ['GLU', 'BIC', 'VME', 'REC', 'TIB', 'GAS']
            titulo = "Activación muscular"
            # dfGraf = daGraf.to_dataframe(name='value').dropna().reset_index()
            # dfGraf['side'] = dfGraf['side'].replace({'L':'Izq', 'R':'Der'})

            g = sns.FacetGrid(
                dfGraf,
                col="n_var",
                col_wrap=3,
                col_order=orden_musculos,
                hue="side",
                subplot_kws=dict(projection="polar"),
                height=4,
                legend_out=False,
                sharex=False,
                sharey=False,
                despine=False,
                palette=["r", "lime"],
            )

            if ensemble_avg == "completo":
                # Dibuja todas las repeticiones
                g.map_dataframe(
                    sns.lineplot,
                    x="AngBielaInRepe_rad",
                    y="value",
                    estimator=None,
                    errorbar=None,
                    units="phase",
                    lw=0.25,
                    palette=["r", "lime"],
                    alpha=0.3,
                )
            # Dibuja gráfica principal (después que línea a línea para que mantenga las características de la leyenda)
            g.map_dataframe(
                sns.lineplot,
                x="AngBielaInRepe_rad",
                y="value",
                estimator=estim,
                errorbar=ci,
                units=unit,
                lw=1,
                alpha=alpha,
            )

            g.add_legend()
            (
                g.set_axis_labels(
                    "", ""
                )  # pone etiquetas en ejes (las del x todas iguales)
                .set_titles(
                    col_template="{col_name}"
                )  # , fontweight='bold') #títulos de cada gráfica
                .tight_layout(w_pad=0)  # separación entre gráficas
            )

            # Pone el valor 0º arriba y giro en sentido horario
            g.set(theta_zero_location="N", theta_direction=-1)

            # Ajusta leyenda
            sns.move_legend(
                g,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.95),
                title="",
                frameon=True,
            )

            for nomvar, ax in g.axes_dict.items():
                ax.set_title(nfa.traduce_variables(nomvar))
                # ax.xaxis.grid(True, linestyle='dashed', dashes=(5, 5), dash_capstyle='round', alpha=0.8, zorder=1)

            plt.tight_layout()
            fig = g.fig

            if carpeta_guardar:
                guarda_grafica(nom=f"_EPG_Grupo{extra_nom_archivo}")
        # ------------------------------------------------

        # TODO: PROBAR ESPECTROGRAMA
        if False:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
            P, freqs, t, im = plt.specgram(
                dataf, NFFT=128, Fs=freq, noverlap=64, cmap=plt.cm.jet
            )
            # P: array of shape (len(times), len(freqs)) of power,
            # freqs: array of frequencies,
            # bins: time points the spectrogram is calculated over,
            # im: matplotlib.image.AxesImage instance
            ax1.set_title("Short-Time Fourier Transform", fontsize=18)
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Frequency [Hz]")
            ax1.set_xlim(t[0], t[-1])
            plt.tight_layout()


def RealizaGraficas_cinem_completo(daGraf) -> None:
    da = daGraf.isel(ID=0)
    RealizaGraficas_triples_cinem(
        da, tipo_graf=["lados_lin"], repes=None, ensemble_avg="completo"
    )  # , carpeta_guardar=carpeta_graficas)


# =============================================================================
# %% Pulir archivos MVC-ISO
# =============================================================================
"""   
ORDEN ARCHIVOS MVCs

00 01 02 03 - Cuádriceps (recto + vasto)
04 05 06 07 - Gastrocnemio
08 09 10 11 - Tibial
12 13 14 15 – Glúteo
16 17 18 19 – Bíceps femoral
"""
# =============================================================================
#########
# Comprobar en la gráfica si están todos correctos, si no, corregir a mano
"""
#Corregir el dato concreto:
daMVC_Max.loc[dict(n_var=['GLU', 'BIC'])] = daMVC_proces.sel(n_var=['GLU', 'BIC'], ID=~daMVC_proces['ID'].str.contains('MVC-02')).max(dim='time').max(dim='ID')
#o poner a cero todo el canal que se ve saturado
daMVC_proces.loc[dict(ID='MVC-02', n_var='REC', side='L')] = 0.0

daMVC_proces.loc[dict(n_var=['GLU', 'BIC'], ID='MVC-02')] = 0.0

"""
##########################################
# Elimina las primeras repeticiones de los sensores desconectados
"""
#Comprobaciones errores medida EMG
daMVC_proces.sel(ID=daMVC_proces.ID[:12].data).plot(x='time', col='side', row='n_var', hue='ID', sharey=False, aspect=3, lw=1, alpha=0.6)
daMVC_proces.sel(ID=daMVC_proces.ID[14:].data).plot(x='time', col='side', row='n_var', hue='ID', sharey=False, aspect=3, lw=1, alpha=0.6)
daMVC_proces.diff(dim='time').plot(x='time', col='side', row='n_var', hue='ID', sharey=False, aspect=3, lw=1, alpha=0.6)

#Con scipy find_peaks
for n, arr in daMVC_proces.sel(n_var='GAS', side='R').groupby('ID'):
    print(n)
    pe=find_peaks(arr, prominence=0.01)#, height=0.008)
    plt.plot(arr)
    plt.plot(pe[0], arr[pe[0]], 'o')
    print(pe[1]['prominences'].mean())



arr=daMVC_proces.isel(ID=3).sel(n_var='GAS', side='L')
from scipy.signal import find_peaks
pe=find_peaks(arr, prominence=0.01)#, height=0.008)
plt.plot(arr)
plt.plot(pe[0], arr[pe[0]], 'o')
print(pe[1]['prominences'].mean())

pe=find_peaks(daMVC_proces, prominence=0.01)#, height=0.008)
xr.apply_ufunc(find_peaks, daMVC_proces, prominence=0.01)
from detect_peaks import detect_peaks
detect_peaks(daMVC_proces.isel(ID=16).sel(n_var='GLU', side='L'),  mpd=500, show=True)
"""

##########################################
# ELIMINA MVCs EN MANIOBRAS NO ESPECÍFICAS
# daMVC_proces_copia = daMVC_proces.copy(deep=True)


def limpia_MVC(daData) -> xr.DataArray:
    from detecta import detect_onset

    # daData.sel(n_var=['GAS']).plot.line(x='time', col='side', row='ID')

    # Elimina repeticiones con GLU y BIC desconectados
    daData.loc[
        dict(
            n_var=["GLU", "BIC"],
            ID=(
                daData.ID.str.contains("-ISO-")
                & ~daData.ID.isin(["MVC-ISO-{0:02d}".format(i) for i in range(12, 20)])
            ),
        )
    ] = np.nan
    # Elimina repeticiones de maniobra TIB en GAS
    daData.loc[
        dict(
            n_var=["GAS"],
            ID=[
                daData.ID.data[0].split("MVC")[0] + "MVC-ISO_{0:02d}".format(i)
                for i in range(8, 12)
            ],
        )
    ] = np.nan
    # Elimina repeticiones de maniobra GAS, BIC en TIB
    daData.loc[
        dict(
            n_var=["TIB"],
            ID=[
                daData.ID.data[0].split("MVC")[0] + "MVC-ISO-{0:02d}".format(i)
                for i in list(range(4, 8)) + list(range(16, 20))
            ],
        )
    ] = np.nan
    # daData.loc[dict(n_var=['TIB'], ID=['MVC-ISO-{0:02d}'.format(i) for i in list(range(4,8))+list(range(16,20))])] = np.nan

    ##########################################
    ##########################################
    # Elimina algunos datos finales para quitar picos raros
    def quita_final(data, num, nom_archivo):
        # print(len(data), data.shape)
        data1 = data.copy()

        # quita num datos del final
        try:
            # data1 = data[~np.isnan(data)][:-num]
            # data1 = np.append(data1, np.zeros(len(data)-len(data1)))
            data1[
                len(data1[~np.isnan(data1)]) - num : len(data1[~np.isnan(data1)]) + 0
            ] = np.zeros(
                num
            )  # np.nan
        except:  # Exception as err:
            # print(u'Error de cálculo', err)
            # data1=data
            pass

        # Intento de quitar picos iniciales, pero no va
        """
        if 'ISO' in nom_archivo:
            try:
                i=detect_onset(data1, threshold=0.05, n_above=300, threshold2=0.1, n_above2=3000, n_below=100, show=False)
                try:
                    data1[0:i[0]] = 0.0 #np.zeros(i[0])
                    #print('Cambiado')
                except:
                    #print('error')
                    pass
            except:
                pass
        """

        return data1

    # data=daMVC_proces[8,2,0,:].data.copy()
    daData2 = xr.apply_ufunc(
        quita_final,
        daData,
        200,  # num. de datos que quita al final
        daData.ID,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],  # datos que devuelve
        exclude_dims=set(
            ("time",)
        ),  # dimensiones que se permite que cambien (tiene que ser un set)
        vectorize=True,
    ).assign_coords(time=daData.time)

    daData2.attrs = daData.attrs
    daData2.name = daData.name

    return daData2


# =============================================================================
# %%
# =============================================================================
if __name__ == "__main__":
    # if False:
    r"""
    sys.path.append(r'F:\Programacion\Python\Mios\Functions')
    import Nexus_FuncionesApoyo as nfa
    import bikefitting_funciones_apoyo as bfa
    r"""
    # Cargando directamente desde c3d
    ruta_trabajo = Path(
        r"F:\Investigacion\Proyectos\BikeFitting\Bikefitting\EstudioEMG_MVC\Registros\17_Eduardo"
    )

    lista_archivos = list(
        ruta_trabajo.glob("**/*.c3d")
    )  # incluye los que haya en subcarpetas
    lista_archivos = [
        x for x in lista_archivos if "MVC-" not in x.name and "Estatico" not in x.name
    ]
    lista_archivos.sort()

    daDatos = nfa.carga_c3d_generico_xr(
        lista_archivos[:10], section="Trajectories"
    )  # , nom_vars_cargar=['HJC', 'KJC', 'AJC'])
    daDatos = nfa.ajusta_etiquetas_lado_final(daDatos)
    daDatos = nfa.separa_trayectorias_lados(daDatos)
    daAngles = calcula_angulos_desde_trajec(daDatos)
    daPos = calcula_variables_posicion(daDatos)
    daCinem = xr.concat([daAngles, daPos], dim="n_var")

    # Cargando preprocesados
    ruta_trabajo = Path(
        r"F:\Investigacion\Proyectos\BikeFitting\Bikefitting\EstudioEMG_MVC\Registros"
    )

    lista_archivos_Din = list(
        ruta_trabajo.glob("**/*.nc")
    )  # incluye los que haya en subcarpetas
    lista_archivos_Din.sort()
    lista_archivos_Din = [
        x
        for x in lista_archivos_Din
        if "PreprocesadoDinamicCinem_" in x.name and not "_Global" in x.name
    ]

    num_cortes = 12
    base_normaliz = "time"
    daDin_global_cinem = xr.load_dataarray(lista_archivos_Din[0])
    # daDin_global_cinem = daDin_global_cinem.rename(
    #     {"Archivo": "ID", "nom_var": "n_var", "lado": "side", "eje": "axis"}
    # )
    daData_trat = segmenta_ModeloBikefitting_xr_cinem(
        daDin_global_cinem, num_cortes=num_cortes
    )
    daData_trat = NormalizaBiela360_xr(daData_trat, base_norm_horiz=base_normaliz)
    # daData_trat .sel(n_var='AngArtKnee', axis='x', side=['L','R']).plot.line(x='AngBielaInRepe', col='side', row='ID', hue='repe')

    # PROBAR GRÁFICO KOPS, PLANO SAGITAL Y EN VERTICAL POSICIÓN PEDAL O META
    RealizaGraficas_cinem(
        daGraf=daData_trat.isel(ID=0).sel(side=["L", "R"]),
        tipo_graf=["planoXY"],
        repes=None,
        nomvars=["KJC"],
        ejes=list("yz"),
        ensemble_avg=bEnsembleAvg,
        carpeta_guardar=ruta_trabajo / "FigurasGenericas",
    )
