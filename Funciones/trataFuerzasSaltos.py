# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:33:55 2022

@author: josel
"""
# =============================================================================
# %% IMPORTA LIBRERIAS
# =============================================================================

from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import polars as pl
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # para guardar gráficas en pdf
import time  # para cuantificar tiempos de procesado

from pathlib import Path
import sys

sys.path.append(r"F:\Programacion\Python\Mios\Functions")

from detecta import detect_onset

# from scipy.signal import find_peaks #sustituir por detecta?????

__author__ = "Jose Luis Lopez Elvira"
__version__ = "v.1.3.5"
__date__ = "12/05/2024"


# TODO: probar detectar umbrales con scipy.stats.threshold
"""
Modificaciones:
    12/05/2024, v.1.3.5
            - Introducido typo return en funciones (sin probar).
    
    2/02/2024, v.1.3.4
            - Cambiada dimensión 'evento' por 'event'.
    
    21/02/2024, v.1.3.3
            - En la detección del evento despegue reducida la ventana de tener
              que estar por debajo del umbral, de 0.15 a 0.01 s.
    
    11/02/2024, v.1.3.2
            - Cambio nombre chequea_fin_plano a chequea_tramo_plano.
              Mejorado: devuelve datos 'continuo' o 'discreto', con tipo_analisis 'std' o 'delta'.
              Si se piden gráficas las hace con graficas_eventos de la ventana final.
    
    04/02/2024, v.1.3.1
            - En afina_peso, incluido tipo_calculo 'peso_media_salto'. Calcula
              el peso haciendo la media de Fz desde iniAnalisis hasta finAnalisis.
    
    22/01/2024, v.1.3.0
            - Añadidas variables de resultado tiempo de los eventos.
                  
    04/10/2023, v.1.1.4
            - Separada función de pasar de polars a dataarray. Cuando carga
              con polars devuelve xarray con todas las variables que carga.
              Después se separa en dimensión axis y plat en otra función.
              
    22/09/2023, v.1.1.3
            - Corregido que no podía dibujar eventos sin nombre convencional.
    
    18/07/2023, v.1.1.2
            - Corregido cálculo ajusta_offsetFz, con tipo_calculo='vuelo' y
              tipo_test distinto de DJ2P, no devolvía los datos corregidos.
    
    17/06/2023, v.1.1.1
            - En recorta ventana se asegura de que no vengan valores menores
              que cero ni mayores que len(data).
            - Para detectar iniMov en CMJ y SJ, busca superar el umbral por 
              encima o por debajo y coge el que esté antes.
            - Creación de gráficas eventos admite parámetro sharey.
            - Al calcular resultados, cambiado nombre de variables var por n_var,
              para evitar confusiones con el método .var (varianza).
            - Al detectar ini/fin imp posit, se queda con la detección última,
              la más cercana al despegue.
            - Al calcular resultados de impulsos, restaba el peso, pero la
              fuerza está calculada en BW, por lo que ahora resta 1 como peso
              y devuelve el impulso normalizado.
            - En resultados EMG calcula activación en periodo de preactivación,
              un tiempo antes del iniMov. Tiene sentido en los SJ, en CMJ sería
              activación residual antes de empezar a moverse.
    
    20/05/2023, v.1.1.0
            - Introducia función para optimizar el ajuste del peso. Probado con
              método iterativo y con curve fitting. Con polinomio de orden 8 
              funciona bien.
            - Al afinar el peso añade el valor de los residuos a la dimensión
              stat.
    
    12/05/2023, v.1.0.7
            - Corregido el hacer gráficas al calcular el peso.
            - Reparado detecta_minFz para DJ y DJ2P.
            - ajusta_offsetFz_vuelo y reset_Fz_vuelo ahora se hace directamente
              con xarray, haciendo la media por debajo del umbral pedido.
     
    06/05/2023, v.1.0.6
            - Los n_above se calculan en segundos teniendo en cuenta la
              frecuencia.
            - Cuando se pide que ajuste inifin no dibuja eventos de búsqueda
              del peso.
            - Probada nueva versión de detecta_fin_mov basada en la velocidad,
              cuando vuelve a ser cero después del aterrizaje. Corregida
              versión basada en fuerza, cuando después del aterrizaje baja del
              peso y vuelve a subir. Parece que funciona mejor.

    16/04/2023, v.1.0.5
            - Corregido en detect_despegue_aterrizaje que se quede siempre
              con el último que encuentra. Si el último está al final del
              registro, coge el anterior.
    
    15/03/2023, v.1.0.4
            - Al buscar despegue y aterrizaje, si no los encuentra da una
              segunda oportunidad ajustando el umbral al valor mínimo del
              registro.
            - Corrección en show gráfica en reset_Fz_vuelo cuando no hay repes.
            - En función gráficas, introducido parámetro show_in_console para
              controlar si se quiere que dibuje las gráficas en la consola o
              sólo en pdf.
            
    '26/02/2023', v.1.0.3
            - Incluye función para encontrar todos los eventos de una vez.
            - Incluidas funciones para detectar saltos que empiezan o terminan
              en cero. Puede ser diferente la ventana inicial y final.
            - Incluida función que valora si es equiparable una ventana inicial
              a la final.
    
    '09/01/2023', v.1.0.2
            - En la función calcula_variables devuelve la fuerza normalizada
            como otra variable del dataset y mantiene el dtype original.
    
    '22/12/2022', v.1.0.1
            - Corrección, devuelve float al buscar inicio movimiento con DJ.
    
    '10/09/2022', v.1.0.0
            - Versión inicial con varias funciones.
"""


# =============================================================================
# ---- VARIABLES
# =============================================================================
g = 9.81  # m/s2

eventos_basicos = [
    "iniAnalisis",
    "preactiv",
    "iniPeso",
    "finPeso",
    "iniMov",
    "maxFz",
    "minFz",
    "maxV",
    "iniImpPos",
    "maxFlex",
    "finImpPos",
    "despegue",
    "aterrizaje",
    "finMov",
    "finAnalisis",
]


# =============================================================================
# ---- FUNCIONES DE APOYO
# =============================================================================


def crea_eventos_saltos_estandar(daData) -> xr.DataArray:
    return (
        xr.full_like(daData.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": eventos_basicos},
            axis=-1,
        )
    ).copy()


def asigna_subcategorias_xr(da, n_estudio=None) -> xr.DataArray:

    if len(da.ID.to_series().iloc[0].split("_")) == 5:
        da = da.assign_coords(
            estudio=(
                n_estudio
            ),  # "ID", da.ID.to_series().str.split("_").str[0].to_list()),
            particip=("ID", da.ID.to_series().str.split("_").str[1].to_list()),
            tipo=("ID", da.ID.to_series().str.split("_").str[2].to_list()),
            subtipo=("ID", da.ID.to_series().str.split("_").str[3].to_list()),
            repe=("ID", da.ID.to_series().str.split("_").str[4].to_list()),
        )

    elif len(da.ID.to_series().iloc[0].split("_")) == 4:
        if n_estudio is None:
            n_estudio = "X"
        da = da.assign_coords(
            estudio=("ID", [n_estudio] * len(da.ID)),
            particip=("ID", da.ID.to_series().str.split("_").str[0].to_list()),
            tipo=("ID", da.ID.to_series().str.split("_").str[1].to_list()),
            subtipo=("ID", da.ID.to_series().str.split("_").str[2].to_list()),
            repe=("ID", da.ID.to_series().str.split("_").str[3].to_list()),
        )

    """
    #versión basada en df polars
    da = da.assign_coords(estudio=('ID', df.filter(pl.col('time')==0.000).get_column('estudio').to_list()),
                                              particip=('ID', df.filter(pl.col('time')==0.000).get_column('particip').to_list()),
                                              tipo=('ID', df.filter(pl.col('time')==0.000).get_column('tipo').to_list()),
                                              subtipo=('ID', df.filter(pl.col('time')==0.000).get_column('subtipo').to_list()),
                                              repe=('ID', df.filter(pl.col('time')==0.000).get_column('repe').to_list()),
                                              )
    """
    """
    #Versión antigua si no se sabe si hay dimensión repe o no.
    #En la versión actual no hay dimensión repe, se guarda en el ID
    est=[] #un poco complejo, pero si no todos tienen repe=1 no funcionaba bien
    tip=[]
    subtip=[]
    partic=[]
    repe=[]
    #da.ID.str.split(dim='splt', sep='_')
    for n in da.ID:
        #if len(n.str.split('_'))
        partes = n.str.split(dim='splt', sep='_')
        if len(partes)==3:
            est.append(n_estudio)
            partic.append(partes.data[0])
            tip.append(partes.data[1])
            subtip.append(subtipo)
            repe.append(partes.data[-1])
        elif len(partes)==4:
            est.append(n_estudio)
            partic.append(partes.data[0])
            tip.append(partes.data[1])
            subtip.append(partes.data[2])
            repe.append(partes.data[-1])
        elif len(partes)==5:
            est.append(partes.data[0])
            partic.append(partes.data[1])
            tip.append(partes.data[2])
            subtip.append(partes.data[3])
            repe.append(partes.data[-1])
    
    da = da.assign_coords(estudio=('ID', est), particip=('ID', partic), tipo=('ID', tip), subtipo=('ID', subtip), repe=('ID', repe))
    """
    # if 'repe' in da.dims: #solo lo añade si no tiene ya la dimensión repe
    #     da = da.assign_coords(estudio=('ID', est), particip=('ID', partic), tipo=('ID', tip), subtipo=('ID', subtip))
    # else:
    #     da = da.assign_coords(estudio=('ID', est), particip=('ID', partic), tipo=('ID', tip), subtipo=('ID', subtip), repe=('ID', repe))

    return da


# def integra_completo(daDatos, daEventos):
#     def _integra(data, t, ini, fin, ID):
#         if np.isnan(ini) or np.isnan(fin):
#             return np.nan
#         ini = int(ini)
#         fin = int(fin)
#         # print(ID)
#         # plt.plot(data[ini:fin])
#         try:
#             dat = integrate.cumulative_trapezoid(data[ini:fin], t[ini:fin], initial=0)[
#                 -1
#             ]
#         except:
#             # print(f'Fallo al integrar en {ID}')
#             dat = np.nan
#         return dat

#     """
#     data = daDatos[0].data
#     time = daDatos.time.data
#     ini = daEventos[0].isel(event=0).data
#     fin = daEventos[0].isel(event=1).data
#     """
#     daInt = xr.apply_ufunc(
#         _integra,
#         daDatos,
#         daDatos.time,
#         daEventos.isel(event=0),
#         daEventos.isel(event=1),
#         daDatos.ID,
#         input_core_dims=[["time"], ["time"], [], [], []],
#         # output_core_dims=[['time']],
#         exclude_dims=set(("time",)),
#         vectorize=True,
#         # join='exact',
#     )
#     return daInt


# def RMS(daDatos, ventana):
#     """
#     Calcula RMS en dataarray indicando ventana
#     """

#     def rms(data):
#         if np.count_nonzero(~np.isnan(data)) == 0:
#             return np.array(np.nan)
#         data = data[~np.isnan(data)]
#         return np.linalg.norm(data[~np.isnan(data)]) / np.sqrt(len(data))

#     """
#     data = daRecortado[0,0,0,0].data
#     """
#     daRecortado = recorta_ventana_analisis(daDatos, ventana)
#     daRMS = xr.apply_ufunc(
#         rms,
#         daRecortado,
#         input_core_dims=[["time"]],
#         vectorize=True,
#     )
#     return daRMS


def pasa_df_a_da(dfTodos, merge_2_plats=1, n_estudio=None, show=False) -> xr.DataArray:
    if isinstance(dfTodos, pl.DataFrame):
        # Transforma df polars a dataarray con todas las variables cargadas
        vars_leidas = dfTodos.select(
            pl.exclude(
                ["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
            ),
        ).columns

        dfpd = dfTodos.melt(
            id_vars=["ID", "time"], value_vars=vars_leidas, variable_name="n_var"
        ).to_pandas()

    else:  # viene con pandas
        vars_leidas = dfTodos.drop(
            columns=["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
        ).columns
        dfpd = dfTodos

    daTodos = (
        dfpd
        # .drop(columns=["estudio", "tipo", "subtipo", "particip", "repe"])
        # .melt(id_vars=["ID", "time"], value_vars='n_var')#, value_name='value2')
        # pd.melt(dfTodos.to_pandas().drop(columns=['estudio','tipo','subtipo']), id_vars=['ID', 'repe', 'time'], var_name='axis')
        .set_index(["ID", "n_var", "time"])
        .to_xarray()
        .to_array()
        .squeeze("variable")
        .drop_vars("variable")
    )

    daTodos = separa_dim_plats(da=daTodos, merge_2_plats=merge_2_plats)
    # Renombra columnas
    # dfTodos = dfTodos.rename({'abs time (s)':'time', 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    # Asigna coordenadas extra
    daTodos = asigna_subcategorias_xr(daTodos, n_estudio=n_estudio)
    # daTodos = daTodos.assign_coords(estudio=('ID', dfTodos.filter(pl.col('time')==0.000).get_column('estudio').to_list()),
    #                                          particip=('ID', dfTodos.filter(pl.col('time')==0.000).get_column('particip').to_list()),
    #                                          tipo=('ID', dfTodos.filter(pl.col('time')==0.000).get_column('tipo').to_list()),
    #                                          subtipo=('ID', dfTodos.filter(pl.col('time')==0.000).get_column('subtipo').to_list()),
    #                                          repe=('ID', dfTodos.filter(pl.col('time')==0.000).get_column('repe').to_list()),
    #                                          )
    # Ajusta tipo de coordenada tiempo, necesario??
    ###########daTodosArchivos = daTodosArchivos.assign_coords(time=('time', daTodosArchivos.time.astype('float32').values))

    # daTodosArchivos.sel(ID='PCF_SCT05', axis='z').plot.line(x='time', col='repe')
    # daTodosArchivos.assign_coords(time=daTodosArchivos.time.astype('float32'))
    daTodos.attrs = {
        "freq": (np.round(1 / (daTodos.time[1] - daTodos.time[0]), 1)).data,
        "units": "N",
    }
    daTodos.time.attrs["units"] = "s"
    daTodos.name = "Forces"

    return daTodos


# ----Separa dimensión repe
def separa_dim_repe(daData) -> xr.DataArray:
    # Traduce desde ID a dimensión repe
    # Asume que hay 3 repeticiones, seguir probando si funciona con otras numeraciones
    # ¿Devuelve la repe con int o con str?

    # rep0 = daData.sel(ID=daData.ID.str.endswith('1'))
    rep0 = daData.where(daData.repe == daData.repe[0], drop=True)
    rep0 = rep0.assign_coords(
        ID=["_".join(s.split("_")[:-1]) for s in rep0.ID.data.tolist()]
    ).drop_vars(  # rep0.ID.str.rstrip(f'_{rep0.repe[0].data}'))
        "repe"
    )

    rep1 = daData.where(daData.repe == daData.repe[1], drop=True)
    rep1 = rep1.assign_coords(
        ID=["_".join(s.split("_")[:-1]) for s in rep1.ID.data.tolist()]
    ).drop_vars(  # rep1.ID.str.rstrip(f'_{rep1.repe[1].data}'))
        "repe"
    )

    rep2 = daData.where(daData.repe == daData.repe[2], drop=True)
    rep2 = rep2.assign_coords(
        ID=["_".join(s.split("_")[:-1]) for s in rep2.ID.data.tolist()]
    ).drop_vars(  # rep2.ID.str.rstrip(f'_{rep2.repe[2].data}'))
        "repe"
    )

    daData = xr.concat([rep0, rep1, rep2], dim="repe").assign_coords(
        repe=[1, 2, 3]
    )  # .transpose('ID', 'n_var', 'repe', 'time')

    print("CUIDADO: revisar que los números de repetición son correctos")

    return daData


# Versión con nombres repe originales
def separa_dim_repe_hojaregistro(daDatos, hoja_registro=None) -> xr.DataArray:
    # Traduce desde ID a dimensión repe respetando el número de la repe original de la hoja de registro
    if hoja_registro is None:
        print("Debes especificar el Dataframe con la hoja de registro")
        return
    h_r = hoja_registro.iloc[:, 1:].dropna(how="all")

    rep0 = []
    rep1 = []
    rep2 = []
    for S in h_r.index:
        for t in ["CMJ_2", "SJ_0L", "SJ_100L", "SJ_100S"]:
            for r in h_r.filter(regex=t).loc[S]:
                # print(daDatos.sel(ID=daDatos.ID.str.contains(f'{S}_{t}_{r}')).ID.data)
                da = daDatos.sel(ID=daDatos.ID.str.contains(f"{S}_{t}_{r}"))
                reps = h_r.filter(regex=t).loc[S]
                if da.size != 0:
                    reID = "_".join(
                        da.ID.data[0].split("_")[0:-1]
                    )  # ['_'.join(n.split('_')[0:-1]) for n in list(da.ID.data)]
                    if da.ID.str.endswith(reps.iloc[0]):  # len(da.ID) >0:
                        rep0.append(da.assign_coords(ID=[reID]))
                    elif da.ID.str.endswith(reps.iloc[1]):  # len(da.ID) >1:
                        rep1.append(da.assign_coords(ID=[reID]))
                    elif da.ID.str.endswith(reps.iloc[2]):  # len(da.ID) >2:
                        rep2.append(da.assign_coords(ID=[reID]))

    """
    rep0=[]
    rep1=[]
    rep2=[]
    for S in h_r.index:
        for t in ['CMJ_2', 'SJ_0L', 'SJ_100L', 'SJ_100S']:            
            #for r in h_r.filter(regex=t).loc[S]:
                #print(daDatos.sel(ID=daDatos.ID.str.contains(f'{S}_{t}_')).ID.data)
            da = daDatos.sel(ID=daDatos.ID.str.contains(f'{S}_{t}_'))
            reps = h_r.filter(regex=t).loc[S]
            if da.size!=0:
                reID = '_'.join(da.ID.data[0].split('_')[0:-1]) #['_'.join(n.split('_')[0:-1]) for n in list(da.ID.data)]
                if da.ID.str.endswith(reps[0]): # len(da.ID) >0:
                    rep0.append(da.isel(ID=0).assign_coords(ID=reID))
                if da.ID.str.endswith(reps[1]): # len(da.ID) >1:
                    rep1.append(da.isel(ID=1).assign_coords(ID=reID))
                if da.ID.str.endswith(reps[2]): # len(da.ID) >2:
                    rep2.append(da.isel(ID=2).assign_coords(ID=reID))
    """
    rep0 = xr.concat(rep0, dim="ID").drop_vars("repe")
    rep1 = xr.concat(rep1, dim="ID").drop_vars("repe")
    rep2 = xr.concat(rep2, dim="ID").drop_vars("repe")

    daDatos = xr.concat([rep0, rep1, rep2], dim="repe").assign_coords(repe=[0, 1, 2])

    try:
        if (
            np.char.find(
                daDatos.n_var.astype(str).data, "EMG", start=0, end=None
            ).mean()
            != -1
        ):  # si es EMG
            daDatos = daDatos.transpose("ID", "n_var", "repe", "time")
        else:
            daDatos = daDatos.transpose(
                "ID", "n_var", "repe", "axis", "time"
            )  # si no es EMG
    except:
        print("No se han podido ordenar las dimensiones")

    return daDatos


# ----Separa dimensión plataformas
def separa_dim_plats(da, merge_2_plats) -> xr.DataArray:
    if "Fz.1" in da.n_var.data:  # pandas
        n_vars_plat2 = ["Fx.1", "Fy.1", "Fz.1"]
    else:  # polars
        n_vars_plat2 = ["Fx_duplicated_0", "Fy_duplicated_0", "Fz_duplicated_0"]

    # Separa el dataarray en plataformas creando una dimensión
    # Si hay 2 plataformas las agrupa en una
    if merge_2_plats == 0:
        plat1 = da.sel(n_var=["Fx", "Fy", "Fz"]).assign_coords(n_var=["x", "y", "z"])
        plat2 = da.sel(n_var=n_vars_plat2).assign_coords(n_var=["x", "y", "z"])
        da = (
            xr.concat([plat1, plat2], dim="plat").assign_coords(plat=[1, 2])
            # .transpose("ID", "plat", "repe", "axis", "time")
        )

    elif merge_2_plats == 1:
        da = da.sel(n_var=["Fx", "Fy", "Fz"]).assign_coords(n_var=["x", "y", "z"])

    elif merge_2_plats == 2:
        da = da.sel(n_var=n_vars_plat2).assign_coords(n_var=["x", "y", "z"])

    elif merge_2_plats == 3:
        plat1 = da.sel(n_var=["Fx", "Fy", "Fz"]).assign_coords(n_var=["x", "y", "z"])
        plat2 = da.sel(n_var=n_vars_plat2).assign_coords(n_var=["x", "y", "z"])
        da = plat1 + plat2

    da = da.rename({"n_var": "axis"})
    return da


def compute_forces_axes(da) -> xr.DataArray:
    """
    Calcula las fuerzas en ejes a partir de las fuerzas raw de cada sensor.
    Usar después de read_kistler_c3d_xr()
    """

    def _split_plataforms(da):
        plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
        plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))

        plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
        plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))

        da = xr.concat([plat1, plat2], dim="plat").assign_coords(
            plat=["plat1", "plat2"]
        )

        return da

    if "plat" not in da.coords:
        da = _split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains("x")).sum(dim="n_var")
    Fy = da.sel(n_var=da.n_var.str.contains("y")).sum(dim="n_var")
    Fz = da.sel(n_var=da.n_var.str.contains("z")).sum(dim="n_var")

    daReturn = xr.concat([Fx, Fy, Fz], dim="axis").assign_coords(axis=["x", "y", "z"])
    # daReturn.plot.line(x='time', col='plat')

    return daReturn


def separa_dim_axis(da) -> xr.DataArray:
    # Separa el xarray en ejes creando dimensión axis

    x = da.sel(n_var=da.n_var.str.contains("x")).rename({"n_var": "axis"})
    y = da.sel(n_var=da.n_var.str.contains("y")).rename({"n_var": "axis"})
    z = da.sel(n_var=da.n_var.str.contains("z")).rename({"n_var": "axis"})
    da = (
        xr.concat([x, y, z], dim="axis")
        # .assign_coords(n_var="plat1")
        # .assign_coords(axis=["x", "y", "z"])
        .expand_dims({"n_var": 1})
    )

    return da


def carga_peso_bioware_csv(file) -> float:
    peso = 0.0
    with open(file, mode="rt") as f:
        num_lin = 0

        # Scrolls through the entire file to find the start and end of the section and the number of lines
        for linea in f:
            num_lin += 1
            # Search for section start label
            if "Normalized force (N):" in linea:
                peso = float(linea.split("\t")[1])
                break
            num_lin += 1
    if peso == 0.0:
        print("No se encontro la etiqueta de peso en el archivo")
    return peso


# Carga un archivo de Bioware como dataframe de Polars
def carga_bioware_pl(
    file, lin_header=17, n_vars_load=None, to_dataarray=False
) -> pl.DataFrame:
    df = (
        pl.read_csv(
            file,
            has_header=True,
            skip_rows=lin_header,
            skip_rows_after_header=1,
            columns=n_vars_load,
            separator="\t",
        )  # , columns=nom_vars_cargar)
        # .slice(1, None) #quita la fila de unidades (N) #no hace falta con skip_rows_after_header=1
        # .select(pl.col(n_vars_load))
        # .rename({'abs time (s)':'time'}) #'Fx':'x', 'Fy':'y', 'Fz':'z',
        #          #'Fx_duplicated_0':'x_duplicated_0', 'Fy_duplicated_0':'y_duplicated_0', 'Fz_duplicated_0':'z'
        #          })
    ).with_columns(pl.all().cast(pl.Float64()))

    # ----Transform polars to xarray
    if to_dataarray:
        x = df.select(pl.col("^*Fx$")).to_numpy()
        y = df.select(pl.col("^*Fy$")).to_numpy()
        z = df.select(pl.col("^*Fz$")).to_numpy()
        data = np.stack([x, y, z])
        freq = 1 / (df[1, "abs time (s)"] - df[0, "abs time (s)"])
        ending = -3
        coords = {
            "axis": ["x", "y", "z"],
            "time": np.arange(data.shape[1]) / freq,
            "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            )
            .astype(float)
            .transpose("n_var", "axis", "time")
        )
        da.name = "Forces"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "N"

        return da

    return df


def carga_bioware_arrow(
    file, lin_header=17, n_vars_load=None, to_dataarray=False
) -> pd.dataframe:
    # En prueba, no funciona cuando hay cols repetidas
    from pyarrow import csv

    read_options = csv.ReadOptions(
        # column_names=['Fx', 'Fy', 'Fz'],
        skip_rows=lin_header,
        skip_rows_after_names=1,
    )
    parse_options = csv.ParseOptions(delimiter="\t")
    data = csv.read_csv(file, read_options=read_options, parse_options=parse_options)
    data.to_pandas()


def carga_bioware_pd(
    file, lin_header=17, n_vars_load=None, to_dataarray=False
) -> pd.DataFrame | xr.DataArray:

    df = (
        pd.read_csv(
            file,
            header=lin_header,
            usecols=n_vars_load,  # ['Fx', 'Fy', 'Fz', 'Fx.1', 'Fy.1', 'Fz.1'], #n_vars_load,
            # skiprows=18,
            delimiter="\t",
            # dtype=np.float64,
            engine="c",  # "pyarrow" con pyarrow no funciona bien de momento cargar columnas con nombre repetido,
        ).drop(index=0)
        # , columns=nom_vars_cargar)
        # .slice(1, None) #quita la fila de unidades (N) #no hace falta con skip_rows_after_header=1
        # .select(pl.col(n_vars_load))
        # .rename({'abs time (s)':'time'}) #'Fx':'x', 'Fy':'y', 'Fz':'z',
        #          #'Fx_duplicated_0':'x_duplicated_0', 'Fy_duplicated_0':'y_duplicated_0', 'Fz_duplicated_0':'z'
        #          })
    )
    # df.dtypes

    # ----Transform pandas to xarray
    if to_dataarray:
        x = df.filter(regex="Fx*")  # .to_numpy()
        y = df.filter(regex="Fy*")
        z = df.filter(regex="Fx*")
        data = np.stack([x, y, z])
        freq = 1 / (df.loc[2, "abs time (s)"] - df.loc[1, "abs time (s)"])
        ending = -3
        coords = {
            "axis": ["x", "y", "z"],
            "time": np.arange(data.shape[1]) / freq,
            "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            )
            .astype(float)
            .transpose("n_var", "axis", "time")
        )
        da.name = "Forces"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "N"

        return da

    return df


# Carga un archivo Bioware C3D a xarray
def carga_bioware_c3d(file, lin_header=17, n_vars_load=None) -> xr.DataArray:
    from read_kistler_c3d import read_kistler_c3d_xr

    # from read_vicon_c3d import read_vicon_c3d_xr, read_vicon_c3d_xr_global

    da = read_kistler_c3d_xr(file)

    return da


def load_merge_vicon_csv(
    ruta,
    section=None,
    n_vars_load=None,
    n_estudio=None,
    tipo_datos=None,
    asigna_subcat=True,
    show=False,
) -> xr.DataArray:
    from readViconCsv import read_vicon_csv_pl_xr

    lista_archivos = sorted(
        list(ruta.glob("**/*.csv"))
    )  # incluye los que haya en subcarpetas
    lista_archivos = [
        x for x in lista_archivos if "error" not in x.name
    ]  # selecciona archivos
    # lista_archivos.sort()

    print("Cargando los archivos...")
    timer_carga = time.perf_timer()  # inicia el contador de tiempo

    daTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    num_archivos_procesados = 0
    for n, file in enumerate(lista_archivos):
        print(f"Cargando archivo: {n}/{len(lista_archivos)} {file.name}")
        try:
            daProvis = read_vicon_csv_pl_xr(
                file, section=section, n_vars_load=n_vars_load
            ).expand_dims(
                {"ID": ["_".join(file.stem.split("_"))]}, axis=0
            )  # Añade columna ID
            daTodos.append(daProvis)
            # daProvis.isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
            num_archivos_procesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(f"{file.name}  {str(err)}")
            continue

    print(
        f"Cargados {num_archivos_procesados} archivos de {len(lista_archivos)} en {time.perf_timer() - timer_carga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    # Agrupa
    daTodos = xr.concat(daTodos, dim="ID")
    # daTodos.sel(axis='z').plot.line(x='time', col='ID', col_wrap=3)

    # Llama asignar subcategorías aquí o después en parte principal?
    if asigna_subcat:
        daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=n_estudio)

    return daTodos


def load_merge_vicon_csv_selectivo(
    ruta,
    hoja_registro=None,
    section=None,
    n_vars_load=None,
    n_estudio=None,
    tipo_datos=None,
    show=False,
) -> xr.DataArray:
    # Carga listado de archivos basado en hoja de registro

    from readViconCsv import read_vicon_csv_pl_xr

    if hoja_registro is None:
        print("Debes especificar la hoja de registro")
        return
    h_r = hoja_registro.iloc[:, 1:].dropna(how="all")
    # lista_archivos = sorted(list(ruta.glob('**/*.csv')))#incluye los que haya en subcarpetas
    # lista_archivos = [x for x in lista_archivos if 'error' not in  x.name] #selecciona archivos
    # lista_archivos.sort()

    print("Cargando los archivos...")
    timer_carga = time.time()  # inicia el contador de tiempo

    daTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    num_archivos_procesados = 0

    for S in h_r.index:
        for t in ["CMJ_2", "SJ_0L", "SJ_100L", "SJ_100S"]:
            for r in h_r.filter(regex=t).loc[S]:
                if r is not np.nan:
                    file = Path((ruta / f"{S}_{t}_{r}").with_suffix(".csv"))
                    print(f"Cargando sección {section}, archivo: {file.name}")
                    # print(f'{S}_{t}_{r}', f'{S}_{t}_{r}' in [x.stem for x in lista_archivos])
                    try:
                        daProvis = read_vicon_csv_pl_xr(
                            file, section=section, n_vars_load=n_vars_load
                        ).expand_dims(
                            {"ID": ["_".join(file.stem.split("_"))]}, axis=0
                        )  # Añade dimensión ID
                        daTodos.append(daProvis)
                        # daProvis.isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
                        num_archivos_procesados += 1

                    except Exception as err:  # Si falla anota un error y continua
                        print(
                            "\nATENCIÓN. No se ha podido procesar " + file.name,
                            err,
                            "\n",
                        )
                        ErroresArchivos.append(f"{file.name}  {str(err)}")
                        continue

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            num_archivos_procesados, time.time() - timer_carga
        )
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    # Agrupa
    daTodos = xr.concat(daTodos, dim="ID")
    # daTodos.sel(axis='z').plot.line(x='time', col='ID', col_wrap=3)

    # Llama asignar subcategorías aquí o después en parte principal?
    daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=n_estudio)

    return daTodos


def load_merge_vicon_c3d_selectivo(
    ruta,
    hoja_registro=None,
    section=None,
    n_vars_load=None,
    tipo_datos=None,
    n_estudio=None,
    show=False,
) -> xr.DataArray:
    from read_vicon_c3d import read_vicon_c3d_xr  # , read_vicon_c3d_xr_global

    if hoja_registro is None:
        print("Debes especificar la hoja de registro")
        return
    h_r = hoja_registro.iloc[:, 1:].dropna(how="all")
    # lista_archivos = sorted(list(ruta.glob('**/*.csv')))#incluye los que haya en subcarpetas
    # lista_archivos = [x for x in lista_archivos if 'error' not in  x.name] #selecciona archivos
    # lista_archivos.sort()

    print("Cargando los archivos...")
    timer_carga = time.time()  # inicia el contador de tiempo

    daTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    num_archivos_procesados = 0

    for S in h_r.index:
        for t in ["CMJ_2", "SJ_0L", "SJ_100L", "SJ_100S"]:
            for r in h_r.filter(regex=t).loc[S]:
                if r is not np.nan:
                    file = Path((ruta / f"{S}_{t}_{r}").with_suffix(".c3d"))
                    print(f"Cargando sección {section}, archivo: {file.name}")
                    # print(f'{S}_{t}_{r}', f'{S}_{t}_{r}' in [x.stem for x in lista_archivos])
                    try:
                        daProvis = read_vicon_c3d_xr(
                            file, section=section, n_vars_load=n_vars_load
                        ).expand_dims(
                            {"ID": ["_".join(file.stem.split("_"))]}, axis=0
                        )  # Añade dimensión ID
                        daTodos.append(daProvis)
                        """
                        dsProvis['Trajectories'].isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
                        dsProvis['Modeled'].isel(ID=0, axis=-1).plot.line(x='time')
                        dsProvis['Forces'].isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
                        dsProvis['EMG'].isel(ID=0, n_var=0).plot.line(x='time')
                        """
                        num_archivos_procesados += 1

                    except Exception as err:  # Si falla anota un error y continua
                        print(
                            "\nATENCIÓN. No se ha podido procesar " + file.name,
                            err,
                            "\n",
                        )
                        ErroresArchivos.append(f"{file.name}  {str(err)}")
                        continue

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            num_archivos_procesados, time.time() - timer_carga
        )
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    # Agrupa
    daTodos = xr.concat(daTodos, dim="ID")
    # daTodos.sel(axis='z').plot.line(x='time', col='ID', col_wrap=3)

    daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=n_estudio)

    return daTodos


def load_merge_bioware_pl(
    ruta,
    n_vars_load=None,
    n_estudio=None,
    data_type=None,
    lin_header=17,
    merge_2_plats=1,
    show=False,
) -> xr.DataArray:
    """
    Parameters
    ----------
    ruta_trabajo : TYPE
        DESCRIPTION.
    n_vars_load : TYPE, optional
        DESCRIPTION. The default is None.
    n_estudio : string, optional
        DESCRIPTION. The name of the study.
    data_type:
        Conversión al tipo de datos indicado. Por defecto es None, que quiere
        decir que se mantiene el tipo original ('float64')
    lin_header : TYPE, optional
        DESCRIPTION. The default is 17.
    merge_2_plats : int, optional
        What to do with more than 1 platform.
        0: keep the twoo platforms apart.
        1: only platform 1.
        2: only plataform 2.
        3: merge both paltforms as one.
        The default is 1.
    show : bool, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    daTodosArchivos : xarray DataArray
        DESCRIPTION.
    dfTodosArchivos : pandas DataFrame
        DESCRIPTION.

    """
    if data_type is None:
        data_type = float

    lista_archivos = sorted(
        list(ruta.glob("**/*.txt"))
    )  #'**/*.txt' incluye los que haya en subcarpetas
    lista_archivos = [
        x for x in lista_archivos if "error" not in x.name
    ]  # selecciona archivos

    if n_vars_load is None:  # si no vienen impuestas las columnas a cargar
        n_vars_load = ["abs time (s)"]  # , 'Fx', 'Fy', 'Fz']
        if merge_2_plats != 2:  # in [0,1]:
            n_vars_load += ["Fx", "Fy", "Fz"]  # ['Fx.1', 'Fy.1', 'Fz.1']
            if merge_2_plats != 1:
                n_vars_load += [
                    "Fx_duplicated_0",
                    "Fy_duplicated_0",
                    "Fz_duplicated_0",
                ]  # ['Fx.1', 'Fy.1', 'Fz.1']
        else:
            n_vars_load += [
                "Fx_duplicated_0",
                "Fy_duplicated_0",
                "Fz_duplicated_0",
            ]  # ['Fx.1', 'Fy.1', 'Fz.1']

    print("\nCargando los archivos...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    numArchivosProcesados = 0
    dfTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daTodos = []
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(lista_archivos):
        print(f"Cargando archivo nº {nf}/{len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = carga_bioware_pl(file, lin_header, n_vars_load)

            if len(file.stem.split("_")) == 5:
                n_estudio = file.stem.split("_")[0] if n_estudio is None else n_estudio
                particip = file.stem.split("_")[-4]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 4:
                # estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 3:
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-2]
                subtipo = "X"
            if n_estudio is None:
                n_estudio = "EstudioX"

            repe = str(int(file.stem.split("_")[-1]))  # int(file.stem.split('.')[0][-1]
            ID = f"{particip}_{tipo}_{subtipo}_{repe}"  # f'{n_estudio}_{file.stem.split("_")[0]}_{tipo}_{subtipo}'

            # freq = np.round(1/(dfProvis['time'][1]-dfProvis['time'][0]),1)

            # Añade categorías
            dfProvis = dfProvis.with_columns(
                [
                    pl.lit(n_estudio).alias("estudio"),
                    pl.lit(tipo).alias("tipo"),
                    pl.lit(subtipo).alias("subtipo"),
                    pl.lit(ID).alias("ID"),
                    pl.lit(particip).alias("particip"),
                    pl.lit(repe).alias("repe"),
                ]
            )  # .select(['estudio', 'tipo', 'subtipo', 'ID', 'repe'] + nom_vars_cargar)

            dfTodos.append(dfProvis)

            print(f"{dfTodos[-1].shape[0]} filas y {dfTodos[-1].shape[1]} columnas")
            print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATENCIÓN. No se ha podido procesar {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            ErroresArchivos.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfTodos = pl.concat(dfTodos)

    print(
        f"Cargados {numArchivosProcesados} archivos en {time.perf_counter()-timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    if isinstance(
        data_type, str
    ):  # si se ha definido algún tipo de datos, por defecto es 'float64'
        cast = pl.Float32() if data_type == "float32" else pl.Float64()
        dfTodos = dfTodos.select(
            # pl.col(['n_estudio', 'tipo', 'subtipo', 'ID', 'repe']),
            pl.exclude(n_vars_load),
            pl.col(n_vars_load).cast(cast),
        )
    dfTodos = dfTodos.rename(
        {"abs time (s)": "time"}
    )  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daTodos = pasa_df_a_da(dfTodos, merge_2_plats=merge_2_plats, n_estudio=n_estudio)

    # TODO: SEGUIR HACIENDO QUE SI CARGA VARIABLES DISTINTAS DE LAS CONVENCIONALES LAS PASE A DAARRAY PERO SIN EJES
    # if  dfTodos.columns == ['abs time (s)', 'Fx', 'Fy', 'Fz'] or n_vars_load == ['abs time (s)', 'Fx', 'Fy', 'Fz', 'Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0']:
    #     daTodos = pasa_df_a_da(dfTodos, merge_2_plats=merge_2_plats, show=show)

    """    
    #Transforma a pandas y a dataarray
    daTodos = (dfTodos.drop(['estudio','tipo','subtipo'])
                .melt(id_vars=['ID', 'repe', 'time'], variable_name='axis')
                .to_pandas()
                .set_index(['ID','repe', 'axis', 'time'])
                .to_xarray().to_array()
                .squeeze('variable').drop_vars('variable')
                )
     
    
    #Si hay 2 plataformas las agrupa en una
    if merge_2_plats==0:
        plat1 = (daTodos.sel(axis=['Fx', 'Fy', 'Fz'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        plat2 = (daTodos.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        daTodos = (xr.concat([plat1, plat2], dim='plat')
                    .assign_coords(plat = [1, 2])
                    .transpose('ID', 'plat', 'repe', 'axis', 'time')
                    )
                             
    elif merge_2_plats==1:
        daTodos = (daTodos.sel(axis=['Fx', 'Fy', 'Fz'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
    
    elif merge_2_plats==2:
        daTodos = (daTodos.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
        
                    
    elif merge_2_plats==3:
       plat1 = (daTodos.sel(axis=['Fx', 'Fy', 'Fz'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       plat2 = (daTodos.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       daTodos = plat1 + plat2
    
    daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=estudio, subtipo='2PAP')
    
    daTodos.name = 'Forces'
    daTodos.attrs['units'] = 'N'
    daTodos.attrs['freq'] = freq
    daTodos.time.attrs['units'] = 's'
    """
    return daTodos


def load_merge_bioware_pd(
    ruta,
    n_vars_load=None,
    n_estudio=None,
    data_type=None,
    lin_header=17,
    merge_2_plats=1,
    show=False,
) -> xr.DataArray:
    """
    Parameters
    ----------
    ruta_trabajo : TYPE
        DESCRIPTION.
    n_vars_load : TYPE, optional
        DESCRIPTION. The default is None.
    n_estudio : string, optional
        DESCRIPTION. The name of the study.
    data_type:
        Conversión al tipo de datos indicado. Por defecto es None, que quiere
        decir que se mantiene el tipo original ('float64')
    lin_header : TYPE, optional
        DESCRIPTION. The default is 17.
    merge_2_plats : int, optional
        What to do with more than 1 platform.
        0: keep the twoo platforms apart.
        1: only platform 1.
        2: only plataform 2.
        3: merge both paltforms as one.
        The default is 1.
    show : bool, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    daTodosArchivos : xarray DataArray
        DESCRIPTION.
    dfTodosArchivos : pandas DataFrame
        DESCRIPTION.

    """
    if data_type is None:
        data_type = float

    lista_archivos = sorted(
        list(ruta.glob("**/*.txt"))
    )  #'**/*.txt' incluye los que haya en subcarpetas
    lista_archivos = [
        x for x in lista_archivos if "error" not in x.name
    ]  # selecciona archivos

    if n_vars_load is None:  # si no vienen impuestas las columnas a cargar
        n_vars_load = ["abs time (s)"]  # , 'Fx', 'Fy', 'Fz']
        if merge_2_plats != 2:  # in [0,1]:
            n_vars_load += ["Fx", "Fy", "Fz"]
            if merge_2_plats != 1:
                n_vars_load += [
                    "Fx.1",
                    "Fy.1",
                    "Fz.1",
                ]
        else:
            n_vars_load += [
                "Fx.1",
                "Fy.1",
                "Fz.1",
            ]

    print("\nCargando los archivos...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    numArchivosProcesados = 0
    dfTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daTodos = []
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(lista_archivos):
        print(f"Cargando archivo nº {nf}/{len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = carga_bioware_pd(file, lin_header, n_vars_load)

            if len(file.stem.split("_")) == 5:
                n_estudio = file.stem.split("_")[0] if n_estudio is None else n_estudio
                particip = file.stem.split("_")[-4]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 4:
                # estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 3:
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-2]
                subtipo = "X"
            if n_estudio is None:
                n_estudio = "EstudioX"

            repe = str(int(file.stem.split("_")[-1]))  # int(file.stem.split('.')[0][-1]
            ID = f"{particip}_{tipo}_{subtipo}_{repe}"  # f'{n_estudio}_{file.stem.split("_")[0]}_{tipo}_{subtipo}'

            # freq = np.round(1/(dfProvis['time'][1]-dfProvis['time'][0]),1)

            # Añade categorías
            dfProvis.insert(0, "repe", repe)
            dfProvis.insert(0, "particip", particip)
            dfProvis.insert(0, "subtipo", subtipo)
            dfProvis.insert(0, "tipo", tipo)
            dfProvis.insert(0, "ID", ID)
            dfProvis.insert(0, "estudio", n_estudio)

            dfTodos.append(dfProvis)

            print(f"{dfTodos[-1].shape[0]} filas y {dfTodos[-1].shape[1]} columnas")
            print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATENCIÓN. No se ha podido procesar {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            ErroresArchivos.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfTodos = pd.concat(dfTodos)

    print(
        f"Cargados {numArchivosProcesados} archivos en {time.perf_counter()-timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    if isinstance(
        data_type, str
    ):  # si se ha definido algún tipo de datos, por defecto es 'float64'
        dfTodos = dfTodos.astype(data_type)

    dfTodos = dfTodos.rename(
        columns={"abs time (s)": "time"}
    )  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daTodos = pasa_dfpd_a_da(dfTodos, merge_2_plats=merge_2_plats, n_estudio=n_estudio)

    # TODO: SEGUIR HACIENDO QUE SI CARGA VARIABLES DISTINTAS DE LAS CONVENCIONALES LAS PASE A DAARRAY PERO SIN EJES
    # if  dfTodos.columns == ['abs time (s)', 'Fx', 'Fy', 'Fz'] or n_vars_load == ['abs time (s)', 'Fx', 'Fy', 'Fz', 'Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0']:
    #     daTodos = pasa_df_a_da(dfTodos, merge_2_plats=merge_2_plats, show=show)

    """    
    #Transforma a pandas y a dataarray
    daTodos = (dfTodos.drop(['estudio','tipo','subtipo'])
                .melt(id_vars=['ID', 'repe', 'time'], variable_name='axis')
                .to_pandas()
                .set_index(['ID','repe', 'axis', 'time'])
                .to_xarray().to_array()
                .squeeze('variable').drop_vars('variable')
                )
     
    
    #Si hay 2 plataformas las agrupa en una
    if merge_2_plats==0:
        plat1 = (daTodos.sel(axis=['Fx', 'Fy', 'Fz'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        plat2 = (daTodos.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        daTodos = (xr.concat([plat1, plat2], dim='plat')
                    .assign_coords(plat = [1, 2])
                    .transpose('ID', 'plat', 'repe', 'axis', 'time')
                    )
                             
    elif merge_2_plats==1:
        daTodos = (daTodos.sel(axis=['Fx', 'Fy', 'Fz'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
    
    elif merge_2_plats==2:
        daTodos = (daTodos.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
        
                    
    elif merge_2_plats==3:
       plat1 = (daTodos.sel(axis=['Fx', 'Fy', 'Fz'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       plat2 = (daTodos.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       daTodos = plat1 + plat2
    
    daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=estudio, subtipo='2PAP')
    
    daTodos.name = 'Forces'
    daTodos.attrs['units'] = 'N'
    daTodos.attrs['freq'] = freq
    daTodos.time.attrs['units'] = 's'
    """
    return daTodos


def load_merge_bioware_c3d(
    ruta,
    n_vars_load=None,
    n_estudio=None,
    data_type=None,
    split_plats=False,
    merge_2_plats=1,
    asigna_subcat=True,
    show=False,
) -> xr.DataArray:
    # from read_kistler_c3d import read_kistler_c3d_xr
    import read_kistler_c3d as rkc3d

    # ruta = Path(r'F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros\2023PotenciaDJ\S01')
    if data_type is None:
        data_type = float

    lista_archivos = sorted(
        list(ruta.glob("*.c3d"))
    )  #'**/*.txt' incluye los que haya en subcarpetas
    lista_archivos = [
        x for x in lista_archivos if "error" not in x.name
    ]  # selecciona archivos

    """if n_vars_load is None: #si no vienen impuestas las columnas a cargar
        n_vars_load = ['abs time (s)'] #, 'Fx', 'Fy', 'Fz']
        if n_vars_load !=2: #in [0,1]:
            n_vars_load += ['Fx', 'Fy', 'Fz'] #['Fx.1', 'Fy.1', 'Fz.1']
            if n_vars_load != 1:
                n_vars_load += ['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'] #['Fx.1', 'Fy.1', 'Fz.1']
        else:
            n_vars_load += ['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'] #['Fx.1', 'Fy.1', 'Fz.1']
    """

    print("\nCargando los archivos...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    numArchivosProcesados = 0
    daTodos = []
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(lista_archivos):
        print(f"Cargando archivo nº {nf+1}/{len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            """
            #Asigna etiquetas de categorías                   
            if len(file.stem.split("_")) == 5:
                estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[-4]
                tipo = file.stem.split('_')[-3]
                subtipo = file.stem.split('_')[-2]
            elif len(file.stem.split("_")) == 4:
                #estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[0]
                tipo = file.stem.split('_')[-3]
                subtipo = file.stem.split('_')[-2]
            elif len(file.stem.split("_")) == 3:
                particip = file.stem.split("_")[0]
                tipo = file.stem.split('_')[-2]
                subtipo = 'X'
            if n_estudio is None:
                estudio = 'EstudioX'
            
            repe = str(int(file.stem.split('_')[-1])) #int(file.stem.split('.')[0][-1]
            ID = f'{estudio}_{particip}_{tipo}_{subtipo}_{repe}' #f'{estudio}_{file.stem.split("_")[0]}_{tipo}_{subtipo}'
            """
            daProvis = rkc3d.read_kistler_c3d_xr(file).expand_dims(
                {"ID": ["_".join(file.stem.split("_"))]}, axis=0
            )  # Añade columna ID

            daTodos.append(daProvis)

            print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATENCIÓN. No se ha podido procesar {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            ErroresArchivos.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    daTodos = xr.concat(daTodos, dim="ID")

    print(
        f"Cargados {numArchivosProcesados} archivos en {time.perf_counter()-timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    if isinstance(
        data_type, str
    ):  # si se ha definido algún tipo de datos, por defecto es 'float64'
        daTodos = daTodos.astype(data_type)

    if merge_2_plats == 0:  # split_plats:
        daTodos = rkc3d.split_plataforms(daTodos)

    # Llama asignar subcategorías aquí o después en parte principal?
    if asigna_subcat:
        daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=n_estudio)

    # daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=estudio, subtipo='2PAP')

    daTodos.name = "Forces"

    return daTodos


def carga_preprocesados(
    ruta_trabajo, nomArchivoPreprocesado, tipo_test
) -> xr.DataArray:
    if Path((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix(".nc")).is_file():
        tpo = time.time()
        daDatos = xr.load_dataarray(
            (ruta_trabajo / (nomArchivoPreprocesado)).with_suffix(".nc")
        ).sel(tipo=tipo_test)
        print(
            "\nCargado archivo preprocesado ",
            nomArchivoPreprocesado
            + "_Vicon.nc en {0:.3f} s.".format(time.time() - tpo),
        )
    else:
        raise Exception("No se encuentra el archivo preprocesado")
    return daDatos


# =============================================================================
# ---- FUNCIONES PARA FUERZAS SALTOS
# =============================================================================
def corrige_mal_ajuste_cero_medicion(daDatos) -> xr.DataArray:
    def _ajuste_cero(data, ID):
        dat = data.copy()
        try:
            ind = detect_onset(
                -dat,
                threshold=max(-dat) * 0.5,
                n_above=int(0.2 * daDatos.freq),
                show=False,
            )
            recorte_ventana = int((ind[0, 1] - ind[0, 0]) * 10 / 100)
            ind[0, 0] += recorte_ventana
            ind[0, 1] -= recorte_ventana
            peso = dat[ind[0, 0] : ind[0, 1]].mean()
            dat -= peso
        except:
            print(f"ATENCIÓN: No se puede corregir {ID}")
            pass

        return dat

    """
    data = daDatos.data#[0,0,0].data
    """

    daCortado = xr.apply_ufunc(
        _ajuste_cero,
        daDatos,
        daDatos.ID,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # join='outer'
    )  # .dropna(dim='time', how='all')
    # daCortado.attrs = daDatos.attrs
    # daCortado.name = daDatos.name
    # daCortado.sel(axis='z').plot.line(x='time', row='ID', col='axis')

    return daCortado


def chequea_igualdad_ini_fin(
    daDatos, margen=20, ventana=None, retorna=None, show=False
) -> xr.DataArray:
    """Mejor que con .where para cuando hay registros que duran menos,
    rellenos con nan.
    El parámetro retorna puede ser 'resta' (valor numérico), o qualquier otro.
    Con 'resta' retorna la diferencia ini-fin en todos, independientemente del umbral;
    con cualquier otra cosa devuelve el registro completo de los que cumplen las
    condiciones de margen en la resta de la ventana ini-fin.
    ventana: duración de la ventana al final para comprobar. Se puede pasar en segundos (float)
             o en nº de fotogramas (int).
    """

    if ventana is None:
        ventana = [int(0.5 * daDatos.freq), int(0.5 * daDatos.freq)]
    elif isinstance(
        ventana, int
    ):  # si se se envía un entero se asume que en nº de datos
        ventana = [ventana, ventana]
    elif isinstance(ventana, float):  # si se se envía un flozt se asume que en segundos
        ventana = [int(ventana * daDatos.freq), int(ventana * daDatos.freq)]

    def resta_ini_fin_aux(data, margen, ventana0, ventana1, retorna, ID):
        resta = np.nan
        if np.count_nonzero(~np.isnan(data)) == 0:
            return resta
        data = data[~np.isnan(data)]
        return data[:ventana0].mean() - data[-ventana1:].mean()

    if "axis" in daDatos.dims:
        daDatosZ = daDatos.sel(axis="z")
    else:
        daDatosZ = daDatos

    """
    data = daDatosZ[0].data
    """
    daResta = xr.apply_ufunc(
        resta_ini_fin_aux,
        daDatosZ,
        margen,
        ventana[0],
        ventana[1],
        retorna,
        daDatos.ID,
        input_core_dims=[["time"], [], [], [], [], []],
        # output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        # join='outer'
    )

    daCorrectos = xr.where(
        abs(daResta) < margen, daDatosZ.ID, np.nan, keep_attrs=True
    ).dropna("ID")

    """
    def chequea_ini_fin_aux(data, margen, ventana, retorna, ID):
        retID = None if retorna=='nombre' else np.nan
        
        if np.count_nonzero(~np.isnan(data))==0:
            return retID
        
        data = data[~np.isnan(data)]
        diferencia = data[:ventana].mean() - data[-ventana:].mean()
        if retorna=='nombre':
            if abs(diferencia) < margen:
                retID = ID
        else: retID = diferencia
        return retID    
    
    daCorrectos = xr.apply_ufunc(chequea_ini_fin_aux, daDatos, margen, ventana, retorna, daDatos.ID,
                   input_core_dims=[['time'], [], [], [], [] ],
                   #output_core_dims=[['time']],
                   exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   )
    if retorna=='nombre':
        daCorrectos = daCorrectos.dropna('ID')
    """
    if show:
        if retorna == "resta":
            daResta.assign_coords(ID=np.arange(len(daResta.ID))).plot.line(
                x="ID", marker="o"
            )
        else:
            no_cumplen = daDatosZ.loc[dict(ID=~daDatosZ.ID.isin(daCorrectos.ID))]
            if len(no_cumplen.ID) > 0:
                no_cumplen.plot.line(x="time", alpha=0.5, add_legend=False)
                plt.title(
                    f"Gráfica con los {len(no_cumplen)} que no cumplen el criterio"
                )
                # graficas_eventos(no_cumplen)
            else:
                print(r"\nTodos los registros cumplen el criterio")

    return (
        daResta
        if retorna == "resta"
        else daDatos.loc[dict(ID=daDatos.ID.isin(daCorrectos.ID))]
    )


def chequea_tramo_plano(
    daDatos,
    daEvento,
    threshold=30,
    margen_ventana=0.1,
    ventana=0.5,
    retorna="discreto",
    tipo_calculo="std",
    show=False,
) -> xr.DataArray:
    """
    #TODO: FALTA POR COMPROBAR
    Comprueba si el registro ya cortado termina con valores estables de velocidad
    ventana: duración de la ventana al final para comprobar en segundos
    tipo_calculo: 'std'- calcula SD del tramo de la ventana. Idealmente sería = 0.0 para ser horizontal
                  'delta' - calcula la diferencia vertical entre dos partes del final (promedio de una ventana margen)
    threshold: umbral de fuerza admisible (en std o en delta)
    retorna: 'discreto' (default). Devuelve todos los valores del cálculo (std o delta).
             'continuo'. Devuelve time series sólo de los que cumplan el criterio del threshold.
    """
    if tipo_calculo not in ["std", "delta"]:
        raise ValueError(r"tipo_calculo debe ser std o delta")
    if retorna not in ["discreto", "continuo"]:
        raise ValueError(r"retorna debe ser discreto o continuo")

    if "axis" in daDatos.dims:
        daDatosZ = daDatos.sel(axis="z")
    else:
        daDatosZ = daDatos

    if isinstance(ventana, float):
        ventana = int(ventana * daDatos.freq)

    if isinstance(margen_ventana, float):
        margen_ventana = int(margen_ventana * daDatos.freq)

    daEvento = daEvento.sel(ID=daDatos.ID)
    daEventosVentana = (
        xr.full_like(daDatosZ.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": ["iniAnalisis", "finAnalisis"]}, axis=-1
        )
    ).copy()
    if ventana < 0:
        daEventosVentana.loc[dict(event="finAnalisis")] = daEvento
        daEventosVentana.loc[dict(event="iniAnalisis")] = (
            daEvento + ventana
        )  # resta (ventana negativo)
    else:
        daEventosVentana.loc[dict(event="iniAnalisis")] = daEvento
        daEventosVentana.loc[dict(event="finAnalisis")] = daEvento + ventana

    # Recorta a una ventana
    daRecortes = recorta_ventana_analisis(daDatosZ, daEventosVentana)

    if tipo_calculo == "std":
        daStd = daRecortes.std(dim="time")
        daStd.name = "SDFinal"

        daCorrectos = xr.where(
            abs(daStd) < threshold, daDatosZ.ID, np.nan, keep_attrs=True
        ).dropna("ID")

        if show:
            if retorna == "discreto":
                daStd.assign_coords(ID=np.arange(len(daStd.ID))).plot.line(
                    x="ID", marker="o"
                )

            elif retorna == "continuo":
                no_cumplen = daDatosZ.loc[dict(ID=~daDatosZ.ID.isin(daCorrectos.ID))]
                if len(no_cumplen.ID) > 0:
                    no_cumplen.plot.line(x="time", alpha=0.5, add_legend=False)
                    plt.title(
                        f"Gráfica con los {len(no_cumplen)} que no cumplen el criterio {tipo_calculo} < {threshold}",
                        fontsize=10,
                    )
                    graficas_eventos(daRecortes.sel(ID=no_cumplen.ID), sharey=True)
                else:
                    print(r"\nTodos los registros cumplen el criterio")

            # daStd.isel(ID=slice(None)).plot.line(x='time', col='ID', col_wrap=4)

        if retorna == "discreto":
            print(f"Devueltos los {len(daStd)} valores {retorna}")
            return daStd
        elif retorna == "continuo":
            print(f"Devueltos {len(daCorrectos.ID)} registos continuos")
            return daDatos.loc[dict(ID=daDatos.ID.isin(daCorrectos.ID))]

    elif tipo_calculo == "delta":
        """Devuelve la diferencia vertical entre dos partes del final (promedio de una ventana margen)"""

        '''def fin_plano_aux(data, margen_ventana, ventana, retorna, ID):
            resta = np.nan
            if np.count_nonzero(~np.isnan(data))==0:
                return resta
            data = data[~np.isnan(data)]
            return data[-margen_ventana:].mean() - data[-ventana+margen_ventana:].mean()
            
        
        
        """
        data = daDatosZ[0].data
        """
        daDelta = xr.apply_ufunc(fin_plano_aux, daDatosZ, margen_ventana, ventana, retorna, daDatos.ID,
                    input_core_dims=[['time'], [], [], [], [] ],
                    #output_core_dims=[['time']],
                    exclude_dims=set(('time',)),
                    vectorize=True,
                    #join='outer'
                    )
        '''
        daDelta = daRecortes.isel(time=slice(0, margen_ventana)).mean(
            dim="time"
        ) - daRecortes.isel(time=slice(-margen_ventana, None)).mean(dim="time")
        daDelta.name = "DeltaFinal"

        daCorrectos = xr.where(
            abs(daDelta) < threshold, daDatosZ.ID, np.nan, keep_attrs=True
        ).dropna("ID")

        if show:
            if retorna == "discreto":
                daDelta.assign_coords(ID=np.arange(len(daDelta.ID))).plot.line(
                    x="ID", marker="o"
                )
            elif retorna == "continuo":
                no_cumplen = daDatosZ.loc[dict(ID=~daDatosZ.ID.isin(daCorrectos.ID))]
                if len(no_cumplen.ID) > 0:
                    no_cumplen.plot.line(x="time", alpha=0.5, add_legend=False)
                    plt.title(
                        f"Gráfica con los {len(no_cumplen)} que no cumplen el criterio {tipo_calculo} < {threshold}",
                        fontsize=10,
                    )
                    graficas_eventos(daRecortes.sel(ID=no_cumplen.ID), sharey=True)
                else:
                    print(r"\nTodos los registros cumplen el criterio")

        if retorna == "discreto":
            print(f"Devueltos los {len(daDelta)} valores {retorna}")
            return daDelta
        elif retorna == "continuo":
            print(f"Devueltos {len(daCorrectos.ID)} registos continuos")
            return daDatos.loc[dict(ID=daDatos.ID.isin(daCorrectos.ID))]

    print(r"Parece que no se ha especificado el tipo de datos a retornar")


def estima_inifin_analisis(
    daDatos, daEventos, ventana=[1.5, 1.5], tipo_test="CMJ", umbral=20.0, show=False
) -> xr.DataArray:
    """
    Intenta estimar el inicio y final del análisis a partir del centro del vuelo
    ventana: tiempo en segundos antes y después del despegue y el aterrizaje, respectivamente
    """
    if not isinstance(
        ventana, list
    ):  # si se aporta un solo valor, considera que la mitad es para el inicio y la otra para el final
        ventana = np.array([ventana, ventana])
    else:
        ventana = np.array(ventana)

    # Transforma a escala de fotogramas
    ventana = ventana * daDatos.freq

    if tipo_test == "DJ2P":
        # Busca primer aterrizaje provisional
        def detect_aterr1(data):
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            try:
                # Busca primer aterrizaje
                aterr = detect_onset(
                    -data,
                    threshold=-umbral,
                    n_above=int(0.05 * daDatos.freq),
                    show=show,
                )[0, 0]

            except:
                aterr = 0  # por si no encuentra el criterio
            return float(aterr)

        aterr1 = xr.apply_ufunc(
            detect_aterr1,
            daDatos,
            input_core_dims=[["time"]],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

        # Busca segundo aterrizaje provisional
        aterr2 = detecta_despegue_aterrizaje(
            daDatos, tipo_test=tipo_test, umbral=umbral
        ).sel(event="aterrizaje")

        daEventos.loc[dict(event="iniAnalisis")] = xr.where(
            (aterr1 - ventana[0]) >= 0, aterr1 - ventana[0], 0
        )  # .astype(int)
        daEventos.loc[dict(event="finAnalisis")] = xr.where(
            (aterr2 + ventana[1]) < len(daDatos.time),
            aterr2 + ventana[1] - 1,
            len(daDatos.time) - 1,
        )  # .astype(int)

    else:
        # Busca despegue y aterrizaje provisionales
        d_a = detecta_despegue_aterrizaje(daDatos, tipo_test=tipo_test, umbral=umbral)
        desp = d_a.sel(event="despegue").where(d_a.sel(event="despegue") > 0, np.nan)
        aterr = d_a.sel(event="aterrizaje").where(
            d_a.sel(event="aterrizaje") < len(daDatos.time) - 1, np.nan
        )

        daEventos.loc[dict(event="iniAnalisis")] = xr.where(
            desp.notnull(),
            xr.where((desp - ventana[0]) >= 0, desp - ventana[0], 0),
            np.nan,
        )
        daEventos.loc[dict(event="finAnalisis")] = xr.where(
            desp.notnull(),
            xr.where(
                (aterr + ventana[1] - 1) < len(daDatos.time),
                aterr + ventana[1] - 1,
                len(daDatos.time) - 1,
            ),
            np.nan,
        )

        """
        centro_vuelo = detecta_despegue_aterrizaje(daDatos, tipo_test=tipo_test, umbral=umbral).mean('evento')
            
        daEventos.loc[dict(event='iniAnalisis')] = xr.where((centro_vuelo - ventana[0]) >= 0, centro_vuelo - ventana[0], 0)#.astype(int)
        daEventos.loc[dict(event='finAnalisis')] = xr.where((centro_vuelo + ventana[1]-1) < len(daDatos.time), centro_vuelo + ventana[1]-1, len(daDatos.time)-1)#.astype(int)
        """
    return daEventos


def calcula_peso(daDatos, ventana_peso=None, show=False) -> xr.DataArray:
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    # Con ventana de peso única para todos
    # daPeso = daDatos.isel(time=slice(ventana[0], ventana[1])).mean(dim='time')

    # Con ventanas personalizadas
    def peso_indiv_xSD(data, vent0, vent1):
        try:
            vent0 = int(vent0)
            vent1 = int(vent1)
            peso = []
            peso.append(data[vent0:vent1].mean())
            peso.append(data[vent0:vent1].std())
        except:
            return np.asarray([np.nan, np.nan])
        # plt.plot(data[vent0:vent1])
        # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)
        return np.asarray(peso)

    """
    data = daDatos[0].data
    vent0 = ventana_peso.sel(event='iniPeso')[0].data
    vent1 = ventana_peso.sel(event='finPeso')[0].data
    """
    daPeso = xr.apply_ufunc(
        peso_indiv_xSD,
        daDatos,
        ventana_peso.isel(event=0),
        ventana_peso.isel(
            event=1
        ),  # .sel(event='iniPeso'), ventana_peso.sel(event='finPeso'),
        input_core_dims=[["time"], [], []],
        output_core_dims=[["stat"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    ).assign_coords(stat=["media", "sd"])

    if show:

        def dibuja_peso(x, y, **kwargs):  # de momento no funciona
            print(x)  # kwargs['data'])
            # plt.plot()

        g = daDatos.plot.line(col="ID", col_wrap=3, hue="repe", alpha=0.8, sharey=False)
        # g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
        # g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')
        col = ["C0", "C1", "C2"]
        for h, ax in enumerate(g.axs):  # extrae cada fila
            for i in range(len(ax)):  # extrae cada axis (gráfica)
                if (
                    g.name_dicts[h, i] == None
                ):  # en los cuadros finales que sobran de la cuadrícula se sale
                    break
                try:
                    idn = g.data.loc[g.name_dicts[h, i]].ID
                    # print('peso=', daPeso.sel(ID=idn).data)#idn)
                    # Rango medida peso
                    # ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.freq)], g.data.time[int(ventana[1]*self.datos.freq)], alpha=0.2, color='C1')
                    for j in daDatos.repe:
                        ax[i].axvspan(
                            ventana_peso.sel(ID=idn, repe=j).isel(event=0)
                            / daDatos.freq,
                            ventana_peso.sel(ID=idn, repe=j).isel(event=1)
                            / daDatos.freq,
                            alpha=0.2,
                            color=col[j.data - 1],
                        )
                        ax[i].axhline(
                            daPeso.sel(ID=idn, repe=j, stat="media").data,
                            color=col[j.data - 1],
                            lw=1,
                            ls="--",
                            alpha=0.6,
                        )
                    # Líneas peso
                    # ax[i].hlines(daPeso.sel(ID=idn, stat='media').data, xmin=daDatos.time[0], xmax=daDatos.time[-1], colors=col, lw=1, ls='--', alpha=0.6)
                except:
                    print("Error al pintar peso en", g.name_dicts[h, i], h, i)
    return daPeso


def afina_final(
    daDatos,
    daEventos=None,
    daPeso=None,
    ventana=0.2,
    margen=0.005,
    tipo_calculo="opt",
    show=False,
) -> xr.DataArray:
    """
    tipo_calculo puede ser 'opt', 'iter', 'iter_gradiente' o 'iter_final'
    """

    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    # TODO: USAR FUNCIÓN GENERAL DE INTEGRAR
    def _integra(data, t, peso):
        dat = np.full(len(data), np.nan)
        try:
            dat = integrate.cumtrapz(data - peso, t, initial=0)
        except:
            pass  # dat = np.full(len(data), np.nan)
        return dat

    if isinstance(ventana, float):
        ventana = ventana * daDatos.freq

    # Recorta a una ventana desde el final
    daEventos.loc[dict(event="iniAnalisis")] = (
        daEventos.sel(event="finAnalisis") - ventana
    )
    daDatosCort = recorta_ventana_analisis(daDatos, daEventos)

    if show:
        daDatosCort.isel(ID=slice(None)).plot.line(x="time", col="ID", col_wrap=4)

    daDatosCort.std(dim="time")

    def _peso_iter_gradiente(data, t, peso, ini, fin, margen, ID):  # , rep):
        """
        Va ajustando el peso teniendo en cuenta la diferncia con la iteración
        anterior, hasta que la diferencia es menor que un nº mínimo.
        """
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            pes = peso - 100
            delta_peso = 1
            tend_pes = []
            v = np.full(len(data), 20.0)
            for i in range(1000):
                v0 = _integra(data, t, pes) / (pes / 9.8)
                v1 = _integra(data, t, pes + delta_peso) / (pes + delta_peso / 9.8)
                pes += v0[-1] - v1[-1]  # *1
                tend_pes.append(pes)
                if i > 2 and pes - tend_pes[-2] < 0.00001:
                    break
            # plt.plot(tend_pes)

        except:
            print("No se encontró")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v0, lw=0.2, label="ajustado")
            v = _integra(data, t, peso) / (peso / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.title("Cálculo iterativo descenso gradiente")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, peso - pes])

    def _peso_iter_final(data, t, peso, ini, fin, margen, ID):  # , rep):
        # Ajuste para saltos con preactivación. Devuelve el peso ajustado y la diferencia entre el peso anterior y el ajustado
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            # primera pasada más gruesa
            itera = 0
            pes = 300
            v = np.arange(len(data))  # np.full(len(data), 20.0)
            # while not -3 < v[-1] < 3 and pes < peso+100:
            while (
                v[int(-0.2 * daDatos.freq) : int(-0.1 * daDatos.freq)].mean()
                - v[int(-0.5 * daDatos.freq) : int(-0.4 * daDatos.freq)].mean()
                > 0.05
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                pes += 5.0
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

            # Segunda pasada más fina
            itera = 0
            pes = pes - 4
            v = _integra(data, t, pes) / (pes / 9.8)
            while (
                v[int(-0.1 * daDatos.freq) : int(-0.05 * daDatos.freq)].mean()
                - v[int(-0.3 * daDatos.freq) : int(-0.25 * daDatos.freq)].mean()
                > margen
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                # plt.plot(len(v)-int(0.2*daDatos.freq), v[int(-0.2*daDatos.freq)], 'o')
                # plt.plot(len(v)-int(0.1*daDatos.freq), v[int(-0.1*daDatos.freq)], 'o')
                # plt.plot(len(v)-int(.6*daDatos.freq), v[int(-.6*daDatos.freq)], 'o')
                # plt.plot(len(v)-int(.5*daDatos.freq), v[int(-.5*daDatos.freq)], 'o')

                pes += 0.01
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

        except:
            print("No se encontró")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v, lw=0.2, label="ajustado")
            v = _integra(data, t, peso) / (peso / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.title("Cálculo iterativo")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, peso - pes])

    """
    #Con repe
    data = daDatos[1,0].data
    t = daDatos.time.data
    peso = daPeso.sel(stat='media')[1,0].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEventos.sel(event=evIni)[1,0].data
    fin = daEventos.sel(event=evFin)[1,0].data

    #Sin repe
    data = daDatos[1].data
    t = daDatos.time.data
    peso = daPeso.sel(stat='media')[1].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEventos.sel(event=evIni)[1].data
    fin = daEventos.sel(event=evFin)[1].data
    """
    if tipo_calculo == "iter_gradiente":
        f_calculo = _peso_iter_gradiente
        evIni = "iniMov"
        evFin = "finMov"
    elif tipo_calculo == "iter_final":
        f_calculo = _peso_iter_final
        evIni = "despegue"
        evFin = "finAnalisis"
    else:
        raise (f"Método de cálculo {tipo_calculo} no implementado")

    daPesoReturn = xr.apply_ufunc(
        f_calculo,
        daDatos,
        daDatos.time,
        daPeso.sel(stat="media"),
        daEventos.sel(event=evIni),
        daEventos.sel(event=evFin),
        margen,
        daDatos.ID,  # daDatos.repe,
        input_core_dims=[["time"], ["time"], [], [], [], [], []],  # , []],
        output_core_dims=[["stat"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    ).assign_coords(stat=["media", "resid"])

    if daPeso is not None:
        daPesoReturn = xr.concat([daPesoReturn, daPeso.sel(stat="sd")], dim="stat")
        daPesoReturn.loc[dict(stat="sd")] = daPeso.sel(stat="sd")

    return daPesoReturn


def afina_peso(
    daDatos, daEventos=None, daPeso=None, margen=0.005, tipo_calculo="opt", show=False
) -> xr.DataArray:
    """
    tipo_calculo puede ser 'opt', 'iter', 'iter_gradiente', 'iter_final', 'peso_media_salto'
    """

    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    def _integra(data, t, peso):
        dat = np.full(len(data), np.nan)
        try:
            dat = integrate.cumtrapz(data - peso, t, initial=0)
        except:
            pass  # dat = np.full(len(data), np.nan)
        return dat

    def _optimiza_peso(data, t, peso, ini, fin, margen, ID):  # , rep):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])
        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin]

            vy = []
            p = []
            for pes in range(
                200, 2000, 50
            ):  # calcula velocidad entre pesos extremos cada x datos
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v)
                vy.append(v[-1])  # v[-int(0.1*daDatos.freq):].mean())
                p.append(pes)

            # Con polinomio
            coefs, (resid, _, _, _) = np.polynomial.polynomial.Polynomial.fit(
                vy, p, deg=8, full=True
            )
            coefs = coefs.convert().coef  # convierte a escala real
            f = np.polynomial.Polynomial(coefs)
            pes = f(0)  # peso cuando la velocidad al final es cero
            """
            vy2 = np.arange(-10, 20)
            px = f(vy2) 
            plt.plot(p,vy,'o-')
            plt.plot(px,vy2)
            plt.show()
            """

            """
            #Con otras funciones
            from scipy.optimize import curve_fit
            def f(x,a,b,c):
                return a*np.exp(-b*x)+c
            popt, pcov = curve_fit(f, vy, p)
            peso = f(0, popt[0], popt[1], popt[2]) #peso cuando la velocidad al final es cero
            
            # def f(x, qi, exp, di):
            #     return qi*(1+exp*di*x)**(-1/exp)
            # popt, pcov = curve_fit(f, vy, p,  factor=1)
                                   
            
            vy2 = np.arange(-10, 10)
            px = f(vy2, popt[0], popt[1], popt[2]) 
            plt.plot(vy,p,'--')
            plt.plot(px,vy2)
            plt.show()
            """

        except:
            print("No se encontró")
            return np.asarray([np.nan, np.nan])

        if show:
            v = _integra(data, t, pes) / (pes / 9.8)
            plt.plot(v, lw=1)
            plt.text(
                0.02,
                0.9,
                f"delta con peso media={peso-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo optimizado polinomio ({ID})")
            plt.show()

        return np.asarray([pes, resid[0]])

    def _peso_iter(data, t, peso, ini, fin, margen, ID):  # , rep):
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            # primera pasada más gruesa
            itera = 0
            pes = 300  # peso-100
            v = np.full(len(data), 20.0)
            # while not -3 < v[-1] < 3 and pes < peso+100:
            while v[-1] > 0.5 and pes < 1300:
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v)
                pes += 5
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

            # Segunda pasada más fina
            itera = 0
            # while not -margen < v[-1] < margen and pes < peso+5:
            while v[-1] > margen and pes < peso + 50:
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v)
                pes += 0.05
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])
        except:
            print("No se encontró")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v, lw=1)
            plt.text(
                0.02,
                0.9,
                f"delta con peso media={peso-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo iterativo ({ID})")
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, peso - pes])

    def _peso_iter_gradiente(data, t, peso, ini, fin, margen, ID):  # , rep):
        """
        Va ajustando el peso teniendo en cuenta la diferncia con la iteración
        anterior, hasta que la diferencia es menor que un nº mínimo.
        """
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            pes = peso - 100
            delta_peso = 1
            tend_pes = []
            v = np.full(len(data), 20.0)
            for i in range(1000):
                v0 = _integra(data, t, pes) / (pes / 9.8)
                v1 = _integra(data, t, pes + delta_peso) / (pes + delta_peso / 9.8)
                pes += v0[-1] - v1[-1]  # *1
                tend_pes.append(pes)
                if i > 2 and pes - tend_pes[-2] < 0.00001:
                    break
            # plt.plot(tend_pes)

        except:
            print("No se encontró")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v0, lw=0.2, label="ajustado")
            v = _integra(data, t, peso) / (peso / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.text(
                0.02,
                0.9,
                f"delta con peso media={peso-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo iterativo descenso gradiente ({ID})")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, peso - pes])

    def _peso_iter_final(data, t, peso, ini, fin, margen, ID):  # , rep):
        # Ajuste para saltos con preactivación. Devuelve el peso ajustado y la diferencia entre el peso anterior y el ajustado
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            # primera pasada más gruesa
            itera = 0
            pes = 300
            v = np.arange(len(data))  # np.full(len(data), 20.0)
            # while not -3 < v[-1] < 3 and pes < peso+100:
            while (
                v[int(-0.2 * daDatos.freq) : int(-0.1 * daDatos.freq)].mean()
                - v[int(-0.5 * daDatos.freq) : int(-0.4 * daDatos.freq)].mean()
                > 0.05
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                pes += 5.0
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

            # Segunda pasada más fina
            itera = 0
            pes = pes - 4
            v = _integra(data, t, pes) / (pes / 9.8)
            while (
                v[int(-0.1 * daDatos.freq) : int(-0.05 * daDatos.freq)].mean()
                - v[int(-0.3 * daDatos.freq) : int(-0.25 * daDatos.freq)].mean()
                > margen
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                # plt.plot(len(v)-int(0.2*daDatos.freq), v[int(-0.2*daDatos.freq)], 'o')
                # plt.plot(len(v)-int(0.1*daDatos.freq), v[int(-0.1*daDatos.freq)], 'o')
                # plt.plot(len(v)-int(.6*daDatos.freq), v[int(-.6*daDatos.freq)], 'o')
                # plt.plot(len(v)-int(.5*daDatos.freq), v[int(-.5*daDatos.freq)], 'o')

                pes += 0.01
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

        except:
            print("No se encontró")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v, lw=0.2, label="ajustado")
            v = _integra(data, t, peso) / (peso / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.text(
                0.02,
                0.9,
                f"delta con peso media={peso-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo iterativo ({ID})")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, peso - pes])

    def _peso_media_salto(data, t, peso, ini, fin, margen, ID):  # , rep):
        # Ajuste para saltos con preactivación. Devuelve el peso ajustado y la diferencia entre el peso anterior y el ajustado
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])
        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            pes = data[ini:fin].mean()

        except:
            print("No se pudo calcular el peso medio")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(data[ini:fin], lw=0.5, label="Fuerza")
            plt.axhline(pes, color="b", ls="-", lw=1, label="Peso media total")
            plt.axhline(
                peso, color="r", alpha=0.7, ls="--", lw=0.5, label="Peso media tramo"
            )
            plt.text(
                0.02,
                0.9,
                f"delta con peso media={peso-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo media salto ({ID})")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, peso - pes])

    """
    #Con repe
    data = daDatos[1,0].data
    t = daDatos.time.data
    peso = daPeso.sel(stat='media')[1,0].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEventos.sel(event=evIni)[1,0].data
    fin = daEventos.sel(event=evFin)[1,0].data

    #Sin repe
    data = daDatos[1].data
    t = daDatos.time.data
    peso = daPeso.sel(stat='media')[1].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEventos.sel(event=evIni)[1].data
    fin = daEventos.sel(event=evFin)[1].data
    """
    if tipo_calculo == "opt":
        f_calculo = _optimiza_peso
        evIni = "iniMov"
        evFin = "finMov"
    elif tipo_calculo == "iter":
        f_calculo = _peso_iter
        evIni = "iniMov"
        evFin = "finMov"
    elif tipo_calculo == "iter_gradiente":
        f_calculo = _peso_iter_gradiente
        evIni = "iniMov"
        evFin = "finMov"
    elif tipo_calculo == "iter_final":
        f_calculo = _peso_iter_final
        evIni = "despegue"
        evFin = "finAnalisis"
    elif tipo_calculo == "peso_media_salto":
        f_calculo = _peso_media_salto
        evIni = "iniAnalisis"
        evFin = "finAnalisis"
    else:
        raise Exception(f"Método de cálculo {tipo_calculo} no implementado")

    daPesoReturn = xr.apply_ufunc(
        f_calculo,
        daDatos,
        daDatos.time,
        daPeso.sel(stat="media"),
        daEventos.sel(event=evIni),
        daEventos.sel(event=evFin),
        margen,
        daDatos.ID,  # daDatos.repe,
        input_core_dims=[["time"], ["time"], [], [], [], [], []],  # , []],
        output_core_dims=[["stat"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    ).assign_coords(stat=["media", "resid"])

    if daPeso is not None:
        daPesoReturn = xr.concat([daPesoReturn, daPeso.sel(stat="sd")], dim="stat")
        daPesoReturn.loc[dict(stat="sd")] = daPeso.sel(stat="sd")

    return daPesoReturn


def detecta_despegue_aterrizaje(
    daDatos, tipo_test, umbral=10.0, show=False
) -> xr.DataArray:
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    if tipo_test == "DJ2P":

        def _detect_onset_aux(data, coords, **args_func_cortes):
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.array([np.nan, np.nan])
            # plt.plot(data)
            # plt.show()
            # print(ID, repe)
            ind = detect_onset(-data, **args_func_cortes)

            if len(ind) < 2:
                # Segunda oportunidad buscando a partir del mínimo
                args_func_cortes["threshold"] -= data.min()
                ind = detect_onset(-data, **args_func_cortes)
                print("Corregido umbral", coords)
                if len(ind) < 1:
                    print(
                        "No se han encontrado dos despegues/aterrizajes en archivo",
                        coords,
                    )
                    return np.array([np.nan, np.nan])
            # if tipo_test == 'CMJ':
            #     ind=ind[0] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
            # elif tipo_test in ['DJ', 'DJ2P']:
            #     ind=ind[1] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral

            # Chequea si ha detectado más de un vuelo
            if len(ind) >= 2:
                ind = ind[-1]  # por defecto se queda con el último

                # TODO: mejorar la comprobación cuando detecta más de un vuelo
                # if ind[-1,-1] < int(len(data) * 0.8) or :
                #     ind=ind[-1] #Independientemente del tipo de salto que sea, se queda con el último que encuentre
                # else:
                #     ind=ind[-2]

            return ind.astype(float)  # [1]

        """
        data = daDatos.sel(ID='S07_DJ_30', repe=1).data
        data = daDatos[0].data
        args_func_cortes = dict(threshold=-umbral, n_above=50, show=True)
        """

        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daDatos.ID,
            input_core_dims=[["time"], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(threshold=-umbral, n_above=int(0.1 * daDatos.freq), show=show),
        ).assign_coords(event=["despegue", "aterrizaje"])

    else:

        def _detect_onset_aux(data, coords, **args_func_cortes):
            if np.count_nonzero(~np.isnan(data)) == 0:  # or data.sum()==0.0:
                return np.array([np.nan, np.nan])
            # plt.plot(data)
            # plt.show()
            # print(ID, repe)
            ind = detect_onset(-data, **args_func_cortes)
            if len(ind) < 1:
                # Segunda oportunidad buscando a partir del mínimo
                args_func_cortes["threshold"] -= data.min()
                ind = detect_onset(-data, **args_func_cortes)
                print("Corregido umbral", coords)
                if len(ind) < 1:
                    print("No se ha encontrado despegue/aterrizaje en archivo", coords)
                    return np.array([np.nan, np.nan])
            # if tipo_test == 'CMJ':
            #     ind=ind[0] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
            # elif tipo_test in ['DJ', 'DJ2P']:
            #     ind=ind[1] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral

            # Chequea si ha detectado más de un vuelo
            if len(ind) > 1:
                ind = (
                    ind[1] if tipo_test == "DJ" else ind[0]
                )  # por defecto se queda con el primero o el segundo
                # TODO: mejorar la comprobación cuando detecta más de un vuelo
                # if ind[-1,-1] < int(len(data) * 0.8) or :
                #     ind=ind[-1] #Independientemente del tipo de salto que sea, se queda con el último que encuentre
                # else:
                #     ind=ind[-2]

            return ind.astype(float)

        """
        data = daDatos.sel(ID='PRE_07-RubenToledoPozuelo_SJ_0', repe=1).data
        data = daDatos[0,0].data
        args_func_cortes = dict(threshold=-umbral, n_above=50, show=True)
        """

        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daDatos.ID,
            input_core_dims=[["time"], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(threshold=-umbral, n_above=int(0.01 * daDatos.freq), show=show),
        ).assign_coords(event=["despegue", "aterrizaje"])
    # Comprobaciones
    # daDatos.sel(axis='z').isel(time=daCorte.sel(event='despegue')-1) #despegue cuando ya ha pasado por debajo del umbral
    # daDatos.sel(axis='z').isel(time=daCorte.sel(event='aterrizaje')-1) #aterrizaje cuando ya ha pasado por debajo del umbral
    return daCorte


def detecta_despegue_aterrizaje_cusum(
    daDatos, tipo_test, umbral=10.0, show=False
) -> xr.DataArray:
    # Prueba para detectar varios eventos a la vez, pero parece muy irregular
    def _detect_onset_aux(data, **args_func_cortes):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.array([np.nan, np.nan])
        # plt.plot(data)
        # plt.show()
        # print(ID, repe)
        from detecta import detect_cusum

        detect_cusum(data, threshold=500, drift=1, ending=True, show=True)
        ind = []
        # ind = detect_onset(-data, **args_func_cortes)
        if len(ind) < 1:
            print("No se ha encontrado despegue/aterrizaje en archivo")
            return np.array([np.nan, np.nan])
        # if tipo_test == 'CMJ':
        #     ind=ind[0] #coge el primer bloque que encuentra
        #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
        # elif tipo_test in ['DJ', 'DJ2P']:
        #     ind=ind[1] #coge el primer bloque que encuentra
        #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral

        # Independientemente del tipo de salto que sea, se queda con el último que en cuentre
        ind = ind[-1]

        return ind.astype("float")  # [1]

    """
    data = daDatos[0,-1].data
    args_func_cortes = dict(threshold=-umbral, n_above=50, show=True)
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    daCorte = xr.apply_ufunc(
        _detect_onset_aux,
        daDatos,
        input_core_dims=[["time"]],
        output_core_dims=[["event"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
        kwargs=dict(threshold=-umbral, n_above=int(0.2 * daDatos.freq), show=show),
    ).assign_coords(event=["despegue", "aterrizaje"])
    # Comprobaciones
    # daDatos.sel(axis='z').isel(time=daCorte.sel(event='despegue')-1) #despegue cuando ya ha pasado por debajo del umbral
    # daDatos.sel(axis='z').isel(time=daCorte.sel(event='aterrizaje')-1) #aterrizaje cuando ya ha pasado por debajo del umbral
    return daCorte


def detecta_ini_mov(
    daDatos,
    tipo_test="CMJ",
    daPeso=None,
    daEventos=None,
    SDx=5,
    umbral=10.0,
    show=False,
) -> xr.DataArray:
    # Aquí SDx es el % del peso
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    # corrector_freq = daDatos.freq_ref if 'freq_ref' in daDatos.attrs else 200.0

    # Función común para buscar umbral por encima o por debajo
    def _detect_iniMov_peso_mayor_menor(
        data, peso, umbral_peso, iinianalisis, idespegue, win_up, win_down, ID
    ):
        # Intenta detectar qué es antes: descenso por debajo del peso o ascenso por encima (dando saltito)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan
        # ini_abajo = 0
        # ini_arriba = 0
        # try:

        # print(ID)
        iinianalisis = int(iinianalisis)
        idespegue = int(idespegue)
        dat = data[iinianalisis:idespegue]
        # plt.plot(data[iinianalisis:idespegue])
        # Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
        ini1 = detect_onset(
            -dat,
            threshold=-(peso - umbral_peso),
            n_above=int(win_down * daDatos.freq),
            threshold2=-(peso - umbral_peso) * 1.2,
            n_above2=int(win_down * daDatos.freq) * 0.5,
            show=show,
        )
        if ini1.size != 0:
            # Pasada hacia atrás buscando ajuste fino que supera el peso
            ini2 = detect_onset(
                dat[ini1[0, 0] : 0 : -1], threshold=peso, n_above=1, show=show
            )
            if ini2.size == 0:
                ini2 = np.array([[0, 0]])  # si no encuentra, suma cero
            elif ini2[0, 0] / daDatos.freq > 0.2:
                ini2[0, 0] = int(
                    0.2 * daDatos.freq
                )  # si la detección atrás se va muy lejos, retrocede un valor arbitrario de segundos
            ini_abajo = (
                iinianalisis + ini1[0, 0] - ini2[0, 0] + 1
            )  # +1 para coger el que ya ha pasado por debajo del peso
        else:
            ini_abajo = len(data)  # por si no encuentra el criterio

        # except:
        #     ini_abajo = len(data) #por si no encuentra el criterio

        # try:
        # Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
        ini1 = detect_onset(
            dat,
            threshold=(peso + umbral),
            n_above=int(win_up * daDatos.freq),
            show=show,
        )
        if ini1.size != 0:
            # Pasada hacia atrás buscando ajuste fino que supera el peso
            ini2 = detect_onset(
                -dat[ini1[0, 0] : 0 : -1], threshold=-peso, n_above=1, show=show
            )
            if ini2.size == 0:
                ini2 = np.array([[0, 0]])  # si no encuentra, suma cero
            elif ini2[0, 0] / daDatos.freq > 0.2:
                ini2[0, 0] = int(
                    0.2 * daDatos.freq
                )  # si la detección atrás se va muy lejos, retrocede un valor arbitrario de segundos
            ini_arriba = (
                iinianalisis + ini1[0, 0] - ini2[0, 0] + 1
            )  # +1 para coger el que ya ha pasado por encima del peso
        else:
            ini_arriba = len(data)  # por si no encuentra el criterio
        # except:
        #     ini_arriba = len(data) #por si no encuentra el criterio

        if ini_arriba == len(data) and ini_abajo == len(data):
            idx = 0
        else:
            idx = np.min([ini_arriba, ini_abajo])

        return float(idx)

    if tipo_test == "DJ":

        def _detect_onset_aux(data, **args_func_cortes):
            # plt.plot(data)
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            ini = detect_onset(-data, **args_func_cortes)[0]
            return float(
                ini[1] + 1
            )  # +1 para que se quede con el que ya ha pasado el umbral

        # data= daDatos[0,0,2].data
        # args_func_cortes = dict(threshold=-10.0, n_above=50, show=False)
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            input_core_dims=[["time"]],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(threshold=-umbral, n_above=int(0.2 * daDatos.freq), show=show),
        )
        # daDatos.sel(axis='z').isel(time=daCorte-1)

    elif tipo_test == "CMJ":
        """
        def detect_iniMov_peso_pcto(data, peso, umbral_peso, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-umbral_peso), n_above=int(0.2*daDatos.freq), show=show)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[-1,0]:0:-1], threshold=(peso-umbral_peso*0.5), n_above=int(0.02*daDatos.freq), show=show)

                ini = ini1[-1,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return float(ini)
        """
        """
        data = daDatos[3,2].data
        plt.plot(data)
        peso = daPeso[3,2].sel(stat='media').data
        sdpeso = (daPeso[3,2].sel(stat='sd')*SDx).data
        umbral_peso = (daPeso[3,2].sel(stat='sd')*SDx).data
        iinianalisis = daEventos[3,2].sel(event='iniAnalisis').data
        idespegue = daEventos[3,2].sel(event='despegue').data
        ID = daEventos[3,2].ID.data
        """
        func_detect = _detect_iniMov_peso_mayor_menor  # detect_iniMov_peso_pcto
        daCorte = xr.apply_ufunc(
            func_detect,
            daDatos,
            daPeso.sel(ID=daDatos.ID, stat="media"),
            daPeso.sel(ID=daDatos.ID, stat="sd") * SDx,
            daEventos.sel(ID=daDatos.ID).sel(event="iniAnalisis"),
            daEventos.sel(ID=daDatos.ID).sel(event="despegue"),
            0.1,
            0.05,
            daDatos.ID,
            input_core_dims=[["time"], [], [], [], [], [], [], []],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

    elif tipo_test == "SJ":
        """
        def detect_iniMov_peso_pcto(data, peso, umbral_peso, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan

            try:
                #Pasada inicial para ver cuándo supera el umbral peso+XSD
                ini1 = detect_onset(data[:int(idespegue)], threshold=(peso+umbral_peso), n_above=int(0.2*daDatos.freq), show=show)

                #Pasada hacia atrás buscando ajuste fino que quede por debajo del peso
                ini2 = detect_onset(-data[ini1[-1,0]:0:-1], threshold=-(peso+umbral_peso*0.5), n_above=int(0.01*daDatos.freq), show=show)

                ini = ini1[-1,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por encima del peso
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return float(ini)
        """
        """
        data = daDatos[6,2].data
        plt.plot(data)        
        peso = daPeso[6,2].sel(stat='media').data
        sdpeso = (daPeso[6,2].sel(stat='sd')*SDx).data
        umbral_peso =  (daPeso[6,2].sel(stat='sd')*SDx).data
        iinianalisis = daEventos[6,2].sel(event='iniAnalisis').data
        idespegue = daEventos[6,2].sel(event='despegue').data
        win_down = 0.05
        win_up = 0.1
        """
        func_detect = _detect_iniMov_peso_mayor_menor  # detect_iniMov_peso_pcto
        daCorte = xr.apply_ufunc(
            func_detect,
            daDatos,
            daPeso.sel(ID=daDatos.ID, stat="media"),
            daPeso.sel(ID=daDatos.ID, stat="media") * SDx / 100,
            daEventos.sel(ID=daDatos.ID).sel(event="iniAnalisis"),
            daEventos.sel(ID=daDatos.ID).sel(event="despegue"),
            0.05,
            0.1,  # ventanas down y up, en segundos
            daDatos.ID,
            input_core_dims=[["time"], [], [], [], [], [], [], []],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

    elif tipo_test == "SJPreac":
        # Como no se observa un patrón estable, coge un tiempo prefijado antes del despegue, basado en la media de fase concéntrica en SJ (#((daEventosForces.sel(event='finImpPos') - daEventosForces.sel(event='iniImpPos'))/daSJ.freq).mean())
        daCorte = daEventos.sel(event="despegue") - 0.25 * daDatos.freq

        """
        from detecta import detect_cusum        
        def detect_iniMov_peso_pcto(data, peso, umbral_peso, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            try:
                _, ini, _, _ = detect_cusum(data[:int(idespegue)], threshold=80, drift=1, ending=True, show=show)
                ini = ini[0]
                
            except:
                ini = 0 #por si no encuentra el criterio
                
            return float(ini)
        
        """
        """
        data = daDatos[0,0].data
        peso = daPeso[0,0].sel(stat='media').data
        pcto = 10
        sdpeso = (daPeso[0,0].sel(stat='sd')*SDx).data
        umbral = (daPeso[0].sel(stat='sd')*SDx).data
        idespegue = daEventos[0,0].sel(event='despegue').data
        """
        """
        daCorte = xr.apply_ufunc(detect_iniMov_peso_pcto, daDatos, daPeso.sel(ID=daDatos.ID, stat='media'), daPeso.sel(ID=daDatos.ID, stat='media')*SDx/100, daEventos.sel(ID=daDatos.ID).sel(event='despegue'),
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """

    elif tipo_test == "DJ2P":

        def _detect_iniMov_peso_pcto(data, peso, pcto_peso, idespegue):
            # Rastrea una ventana del peso (entre peso*(100-SDx)/100 a peso*(100+SDx)/100)
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            try:
                idespegue = int(idespegue)
                # Busca primer aterrizaje
                aterr = detect_onset(
                    -data[:idespegue],
                    threshold=-umbral,
                    n_above=int(0.05 * daDatos.freq),
                    show=show,
                )[0, 0]
                # Escanea desde el peso hacia abajo en porcentajes
                for pct in range(0, -pcto_peso, -1):
                    ini = detect_onset(
                        data[aterr::-1],
                        threshold=peso * (100 - pct) / 100,
                        n_above=int(0.05 * daDatos.freq),
                        show=show,
                    )  #
                    # print(idx[0,1] - idx[0,0])
                    if len(ini) > 1:
                        if ini[0, 1] - ini[0, 0] < 0.3 * daDatos.freq:
                            break
                    else:
                        continue

                ini = aterr - ini[0, 1] if len(ini) > 1 else aterr - ini[0, 0]

            except:
                ini = 0  # por si no encuentra el criterio
            return float(ini)

        """
        data = daDatos[0,0].data        
        peso = daPeso[0,0].sel(stat='media').data
        pcto = 10
        sdpeso = (daPeso[0,0].sel(stat='sd')*SDx).data
        umbral_peso = daPeso[0,0].sel(stat='media').data*SDx/100
        idespegue = daEventos[0,0].sel(event='despegue').data
        
        data = daDatos.sel(ID='S02_DJ_30', repe=1).data
        peso = daPeso.sel(ID='S02_DJ_30', repe=1, stat='media').data
        umbral_peso = daPeso.sel(ID='S02_DJ_30', repe=1).sel(stat='media').data*SDx/100
        idespegue = daEventos.sel(ID='S02_DJ_30', repe=1).sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_iniMov_peso_pcto,
            daDatos,
            daPeso.sel(stat="media"),
            SDx,
            daEventos.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

        """
        def detect_iniMov_peso_XSD(data, peso, sdpeso, idespegue):
            #Parte del despegue hacia atrás buscando cuándo supera el umbral del peso - sd*SDx
            #ini = detect_onset(-data[:int(idespegue):-1], threshold=-umbral, n_above=50, show=True)[0]
            #ini = detect_onset(data[int(idespegue)::-1], threshold=umbral, n_above=5, show=True)
            #ini = idespegue - ini[1,0] + 1 #+1 para coger el que ya ha superado el umbral
            
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-sdpeso), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:ini1[0,0]-100:-1], threshold=peso, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha superado el umbral
                
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
        daCorte = xr.apply_ufunc(detect_iniMov_peso_XSD, daDatos.sel(axis='z'), daPeso.sel(stat='media'), daPeso.sel(stat='sd')*SDx, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """

        # Comprobaciones
        # daDatos.sel(axis='z').isel(time=daCorte-1) #

    # Si hay datos, ajusta el límite al inicio análisis
    daCorte = xr.where(
        daCorte.notnull(),
        daCorte.where(
            daCorte > daEventos.sel(event="iniAnalisis"),
            daEventos.sel(event="iniAnalisis"),
        ),
        np.nan,
    )

    return daCorte


def detecta_fin_mov(
    daDatos,
    tipo_test,
    daPeso=None,
    daEventos=None,
    tipo_calculo="fuerza",
    SDx=2,
    show=False,
) -> xr.DataArray:
    """
    tipo_calculo puede ser 'velocidad', 'fuerza' o 'ventana_plana'
    """

    # #Con umbral velocidad cero. No funciona bien cuando la velocidad se queda al final por encima de cero
    # def detect_onset_aux(data, umbral, iaterrizaje, ID):
    #     # print(ID)
    #     # plt.plot(data)
    #     if np.count_nonzero(~np.isnan(data))==0:
    #         return np.nan

    #     try:
    #         fin = detect_onset(data[int(iaterrizaje):], threshold=0.0, n_above=int(0.1*daDatos.freq), show=show)
    #         fin = iaterrizaje + fin[0,1] + 1 #+1 para coger el que ya ha superado el umbral
    #                            #fin[0,0] o fin[1,1] ?????
    #     except:
    #         fin = len(data[~np.isnan(data)]) #por si no encuentra el criterio
    #     return float(fin)

    # TODO: IMPLEMENTAR 'ventana_plana', QUE BUSQUE CUÁNDO SE CUMPLE QUE UNA VENTANA TENGA SD MENOR QUE UMBRAL

    # Calcula la diferencia en velocidad dato a dato y detecta cuándo está por encima de umbral pequeño
    def _detect_onset_dif_v(data, umbral, sd, iaterrizaje, finAn, ID):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        try:
            # plt.plot(data[int(fin):int(iaterrizaje):-1])
            dif = np.diff(data[int(finAn) : int(iaterrizaje) : -1])
            fin = detect_onset(
                dif, threshold=0.00005, n_above=int(0.05 * daDatos.freq), show=show
            )
            fin = finAn - fin[0, 0] + 1  # +1 para coger el que ya ha superado el umbral
        except:
            fin = finAn  # len(data[~np.isnan(data)]) #por si no encuentra el criterio

        return float(fin)

    def _detect_onset_fuerza(data, umbral, sd, iaterrizaje, finAn, ID):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        try:
            fin = detect_onset(
                -data[int(finAn) : int(iaterrizaje) : -1],
                threshold=-umbral + sd * SDx,
                n_above=int(0.1 * daDatos.freq),
                show=show,
            )
            fin = (
                finAn - fin[-1, 0] + 1
            )  # +1 para coger el que ya ha superado el umbral
            # fin[0,0] o fin[1,1] ?????
        except:
            fin = finAn  # len(data[~np.isnan(data)]) #por si no encuentra el criterio
        return float(fin)

    if tipo_calculo == "velocidad":
        f_calculo = _detect_onset_dif_v
        datos = calcula_variables(
            daDatos,
            daPeso=daPeso,
            daEventos=daEventos.sel(event=["iniMov", "finAnalisis"]),
        )["v"]
    elif tipo_calculo == "fuerza":
        f_calculo = _detect_onset_fuerza
        datos = daDatos
    else:
        raise Exception(f"Método de cálculo {tipo_calculo} no implementado")

    # datos.plot.line(x='time', col='ID', col_wrap=3)

    """
    #data = daDatos[0,0].data
    data = datos[0,0].data
    iaterrizaje = daEventos.sel(event='aterrizaje')[0,0].data
    finAn = daEventos.sel(event='finAnalisis')[0,0].data
    umbral = daPeso.sel(stat='media')[0,0].data
    sd = daPeso.sel(stat='sd')[0,0].data
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    daCorte = xr.apply_ufunc(
        f_calculo,
        datos,
        daPeso.sel(stat="media"),
        daPeso.sel(stat="sd"),
        daEventos.sel(event="aterrizaje"),
        daEventos.sel(event="finAnalisis"),
        daDatos.ID,
        input_core_dims=[["time"], [], [], [], [], []],
        # output_core_dims=[['peso']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(show=show)
    )

    # Ajusta el límite al inicio análisis
    daCorte = xr.where(
        daCorte.notnull(),
        daCorte.where(
            daCorte < daEventos.sel(event="finAnalisis"),
            daEventos.sel(event="finAnalisis"),
        ),
        np.nan,
    )

    return daCorte


# Encuentra fin cuando fuerza baja del peso y vuelve a subir
def detecta_fin_mov_convencional(
    daDatos, tipo_test, daPeso=None, daEventos=None, SDx=2
) -> xr.DataArray:
    def _detect_onset_aux(data, umbral, sd, iaterrizaje, ID):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan
        fin = detect_onset(
            -data[int(iaterrizaje) :],
            threshold=-umbral + sd * SDx,
            n_above=int(0.1 * daDatos.freq),
            show=False,
        )
        try:
            fin = (
                iaterrizaje + fin[0, 1] + 1
            )  # +1 para coger el que ya ha superado el umbral
            # fin[0,0] o fin[1,1] ?????
        except:
            fin = len(data[~np.isnan(data)])  # por si no encuentra el criterio
        return float(fin)

    """    
    data = daDatos[0,0].data
    umbral = daPeso.sel(stat='media')[0,0].data
    sd =  daPeso.sel(stat='sd')[0,0].data
    iaterrizaje = daEventos.sel(event='aterrizaje')[0,0].data
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")
    daCorte = xr.apply_ufunc(
        _detect_onset_aux,
        daDatos,
        daPeso.sel(stat="media"),
        daPeso.sel(stat="sd"),
        daEventos.sel(event="aterrizaje"),
        daDatos.ID,
        input_core_dims=[["time"], [], [], [], []],
        # output_core_dims=[['peso']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )

    # Ajusta el límite al inicio análisis
    daCorte = xr.where(
        daCorte.notnull(),
        daCorte.where(
            daCorte < daEventos.sel(event="finAnalisis"),
            daEventos.sel(event="finAnalisis"),
        ),
        np.nan,
    )

    return daCorte


def detecta_maxFz(daDatos, tipo_test, daPeso=None, daEventos=None) -> xr.DataArray:
    # from detecta import detect_peaks
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    if tipo_test in ["SJ", "SJPreac", "CMJ", "DJ", "DJ2P"]:

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ind = float(
                    np.argmax(data[ini:fin]) + ini
                )  # con -1 es el anterior a superar el umbral de 0. Coincide mejor con maxFz???
                # plt.plot(data[ini:fin])
                # plt.show()
                # detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return np.array([ind])

        """      
        data = daDatos[0,1,-1].data
        ini = daEventos[0,1].sel(event='iniMov').data
        fin = daEventos[0,1].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daEventos.sel(event="iniMov").data,
            daEventos.sel(event="despegue").data,
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
        )  # .assign_coords(event=['minFz'])

        return daCorte


def detecta_minFz(
    daDatos, tipo_test, daPeso=None, daEventos=None, umbral=10.0, show=False
) -> xr.DataArray:
    # from detecta import detect_peaks
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    if tipo_test == "CMJ":

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                if ini >= fin:  # puede pasar en SJ bien hechos
                    ind = ini
                else:
                    ind = float(np.argmin(data[ini:fin]) + ini)
                    # plt.plot(data[ini:fin])
                    # plt.show()
                # detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return np.array([ind])

        """       
        data = daDatos[0].data
        ini = daEventos[0].sel(event='iniMov').data
        fin = daEventos[0].sel(event='maxFlex').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daEventos.sel(event="iniMov").data,
            daEventos.sel(event="maxFlex").data,
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
        )  # .assign_coords(event=['minFz'])

    elif tipo_test in ["SJ", "SJPreac"]:

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                if ini >= fin:  # puede pasar en SJ bien hechos
                    ind = ini
                else:
                    ind = float(np.argmin(data[ini:fin]) + ini)
                    # plt.plot(data[ini:fin])
                    # plt.plot(int(ind-ini), data[int(ind)] ,'o')
                    # plt.show()
                # detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return np.array([ind])

        """       
        data = daDatos[0,0].data
        ini = daEventos[0,0].sel(event='iniMov').data
        fin = daEventos[0,0].sel(event='maxFz').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daEventos.sel(event="iniMov").data,
            daEventos.sel(event="maxFz").data,
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
        )  # .assign_coords(event=['minFz'])

    elif tipo_test in ["DJ", "DJ2P"]:

        def _detect_onset_aux(data, fin, **args_func_cortes):
            # plt.plot(data)
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            try:
                fin = int(fin)
                data = data[fin::-1]
                ind = detect_onset(-data, **args_func_cortes)[0, 0]
                ind = float(fin - ind)
            except:
                ind = np.nan  # por si no encuentra el criterio
            return ind  # +1 para que se quede con el que ya ha pasado el umbral

        """
        data= daDatos[0,1].data
        ini = daEventos[0,1].sel(event='iniMov').data
        fin = daEventos[0,1].sel(event='iniImpPos').data
        args_func_cortes = dict(threshold=-umbral, n_above=int(0.1*daDatos.freq), show=True)
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daEventos.sel(event="iniImpPos").data,
            input_core_dims=[["time"], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(threshold=-umbral, n_above=int(0.1 * daDatos.freq), show=show),
        )

    return daCorte


def detecta_ini_fin_impulso(
    daDatos, tipo_test, daPeso=None, daEventos=None, show=False
) -> xr.DataArray:
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    if tipo_test in ["SJ", "SJPreac", "CMJ", "DJ"]:

        def _detect_onset_aux(data, peso, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ini1 = detect_onset(
                    data[ini:fin],
                    threshold=peso,
                    n_above=int(0.1 * daDatos.freq),
                    show=show,
                )
                ind = (
                    ini + ini1[-1]
                )  # se queda con el último paso por el peso, próximo al despegue
                ind[1] += 1  # +1 para coger el que ya ha pasado por debajo del peso

                # #Evaluando si hay más de un evento
                # ind = ini + ini1[0] #si sólo ha detectado una subida y bajada
                # if len(ini1) > 1: #si ha detectado más de una subida y bajada
                #     ind[1] = ini + ini1[-1,-1]
                # ind[1] += 1 #+1 para coger el que ya ha pasado por debajo del peso

                # data[ini[1]+1] #peso
            except:
                ind = np.array([np.nan, np.nan])  # por si no encuentra el criterio
            return ind.astype("float")

        """
        data = daDatos[5,0].data
        peso = daPeso[5,0].sel(stat='media').data
        ini = daEventos[5,0].sel(event='iniMov').data
        fin = daEventos[5,0].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daPeso.sel(stat="media").data,
            daEventos.sel(event="iniMov"),
            daEventos.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
        ).assign_coords(event=["iniImpPos", "finImpPos"])

    elif tipo_test in ["DJ2P"]:

        def _detect_onset_aux(data, peso, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)

                # busca cuándo inicia primer despegue
                ini0 = detect_onset(
                    data, threshold=30.0, n_above=int(0.1 * daDatos.freq), show=False
                )[1, 0]
                ini1 = detect_onset(
                    data[ini0:fin],
                    threshold=peso,
                    n_above=int(0.1 * daDatos.freq),
                    show=False,
                )
                ind = ini0 + ini1[0]
                ind[1] += 1  # +1 para coger el que ya ha pasado por debajo del peso
                # data[ind[0]-1] #peso
            except:
                ind = np.array([np.nan, np.nan])  # por si no encuentra el criterio
            return ind.astype("float")

        """
        data = daDatos[1,0].data
        peso = daPeso[1,0].sel(stat='media').data
        ini = daEventos[1,0].sel(event='iniMov').data
        fin = daEventos[1,0].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            daPeso.sel(stat="media").data,
            daEventos.sel(event="iniMov"),
            daEventos.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
        ).assign_coords(event=["iniImpPos", "finImpPos"])

    return daCorte


def detecta_max_flex(
    daDatos, tipo_test="CMJ", v=None, daPeso=None, daEventos=None
) -> xr.DataArray:
    """
    Para calcular desde 'DJ' tiene que venir reversed y con la velocidad como
    parámetro.
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    if tipo_test in ["SJ", "SJPreac", "CMJ", "DJ", "DJ2P"]:

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ind = detect_onset(
                    data[ini:fin],
                    threshold=0,
                    n_above=int(0.01 * daDatos.freq),
                    show=False,
                )  # los datos que llegan de velocidad están cortados desde el iniMov
                ind = ind[0, 0] + ini
                # data[ind-5:ind+5] #data[ind]
                ind = float(ind)
                # data[ind-1] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return np.array(ind)

        # Calcula la velocidad, OJO, sin haber hecho el ajuste de offsetFz
        # TODO: CALCULAR VELOCIDAD DESDE FUNCIÓN EXTERNA
        if not isinstance(v, xr.DataArray):
            v = calcula_variables(
                daDatos,
                daPeso=daPeso,
                daEventos=daEventos.sel(event=["iniMov", "finMov"]),
            )["v"]
        # v = v.sel(axis='z') #se queda solo con axis z la haya calculado aquí o venga calculada del reversed
        # v.plot.line(x='time', col='ID', col_wrap=5, sharey=False)
        """
        data = v[0,:].data
        peso = daPeso[0].sel(stat='media').data
        ini = daEventos[0].sel(event='iniMov').data
        fin = daEventos[0].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            v,
            daEventos.sel(event="iniImpPos"),
            daEventos.sel(event="despegue"),
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

    return daCorte


def detecta_max_flex_desdeV(daDatos, tipo_test, daInstantes=None) -> xr.DataArray:
    if tipo_test == "DJ":
        return

    elif tipo_test == "CMJ":

        def _detect_onset_aux(data, peso, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ind = detect_onset(
                    data[ini:],
                    threshold=peso,
                    n_above=int(0.01 * daDatos.freq),
                    show=False,
                )
                # ind += ini
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return ind

        # data = daDatos[0,1,-1].data
        # ini = daInstantes[0,1].sel(event='iniMov').data
        # fin = daInstantes[0,1].sel(event='despegue').data

        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos.sel(axis="z"),
            daPeso.sel(stat="media").data,
            daInstantes.sel(event="iniMov"),
            daInstantes.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            # output_core_dims=[['peso']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )
        """
        def detect_iniMov_peso_XSD(data, peso, sdpeso, idespegue):
            #Parte del despegue hacia atrás buscando cuándo supera el umbral del peso - sd*SDx
            #ini = detect_onset(-data[:int(idespegue):-1], threshold=-umbral, n_above=50, show=True)[0]
            #ini = detect_onset(data[int(idespegue)::-1], threshold=umbral, n_above=5, show=True)
            #ini = idespegue - ini[1,0] + 1 #+1 para coger el que ya ha superado el umbral
            
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-sdpeso), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:ini1[0,0]-100:-1], threshold=peso, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha superado el umbral
                
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
        daCorte = xr.apply_ufunc(detect_iniMov_peso_XSD, daDatos.sel(axis='z'), daPeso.sel(stat='media'), daPeso.sel(stat='sd')*SDx, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """

        # Comprobaciones
        # daDatos.sel(axis='z').isel(time=daCorte-1) #

    return daCorte


def detecta_inicio_cero(daDatos, umbral=100) -> xr.DataArray:
    """Detecta la parte final si se ha salido de la plataforma antes de tiempo.
    Se le deben pasar los archivos ya detectados como que tienen el final incorrecto.
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")
        print("ajustado")

    def _detecta_inicio_cero_aux(data, umbral):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        try:
            ind = detect_onset(
                data, threshold=umbral, n_above=int(0.1 * daDatos.freq), show=False
            )
            ind = ind[0, 0]  # +1 para coger el que ya ha superado el umbral
        except:
            ind = len(data)  # por si no encuentra el criterio. Poner nan?
        return float(ind)

    """
    data = daDatos[0].data    
    """
    da = xr.apply_ufunc(
        _detecta_inicio_cero_aux,
        daDatos,
        umbral,
        input_core_dims=[["time"], []],
        # output_core_dims=[['peso']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )
    return da


def detecta_final_cero(daDatos, umbral=100) -> xr.DataArray:
    """Detecta la parte final si se ha salido de la plataforma antes de tiempo.
    Se le deben pasar los archivos ya detectados como que tienen el final incorrecto.
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")

    def _detecta_final_cero_aux(data, umbral):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan
        try:
            ind = detect_onset(
                data, threshold=umbral, n_above=int(0.1 * daDatos.freq), show=False
            )
            ind = ind[-1, 1]
        except:
            ind = len(data)  # por si no encuentra el criterio. Poner nan?
        return float(ind)

    da = xr.apply_ufunc(
        _detecta_final_cero_aux,
        daDatos,
        umbral,
        input_core_dims=[["time"], []],
        # output_core_dims=[['peso']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )
    return da


def detecta_eventos_estandar(
    daDatos,
    daEventos=None,
    daPeso=None,
    tipo_test="CMJ",
    tipo_calculo_fin_mov="fuerza",
    umbral=30.0,
    SDx=5,
) -> xr.DataArray:
    if daPeso is None:
        raise Exception("No has introducido los datos del peso")

    if daEventos is None:
        daEventos = crea_eventos_saltos_estandar(daDatos)

    # Despegue y aterrizaje definitivo
    daEventos.loc[dict(event=["despegue", "aterrizaje"])] = detecta_despegue_aterrizaje(
        daDatos, tipo_test=tipo_test, umbral=umbral
    )  # , eventos=daEventosCMJ)

    # Inicio movimiento, después de detectar el despegue
    daEventos.loc[dict(event="iniMov")] = detecta_ini_mov(
        daDatos,
        tipo_test=tipo_test,
        daPeso=daPeso,
        daEventos=daEventos,
        umbral=umbral,
        SDx=SDx,
    )  # .sel(event='despegue'))

    # Final del movimiento
    daEventos.loc[dict(event="finMov")] = detecta_fin_mov(
        daDatos,
        tipo_test=tipo_test,
        daPeso=daPeso,
        daEventos=daEventos,
        tipo_calculo=tipo_calculo_fin_mov,
    )  # .sel(event='aterrizaje'))

    # Ini y fin del impulso positivo
    daEventos.loc[dict(event=["iniImpPos", "finImpPos"])] = detecta_ini_fin_impulso(
        daDatos, tipo_test=tipo_test, daPeso=daPeso, daEventos=daEventos
    )  # .sel(event='iniMov'))

    # Maxima flexión rodillas batida
    if tipo_test not in ["DJ", "SJPreac"]:
        daEventos.loc[dict(event="maxFlex")] = detecta_max_flex(
            daDatos, tipo_test=tipo_test, daPeso=daPeso, daEventos=daEventos
        )

    # MaxFz, entre de iniMov y despegue
    daEventos.loc[dict(event="maxFz")] = detecta_maxFz(
        daDatos, tipo_test=tipo_test, daPeso=daPeso, daEventos=daEventos
    )

    # MinFz, entre de iniMov y maxFlex
    daEventos.loc[dict(event="minFz")] = detecta_minFz(
        daDatos, tipo_test=tipo_test, daPeso=daPeso, daEventos=daEventos, umbral=umbral
    )

    if "preactiv" in daEventos.event:
        daEventos.loc[dict(event="preactiv")] = (
            daEventos.loc[dict(event="iniMov")] - np.array(0.5) * daDatos.freq
        )  # para calcular preactivación en ventana de 0.5 s

    return daEventos


colr = {
    "iniAnalisis": "grey",
    "finAnalisis": "grey",
    "iniMov": "C0",
    "finMov": "C0",
    "preactiv": "dodgerblue",
    "iniPeso": "deepskyblue",
    "finPeso": "deepskyblue",
    #'iniPeso2':'deepskyblue', 'finPeso2':'deepskyblue', #para cuando se ponen a la vez el peso al inicio y al final
    "iniImpPos": "orange",
    "finImpPos": "orange",
    "despegue": "r",
    "aterrizaje": "r",
    "maxFz": "brown",
    "minFz": "cyan",
    "maxFlex": "g",
}


def _completar_en_grafica_xr(g, ajusta_inifin, daPeso, daEventos) -> None:
    for h, ax in enumerate(g.axs):  # .axes): #extrae cada fila
        for i in range(len(ax)):  # extrae cada axis (gráfica)
            dimensiones = g.name_dicts[h, i]
            if (
                dimensiones is None
            ):  # para cuando quedan huecos al final de la cuadrícula
                continue
            # print(dimensiones)
            # plt.plot(g.data.sel(g.name_dicts[h, i]))
            # ID = str(g.data.loc[g.name_dicts[h, i]].ID.data)
            if "repe" not in g.data.dims:
                ax[i].set_title(
                    str(g.data.loc[g.name_dicts[h, i]].ID.data)
                )  # pone el nombre completo porque a veces lo recorta

            if daPeso is not None:  # isinstance(daPeso, xr.DataArray):
                # Pasar pesos solamente cuando se grafiquen fuerzas absolutas
                ax[i].axhline(
                    daPeso.sel(dimensiones).sel(stat="media").data,
                    color="C0",
                    lw=0.7,
                    ls="--",
                    dash_capstyle="round",
                    alpha=0.7,
                )

            if isinstance(daEventos, xr.DataArray):
                for ev in daEventos.sel(dimensiones):  # .event:
                    # print(ev.data)
                    if (
                        ev.isnull().all() or ev.count() > 1
                    ):  # np.isnan(ev): #si no existe el evento
                        continue

                    # No muestra ventana búsqueda peso si ajusta ini-fin
                    if (
                        str(ev.event.data) in ["iniPeso", "finPeso"] and ajusta_inifin
                    ):  # se salta estos dos porque el array viene cortado por sus valores y tienen escala distinta
                        continue
                    # print(str(ev.data))

                    # Si no es un evento conocido le pone un color cualquiera
                    try:
                        col = colr[str(ev.event.data)]
                    except:
                        col = "k"
                    ax[i].axvline(
                        x=ev / g.data.freq,
                        c=col,
                        lw=1,
                        ls="--",
                        dashes=(5, 5),
                        dash_capstyle="round",
                        alpha=0.6,
                    )

                    # Ajusta altura de etiquetas
                    if str(ev.event.data) in ["iniImpPos", "finImpPos"]:
                        y_texto = ax[i].get_ylim()[1] * 0.7
                    elif str(ev.event.data) in ["minFz", "maxFlex"]:
                        y_texto = ax[i].get_ylim()[1] * 0.8
                    else:
                        y_texto = ax[i].get_ylim()[1] * 0.97

                    ax[i].text(
                        ev / g.data.freq,
                        y_texto,
                        ev.event.data,
                        ha="right",
                        va="top",
                        rotation="vertical",
                        c="k",
                        alpha=0.7,
                        fontsize="xx-small",
                        bbox=dict(
                            facecolor=col,
                            alpha=0.3,
                            edgecolor="none",
                            boxstyle="round,pad=0.3" + ",rounding_size=.5",
                        ),
                        transform=ax[i].transData,
                    )
            if ajusta_inifin:
                fr = g.data.freq  # .loc[g.name_dicts[0, 0]]
                plt.xlim(
                    [
                        daEventos.sel(dimensiones).sel(event="iniAnalisis") / fr,
                        daEventos.sel(dimensiones).sel(event="finAnalisis") / fr,
                    ]
                )


def graficas_eventos(
    daDatos,
    daEventos=None,
    daPeso=None,
    n_en_bloque=4,
    show_in_console=True,
    ajusta_inifin=False,
    sharey=False,
    ruta_trabajo=None,
    nom_archivo_graf_global=None,
) -> None:
    """
    n_en_bloque sirve para ajustar el nº de gráficas por hoja. Si los datos
    tienen dimensión repe, n_en_bloque indica el nº de filas por hoja con repe columnas.
    Si no tienen repe, en cada hoja van n_en_bloque x n_en_bloque gráficas.
    """
    timerGraf = time.perf_counter()  # inicia el contador de tiempo
    print("\nCreando gráficas...")

    # import seaborn as sns

    # Si no se incluye nombre archivo no guarda el pdf
    if nom_archivo_graf_global != None:
        if not isinstance(ruta_trabajo, Path):
            ruta_trabajo = Path(ruta_trabajo)
        nompdf = (ruta_trabajo / nom_archivo_graf_global).with_suffix(".pdf")
        pdf_pages = PdfPages(nompdf)

    if "axis" in daDatos.dims:  # por si se envía un da filtrado por axis
        daDatos = daDatos.sel(axis="z")
    if (
        daEventos is not None and "axis" in daEventos.dims
    ):  # por si se envía un da filtrado por eje
        daEventos = daEventos.sel(axis="z")
    if (
        daPeso is not None and "axis" in daPeso.dims
    ):  # por si se envía un da filtrado por eje
        daPeso = daPeso.sel(axis="z")

    if ajusta_inifin:
        ini = daEventos.isel(event=daEventos.argmin(dim="event"))
        daDatos = recorta_ventana_analisis(
            daDatos, daEventos.sel(event=["iniAnalisis", "finAnalisis"])
        )
        if daEventos is not None:
            daEventos = daEventos - daEventos.sel(event="iniAnalisis")

    # Por si no hay dimensión 'repe'
    if "repe" in daDatos.dims:  # dfDatos.columns:
        fils_cols = dict(row="ID", col="repe")
    else:
        fils_cols = dict(col="ID", col_wrap=n_en_bloque)

    if "repe" in daDatos.dims:
        distribuidor = n_en_bloque
    else:
        distribuidor = n_en_bloque**2

    # daDatos.drop_vars('tipo').plot.line(x='time',col='ID')
    # daDatos.to_pandas().T.plot()
    for n in range(0, len(daDatos.ID), distribuidor):
        dax = daDatos.isel(ID=slice(n, n + distribuidor))

        g = dax.plot.line(
            x="time", alpha=0.8, aspect=1.5, sharey=sharey, **fils_cols
        )  # , lw=1)
        _completar_en_grafica_xr(g, ajusta_inifin, daPeso, daEventos)

        if nom_archivo_graf_global is not None:
            pdf_pages.savefig(g.fig)

        print(f"Completadas gráficas {n} a {n + distribuidor} de {len(daDatos.ID)}")

        if not show_in_console:
            plt.close()  # para que no muestre las gráficas en consola y vaya más rápido

    # if 'repe' not in daDatos.dims and n_en_bloque is not None:
    #     #for n, dax in daDatos.assign_coords(ID=np.arange(len(daDatos.ID))).groupby_bins('ID', bins=range(0, len(daDatos.ID) + n_en_bloque**2, n_en_bloque**2), include_lowest=True):
    #     for n in range(0,len(daDatos.ID), n_en_bloque**2):
    #         dax = daDatos.isel(ID=slice(n,n+n_en_bloque**2))

    #         g=dax.plot.line(x='time', alpha=0.8, aspect=1.5, sharey=False, **fils_cols, lw=1)
    #         completa_grafica_xr(g)

    #         if nom_archivo_graf_global is not None:
    #             pdf_pages.savefig(g.fig)

    # else:
    #     for n in range(0,len(daDatos.ID), n_en_bloque):
    #         dax = daDatos.isel(ID=slice(n,n+n_en_bloque))

    #         g=dax.plot.line(x='time', alpha=0.8, aspect=1.5, sharey=False, **fils_cols, lw=1)
    #         completa_grafica_xr(g)

    #     if nom_archivo_graf_global is not None:
    #         pdf_pages.savefig(g.fig)

    """
    def fun(x,y): #prueba para dibujar con xarray directamente
        print(x,y)
    g=daDatos.isel(ID=slice(None,3)).plot.line(x='time', **fils_cols)
    g.map_dataarray_line(fun, x='time', y='fuerza', hue='repe')
    """
    # g = sns.relplot(data=dfDatos, x='time', y='Fuerza', col='ID', col_wrap=4, hue='repe',
    #                 estimator=None, ci=95, units='repe',
    #                 facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
    #                 palette=sns.color_palette(col), alpha=0.7)

    """
    #Versión Seaborn
    def dibuja_X(x,y, color, **kwargs):   
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        #print(y, ID, repe, color, kwargs.keys())
        #plt.vlines(daEventos.sel(ID=ID, repe=repe)/daDatos.freq, ymin=kwargs['data'].loc[:,'Fuerza'].min(), ymax=kwargs['data'].loc[:,'Fuerza'].max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
        #Líneas del peso
        if daPeso is not None: #isinstance(daPeso, xr.DataArray):
            plt.axhline(daPeso.sel(ID=ID, repe=repe, stat='media').data, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.6)
       
        for ev in daEventos.sel(ID=ID, repe=repe).event:
            if str(ev.data) not in ['iniAnalisis', 'finAnalisis']: #se salta estos dos porque el array viene cortado por sus valores y tienen escala distinta
                #print(str(ev.data))
                #print(daEventos.sel(ID=ID, repe=repe,event=ev))
            # for num, ev in daEventos.sel(ID=ID, repe=repe).groupby('evento'):
            #     print('\n',num)
                if not np.isnan(daEventos.sel(ID=ID, repe=repe, event=ev)): #si existe el evento
                    plt.axvline(x=daEventos.sel(ID=ID, repe=repe, event=ev)/daDatos.freq, c=col[str(ev.data)], lw=0.5, ls='--', dashes=(5, 5), dash_capstyle='round', alpha=0.5)
                    y_texto = plt.gca().get_ylim()[1] if str(ev.data) not in ['minFz', 'despegue', 'maxFz'] else plt.gca().get_ylim()[1]*0.8
                    plt.text(daEventos.sel(ID=ID, repe=repe, event=ev).data/daDatos.freq, y_texto, ev.data,
                             ha='right', va='top', rotation='vertical', c='k', alpha=0.6, fontsize=8, 
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), transform=plt.gca().transData)
    
    g = sns.relplot(data=dfDatos[dfDatos['Fuerza'].notnull()], x='time', y='Fuerza', #ATENCIÓN en esta versión de seaborn (12.1) falla con datos nan, por eso se seleccionan los notnull()                    
                    #estimator=None, ci=95, units='repe',
                    facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
                    alpha=0.7, aspect=1.5,
                    **fils_cols) #palette=sns.color_palette(col), 
    if daEventos is not None:
        g.map_dataframe(dibuja_X, x='time', y='Fuerza', lw=0.25, alpha=0.3)
    """

    """
    def dibuja_xr(x,y, **kwargs):
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        print(y, ID, repe, color, kwargs.keys())
    
    g=daDatos.sel(axis='z').plot.line(x='time', col='ID', col_wrap=4, hue='repe', sharey=False)
    #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
    g.map_dataarray_line(dibuja_xr, x='time', y=None, hue='repe')#, y='trial')
    col=['C0', 'C1', 'C2']
    for h, ax in enumerate(g.axs): #extrae cada fila
        for i in range(len(ax)): #extrae cada axis (gráfica)     
            try:
                idn = g.data.loc[g.name_dicts[h, i]].ID
                #print('peso=', daPeso.sel(ID=idn).data)#idn)
                #Rango medida peso
                #ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.freq)], g.data.time[int(ventana[1]*self.datos.freq)], alpha=0.2, color='C1')
                #for j in daDatos.repe:
                for e in daEventos.sel(ID=idn):
                    #print(e)
                    ax[i].vlines(e/daDatos.freq, ymin=g.data.sel(ID=idn).min(), ymax=g.data.sel(ID=idn).max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
            except:
                print("No va el", h,i)
    """

    # Cierra el pdf
    if nom_archivo_graf_global is not None:
        # pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print(f"\nGuardada la gráfica {nompdf}")
    print(
        "Creadas las gráficas en {0:.3f} s \n".format(time.perf_counter() - timerGraf)
    )


def graficas_todas_variables(
    dsDatos,
    show_eventos=False,
    daPeso=None,
    n_en_bloque=4,
    show_in_console=True,
    ajusta_inifin=False,
    ruta_trabajo=None,
    nom_archivo_graf_global=None,
) -> None:
    """
    Plots all variables in the dataset, with options for block size, console display,
    adjustment, working directory, and global file name.
    ajusta_inifin: si es True recorta según iniAnalisis y fin Analisis.
    """

    if "axis" in dsDatos.dims:  # por si se envía un da filtrado por axis
        dsDatos = dsDatos.sel(axis="z")

    if n_en_bloque > len(dsDatos.ID):
        n_en_bloque = len(dsDatos.ID)

    daDatos = dsDatos[["BW", "v", "s", "P"]].to_array()

    daEventos = dsDatos["events"] if "events" in list(dsDatos.keys()) else None
    daPeso = dsDatos["peso"] if "peso" in list(dsDatos.keys()) else None

    # daDatos.loc[dict(variable='P')] = daDatos.loc[dict(variable='P')] / daPeso.sel(stat='media')

    timerGraf = time.time()  # inicia el contador de tiempo
    print("\nCreando gráficas...")

    # import seaborn as sns

    # Si no se incluye nombre archivo no guarda el pdf
    if nom_archivo_graf_global != None:
        if not isinstance(ruta_trabajo, Path):
            ruta_trabajo = Path(ruta_trabajo)
        nompdf = (ruta_trabajo / nom_archivo_graf_global).with_suffix(".pdf")
        pdf_pages = PdfPages(nompdf)

    # if 'axis' in daDatos.dims: #por si se envía un da filtrado por axis
    #     daDatos=daDatos.sel(axis='z')
    # if daEventos is not None and 'axis' in daEventos.dims: #por si se envía un da filtrado por eje
    #     daEventos=daEventos.sel(axis='z')
    # if daPeso is not None and 'axis' in daPeso.dims: #por si se envía un da filtrado por eje
    #     daPeso=daPeso.sel(axis='z')

    if ajusta_inifin:
        daDatos = recorta_ventana_analisis(
            daDatos, daEventos.sel(event=["iniAnalisis", "finAnalisis"])
        )
        if daEventos:  # is not None:
            daEventos = daEventos - daEventos.sel(event="iniAnalisis")

    # Por si no hay dimensión 'repe'
    if "repe" in daDatos.dims:  # dfDatos.columns:
        fils_cols = dict(row="ID", col="repe")
        distribuidor = n_en_bloque
    else:
        fils_cols = dict(col="ID", col_wrap=n_en_bloque)
        distribuidor = n_en_bloque**2

    for n in range(0, len(daDatos.ID), distribuidor):
        dax = daDatos.isel(ID=slice(n, n + distribuidor))

        g = dax.plot.line(
            x="time", alpha=0.8, aspect=1.5, sharey=False, **fils_cols
        )  # , lw=1)
        _completar_en_grafica_xr(g, ajusta_inifin, daPeso, daEventos)

        if nom_archivo_graf_global is not None:
            pdf_pages.savefig(g.fig)

        print(f"Completadas gráficas {n} a {n + distribuidor} de {len(daDatos.ID)}")

        if not show_in_console:
            plt.close()  # para que no muestre las gráficas en consola y vaya más rápido

    # Cierra el pdf
    if nom_archivo_graf_global is not None:
        # pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print(f"Guardada la gráfica {nompdf}")
    print("Creadas las gráficas en {0:.3f} s \n".format(time.time() - timerGraf))


def graficas_chequeo_peso(
    daDatos,
    daEventos=None,
    dsPesos=None,
    daPeso=None,
    umbral_dif_peso=20,
    daPeso_med=None,
    margen_ventana=0.6,
    n_en_bloque=4,
    show_in_console=True,
    ruta_trabajo=None,
    nom_archivo_graf_global=None,
) -> None:
    """
    margen_ventana: tiempo en segundos por delante y por delante y detrás de iniPeso y finPeso
    """
    timerGraf = time.time()  # inicia el contador de tiempo
    print("\nCreando gráficas...")

    # import seaborn as sns

    # Si no se incluye nombre archivo no guarda el pdf
    if nom_archivo_graf_global != None:
        if not isinstance(ruta_trabajo, Path):
            ruta_trabajo = Path(ruta_trabajo)
        nompdf = (ruta_trabajo / nom_archivo_graf_global).with_suffix(".pdf")
        pdf_pages = PdfPages(nompdf)

    if isinstance(margen_ventana, float):
        margen_ventana = margen_ventana * daDatos.freq

    if "axis" in daDatos.dims:  # por si se envía un da filtrado por axis
        daDatos = daDatos.sel(axis="z")
    if (
        daEventos is not None and "axis" in daEventos.dims
    ):  # por si se envía un da filtrado por eje
        daEventos = daEventos.sel(axis="z")
    if (
        dsPesos is not None and "axis" in dsPesos.dims
    ):  # por si se envía un da filtrado por eje
        dsPesos = dsPesos.sel(axis="z")
    if (
        daPeso is not None and "axis" in daPeso.dims
    ):  # por si se envía un da filtrado por eje
        daPeso = daPeso.sel(axis="z")
    if (
        daPeso_med is not None and "axis" in daPeso_med.dims
    ):  # por si se envía un da filtrado por eje
        daPeso_med = daPeso_med.sel(axis="z")

    # Por si no hay dimensión 'repe'
    if "repe" in daDatos.dims:  # dfDatos.columns:
        fils_cols = dict(row="ID", col="repe")
    else:
        fils_cols = dict(col="ID", col_wrap=n_en_bloque)

    if "repe" in daDatos.dims:
        distribuidor = n_en_bloque
    else:
        distribuidor = n_en_bloque**2

    """
    def completa_peso(g, daEventos, daPeso, daPeso_med):
        for h, ax in enumerate(g.axs):#.axes): #extrae cada fila
            for i in range(len(ax)): #extrae cada axis (gráfica)                    
                dimensiones = g.name_dicts[h, i]
                peso_afinado = daPeso.sel(dimensiones).sel(stat='media').data
                peso_media = daPeso_med.sel(dimensiones).sel(stat='media').data
                
                #print(dimensiones)
                if dimensiones is None: #para cuando quedan huecos al final de la cuadrícula
                    continue
                #plt.plot(g.data.sel(g.name_dicts[h, i]))
                #ID = str(g.data.loc[g.name_dicts[h, i]].ID.data)
                if 'repe' not in g.data.dims:
                    ax[i].set_title(str(g.data.loc[g.name_dicts[h, i]].ID.data)) #pone el nombre completo porque a veces lo recorta
                
                
                if daPeso is not None: #isinstance(daPeso, xr.DataArray):
                    ax[i].axhline(peso_afinado, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
                    ax[i].text(0.0, peso_afinado, 'peso afinado', 
                                ha='left', va='bottom', rotation='horizontal', c='C0', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transData
                                )
                    ax[i].text(0.05, 0.1, f'Peso afinado={peso_afinado:.1f} N', 
                                ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transAxes
                                )
            
                
                if daPeso_med is not None: #isinstance(daPeso, xr.DataArray):
                    ax[i].axhline(peso_media, color='C1', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
                    ax[i].text(0.3, peso_media, 'peso media', 
                                ha='left', va='bottom', rotation='horizontal', c='C1', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transData
                                )
                    ax[i].text(0.05, 0.05, f'Peso media={peso_media:.1f} N', 
                                ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transAxes
                                )
                    
                #Si la diferencia es mayor que el umbral, avisa
                if abs(peso_media-peso_afinado) > umbral_dif_peso:
                    ax[i].text(0.5, 0.9, f'REVISAR (dif={peso_media-peso_afinado:.1f} N)', 
                                ha='center', va='center', rotation='horizontal', c='r', alpha=0.7, fontsize='large', 
                                bbox=dict(facecolor='lightgrey', alpha=0.3, edgecolor='r', boxstyle='round,pad=0.3'+',rounding_size=.5'), 
                                transform=ax[i].transAxes
                                )
                
                
                if isinstance(daEventos, xr.DataArray):
                    for ev in daEventos.sel(dimensiones):#.event:
                        if ev.isnull().all():#np.isnan(ev): #si no existe el evento
                            continue                       
                        
                        ax[i].axvline(x=ev/g.data.freq, c=colr[str(ev.event.data)], lw=1, ls='--', dashes=(5, 5), dash_capstyle='round', alpha=0.7)
                        
                        y_texto = ax[i].get_ylim()[1]*0.97
                        ax[i].text(ev / g.data.freq, y_texto, ev.event.data,
                                 ha='right', va='top', rotation='vertical', c='k', alpha=0.7, fontsize='small', 
                                 bbox=dict(facecolor=colr[str(ev.event.data)], alpha=0.2, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), 
                                 transform=ax[i].transData)
    """

    ventana = daEventos.sel(event=["iniPeso", "finPeso"]).copy()
    ventana.loc[dict(event="iniPeso")] = xr.where(
        ventana.loc[dict(event="iniPeso")] - margen_ventana > 0,
        ventana.loc[dict(event="iniPeso")] - margen_ventana,
        0,
    )  # ventana.loc[dict(event='iniPeso')] - margen_ventana
    ventana.loc[dict(event="finPeso")] = xr.where(
        ventana.loc[dict(event="finPeso")] + margen_ventana
        < daEventos.loc[dict(event="finAnalisis")],
        ventana.loc[dict(event="finPeso")] + margen_ventana,
        daEventos.loc[dict(event="finAnalisis")],
    )  # ventana.loc[dict(event='finPeso')] + margen_ventana
    # #TODO: ESTO NO SE AJUSTA BIEN CUANDO LA VENTANA ESTÁ CERCA DE CERO Y SALE NEGATIVO
    # ventana = xr.where(ventana.loc[dict(event='iniPeso')] < 0, ventana - ventana.loc[dict(event='iniPeso')], ventana)
    # ventana = xr.where(ventana.loc[dict(event='finPeso')] > len(daDatos.time), ventana - (ventana.loc[dict(event='finPeso')] - len(daDatos.time)), ventana)

    daDat = recorta_ventana_analisis(daDatos, ventana)
    daEventos = daEventos - ventana.loc[dict(event="iniPeso")]

    # ventana = ventana - ventana.sel(event='iniPeso') #+ 0.5*daDatos.freq
    # ventana.loc[dict(event='iniPeso')] = ventana.loc[dict(event='iniPeso')] + margen_ventana*daDat.freq
    # ventana.loc[dict(event='finPeso')] = ventana.loc[dict(event='finPeso')] - margen_ventana*daDat.freq

    """
    for n in range(0,len(daDatos.ID), distribuidor):
        dax = daDat.isel(ID=slice(n, n + distribuidor))
        
        g=dax.plot.line(x='time', alpha=0.8, aspect=1.5, color='lightgrey', sharey=False, **fils_cols) #, lw=1)
        completa_peso(g, daEventos.sel(event=['iniPeso', 'finPeso']), daPeso, daPeso_med)
                        
        if nom_archivo_graf_global is not None:
            pdf_pages.savefig(g.fig)
            
        print(f'Completadas gráficas {n} a {n + distribuidor} de {len(daDatos.ID)}')
        
        if not show_in_console: plt.close() #para que no muestre las gráficas en consola y vaya más rápido
    """

    def _completa_graf_xr(*args, color):
        ID, time = args
        data = g.data.loc[dict(ID=ID)]
        # peso_afinado = daPeso.sel(ID=ID).sel(stat='media').data
        # peso_media = daPeso_med.sel(ID=ID).sel(stat='media').data

        if "repe" not in g.data.dims:
            plt.title(ID)  # pone el nombre completo porque a veces lo recorta

        """
        if peso_afinado is not None: #isinstance(daPeso, xr.DataArray):
            plt.axhline(peso_afinado, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
            plt.text(0.0, peso_afinado, 'peso afinado', 
                        ha='left', va='bottom', rotation='horizontal', c='C0', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transData
                        )
            plt.text(0.05, 0.1, f'Peso afinado={peso_afinado:.1f} N', 
                        ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transAxes
                        )
        if peso_media is not None: #isinstance(daPeso, xr.DataArray):
            plt.axhline(peso_media, color='C1', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
            plt.text(0.3, peso_media, 'peso media', 
                        ha='left', va='bottom', rotation='horizontal', c='C1', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transData
                        )
            plt.text(0.05, 0.05, f'Peso media={peso_media:.1f} N', 
                        ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transAxes
                        )
        #Si la diferencia es mayor que el umbral, avisa
        if abs(peso_media-peso_afinado) > umbral_dif_peso:
            plt.text(0.5, 0.9, f'REVISAR (dif={peso_media-peso_afinado:.1f} N)', 
                        ha='center', va='center', rotation='horizontal', c='r', alpha=0.7, fontsize='large', 
                        bbox=dict(facecolor='lightgrey', alpha=0.3, edgecolor='r', boxstyle='round,pad=0.3'+',rounding_size=.5'), 
                        transform=plt.gca().transAxes
                        )
        """
        # Dibuja líneas distintos tipos de cálculo peso
        if isinstance(dsPesos, xr.Dataset):
            color = ["C0", "C1", "C2"]
            for n, n_tipo_peso in enumerate(dsPesos.sel(ID=ID)):
                peso = dsPesos[n_tipo_peso].sel(ID=ID).data
                plt.axhline(
                    peso,
                    color=color[n],
                    lw=1,
                    ls="--",
                    dash_capstyle="round",
                    alpha=0.7,
                )
                plt.text(
                    0.3,
                    peso,
                    n_tipo_peso,
                    ha="left",
                    va="bottom",
                    rotation="horizontal",
                    c=color[n],
                    alpha=0.8,
                    fontsize="x-small",
                    transform=plt.gca().transData,
                )
                plt.text(
                    0.0,
                    peso,
                    f"Peso {n_tipo_peso}={peso:.1f} N",
                    ha="left",
                    va="bottom",
                    rotation="horizontal",
                    c=color[n],
                    alpha=0.7,
                    fontsize="x-small",
                    transform=plt.gca().transData,
                )

                if n > 0:
                    # Si la diferencia es mayor que el umbral, avisa
                    if abs(peso - dsPesos["media"].sel(ID=ID)) > umbral_dif_peso:
                        plt.text(
                            0.5,
                            0.9,
                            f'REVISAR (delta={peso-dsPesos["media"].sel(ID=ID):.1f} N)',
                            ha="center",
                            va="center",
                            rotation="horizontal",
                            c="r",
                            alpha=0.7,
                            fontsize="large",
                            bbox=dict(
                                facecolor="lightgrey",
                                alpha=0.3,
                                edgecolor="r",
                                boxstyle="round,pad=0.3" + ",rounding_size=.5",
                            ),
                            transform=plt.gca().transAxes,
                        )

        if isinstance(daEventos, xr.DataArray):
            for ev in daEventos.sel(ID=ID, event=["iniPeso", "finPeso"]):  # .event:
                if ev.isnull().all():  # np.isnan(ev): #si no existe el evento
                    continue

                plt.axvline(
                    x=ev / g.data.freq,
                    c=colr[str(ev.event.data)],
                    lw=1,
                    ls="--",
                    dashes=(5, 5),
                    dash_capstyle="round",
                    alpha=0.7,
                )
                y_texto = (
                    plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
                ) * 0.97 + plt.gca().get_ylim()[
                    0
                ]  # escala a las coordenadas de la variable en cada gráfica
                plt.text(
                    ev / g.data.freq,
                    y_texto,
                    ev.event.data,
                    ha="right",
                    va="top",
                    rotation="vertical",
                    c="k",
                    alpha=0.7,
                    fontsize="small",
                    bbox=dict(
                        facecolor=colr[str(ev.event.data)],
                        alpha=0.2,
                        edgecolor="none",
                        boxstyle="round,pad=0.3" + ",rounding_size=.5",
                    ),
                    transform=plt.gca().transData,
                )

    for n in range(0, len(daDatos.ID), distribuidor):
        dax = daDat.isel(ID=slice(n, n + distribuidor))

        g = dax.plot.line(
            x="time",
            alpha=0.8,
            aspect=1.5,
            color="lightgrey",
            sharey=False,
            **fils_cols,
        )  # , lw=1)
        g.map(_completa_graf_xr, "ID", "time", color=0)
        # completa_graf_xr(g, daEventos.sel(event=['iniPeso', 'finPeso']), daPeso, daPeso_med)

        if nom_archivo_graf_global is not None:
            pdf_pages.savefig(g.fig)

        print(f"Completadas gráficas {n} a {n + distribuidor} de {len(daDatos.ID)}")

        if not show_in_console:
            plt.close()  # para que no muestre las gráficas en consola y vaya más rápido

    # Cierra el pdf
    if nom_archivo_graf_global is not None:
        # pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print(f"Guardada la gráfica {nompdf}")
    print("Creadas las gráficas en {0:.3f} s \n".format(time.time() - timerGraf))


def recorta_ventana_analisis(daDatos, daEvents=None, ventana=None) -> xr.DataArray:
    """
    Si se pasa un valor a ventana, debería pasarse sólo un evento.
    Suma la ventana (en segundos) al evento inicial
    """
    # TODO: PROBAR CON DA.PAD

    def _corta_ventana(datos, ini, fin):
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

    if ventana is not None:
        if ventana > 0:
            daIni = daEvents
            daFin = daEvents + ventana * daDatos.freq
        else:
            daIni = daEvents + ventana * daDatos.freq
            daFin = daEvents

    else:
        daIni = daEvents.isel(event=0)
        daFin = daEvents.isel(event=1)

    daCortado = (
        xr.apply_ufunc(
            _corta_ventana,
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


def ajusta_offsetFz_vuelo_min(
    daDatos, tipo_test=None, umbral=20.0, pcto_ventana=5, show=False
) -> xr.DataArray:
    # Hace media de valores por debajo del umbral. Si es DJ2P considera los dos tramos en vuelo
    # Mejor aplicarlo antes de filtrar
    daReturn = daDatos.copy()

    if "axis" in daDatos.dims:
        daReturn = daReturn.sel(axis="z")

    minim = daReturn.min("time")
    # minim.sel(ID='S06_DJ_30', repe=2)
    offset = minim + umbral
    # offset.sel(ID='S06_DJ_30', repe=2)
    resta = daReturn.where(daReturn < offset).mean("time")
    # resta.sel(ID='S06_DJ_30', repe=2)
    # daReturn.sel(ID='S06_DJ_30', repe=2).plot.line(x='time')
    daReturn = daReturn - resta

    if show:
        if "plat" in daDatos.dims:
            if "repe" in daDatos.dims:
                daReturn.stack(ID_repe=("ID", "repe")).where(
                    daReturn.stack(ID_repe=("ID", "repe"))
                    < offset.stack(ID_repe=("ID", "repe"))
                ).plot.line(x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7)
                # Sospechosos
                sospechosos = daReturn.stack(ID_repe=("ID", "repe", "plat")).where(
                    np.abs(resta.stack(ID_repe=("ID", "repe", "plat"))) > umbral,
                    drop=True,
                )
                sospechosos.plot.line(
                    x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                )
                print(f"Sospechosos {sospechosos.ID_repe.data}")
        else:
            if "repe" in daDatos.dims:
                daReturn.stack(ID_repe=("ID", "repe")).where(
                    daReturn.stack(ID_repe=("ID", "repe"))
                    < offset.stack(ID_repe=("ID", "repe"))
                ).plot.line(x="time", col="ID", col_wrap=4)
            else:
                daReturn.where(daReturn < umbral * 2.5).plot.line(
                    x="time", col="ID", col_wrap=4
                )

    daReturn = daReturn - resta

    if "axis" in daDatos.dims:
        daDatos.loc[dict(axis="z")] = daReturn
    else:
        daDatos = daReturn

    return daDatos


def ajusta_offsetFz(
    daDatos,
    tipo_test=None,
    umbral=20.0,
    pcto_ventana=5,
    tipo_calculo="vuelo",
    show=False,
) -> xr.DataArray:
    """
    "The offset voltage of the unloaded force platform was determined by finding the 0.4 s
    moving average during the flight phase with the smallest standard deviation."
    Street, G., McMillan, S., Board, W., Rasmussen, M., & Heneghan, J. M. (2001). Sources of Error in Determining Countermovement Jump Height with the Impulse Method. Journal of Applied Biomechanics, 17(1), 43-54. https://doi.org/10.1123/jab.17.1.43

    """
    # Mejor aplicarlo antes de filtrar
    """
    tipo_calculo puede ser: media, vuelo, umbral
    """

    daReturn = daDatos.copy()
    if "axis" in daDatos.dims:
        daReturn = daReturn.sel(axis="z")

    if tipo_calculo == "media":
        if tipo_test == "DJ2PApart":
            # Se asume que es cero plat1 al principio y plat2 al final
            offset_plat1 = daReturn.sel(plat=1, time=slice(0, 1)).mean("time")
            offset_plat2 = daReturn.sel(
                plat=2, time=slice(daReturn.time[-1] - 1.5, daReturn.time[-1] - 0.5)
            ).mean("time")

            daReturn.loc[dict(plat=1)] -= offset_plat1
            daReturn.loc[dict(plat=2)] -= offset_plat2

            if show:
                if "plat" in daDatos.dims:
                    if "repe" in daDatos.dims:
                        daReturn.stack(ID_repe=("ID", "repe")).where(
                            daReturn.stack(ID_repe=("ID", "repe")) < umbral
                        ).plot.line(
                            x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                        )

                else:
                    if "repe" in daDatos.dims:
                        daReturn.stack(ID_repe=("ID", "repe")).where(
                            daReturn.stack(ID_repe=("ID", "repe")) < umbral
                        ).plot.line(x="time", col="ID", col_wrap=4)
                    else:
                        daReturn.where(daReturn < umbral).plot.line(
                            x="time", col="ID", col_wrap=4
                        )

    if tipo_calculo == "vuelo":  # sin comprobar
        # Ajusta buscando los vuelos concretamente
        if tipo_test == "DJ2P":
            # busca despegue y aterrizaje provisionales
            vuelo = detecta_despegue_aterrizaje(
                daReturn, tipo_test, umbral=umbral, show=show
            )
            recorte_ventana = (
                (
                    vuelo.loc[dict(event="aterrizaje")]
                    - vuelo.loc[dict(event="despegue")]
                )
                * pcto_ventana
                / 100
            ).astype("int32")
            vuelo.loc[dict(event="despegue")] += recorte_ventana
            vuelo.loc[dict(event="aterrizaje")] -= recorte_ventana
            offset_vuelo = recorta_ventana_analisis(daReturn, vuelo).mean(dim="time")

        else:
            # busca despegue y aterrizaje provisionales
            vuelo = detecta_despegue_aterrizaje(
                daReturn, tipo_test, umbral=umbral, show=show
            )
            # reduce la ventana un poco para evitar los rebotes posibles del filtrado
            recorte_ventana = (
                (
                    vuelo.loc[dict(event="aterrizaje")]
                    - vuelo.loc[dict(event="despegue")]
                )
                * pcto_ventana
                / 100
            ).astype("int32")
            vuelo.loc[dict(event="despegue")] += recorte_ventana
            vuelo.loc[dict(event="aterrizaje")] -= recorte_ventana

            offset_vuelo = recorta_ventana_analisis(daReturn, vuelo).mean(dim="time")
            # recorta_ventana_analisis(daDatos, vuelo).sel(axis='x').plot.line(x='time', col='ID', col_wrap=4)
            # offset_vuelo.sel(axis='z').plot.line(col='ID', col_wrap=4, hue='repe')
            # daDatos -= offset_vuelo
        daReturn = daReturn - offset_vuelo
        # daDatos = daDatos - offset_vuelo

        if show:
            if "plat" in daDatos.dims:
                if "repe" in daDatos.dims:
                    offset_vuelo.plot.line(
                        col="ID", col_wrap=4, hue="axis", sharey=False
                    )
            else:
                recorta_ventana_analisis(
                    daReturn, vuelo.sel(event=["despegue", "aterrizaje"])
                ).plot.line(col="ID", col_wrap=4, hue="repe", sharey=False)

    elif tipo_calculo == "umbral":
        # Hace media de valores por debajo del umbral. Si es DJ2P considera los dos tramos en vuelo

        offset = daReturn.where(daReturn < umbral).mean("time")

        if show:
            if "plat" in daDatos.dims:
                if "repe" in daDatos.dims:
                    daReturn.stack(ID_repe=("ID", "repe")).where(
                        daReturn.stack(ID_repe=("ID", "repe")) < umbral * 2.5
                    ).plot.line(
                        x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                    )
            else:
                daReturn.where(daReturn < umbral * 2.5).plot.line(
                    x="time", col="ID", col_wrap=4
                )

        daReturn = daReturn - offset

    elif tipo_calculo == "min":
        # Hace media de valores por debajo del umbral. Si es DJ2P considera los dos tramos en vuelo

        minim = daReturn.min("time")
        # minim.sel(ID='S06_DJ_30', repe=2)
        offset = minim + umbral
        # offset.sel(ID='S06_DJ_30', repe=2)
        resta = daReturn.where(daReturn < offset).mean("time")
        # resta.sel(ID='S06_DJ_30', repe=2)
        # daReturn.sel(ID='S06_DJ_30', repe=2).plot.line(x='time')
        daReturn = daReturn - resta

        if show:
            if "plat" in daDatos.dims:
                if "repe" in daDatos.dims:
                    daReturn.stack(ID_repe=("ID", "repe")).where(
                        daReturn.stack(ID_repe=("ID", "repe"))
                        < offset.stack(ID_repe=("ID", "repe"))
                    ).plot.line(
                        x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                    )
                    # Sospechosos
                    sospechosos = daReturn.stack(ID_repe=("ID", "repe", "plat")).where(
                        np.abs(resta.stack(ID_repe=("ID", "repe", "plat"))) > umbral,
                        drop=True,
                    )
                    sospechosos.plot.line(
                        x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                    )
                    print(f"Sospechosos {sospechosos.ID_repe.data}")
            else:
                if "repe" in daDatos.dims:
                    daReturn.stack(ID_repe=("ID", "repe")).where(
                        daReturn.stack(ID_repe=("ID", "repe"))
                        < offset.stack(ID_repe=("ID", "repe"))
                    ).plot.line(x="time", col="ID", col_wrap=4)
                else:
                    daReturn.where(daReturn < umbral * 2.5).plot.line(
                        x="time", col="ID", col_wrap=4
                    )

        daReturn = daReturn - resta

    if "axis" in daDatos.dims:
        daDatos.loc[dict(axis="z")] = daReturn
    else:
        daDatos = daReturn

    return daDatos


def ajusta_offsetFz_vuelo_convencional(
    daDatos, tipo_test="CMJ", umbral=20.0, pcto_ventana=5, show=False
) -> xr.DataArray:
    # Ajusta buscando los vuelos concretamente
    if tipo_test == "DJ2P":
        vuelo = detecta_despegue_aterrizaje(
            daDatos, tipo_test, umbral=umbral
        )  # , show=show)
        recorte_ventana = (
            (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
            * pcto_ventana
            / 100
        ).astype("int32")
        vuelo.loc[dict(event="despegue")] += recorte_ventana
        vuelo.loc[dict(event="aterrizaje")] -= recorte_ventana
        offset_vuelo = recorta_ventana_analisis(daDatos, vuelo).mean(dim="time")

    else:
        # busca despegue y aterrizaje provisionales
        vuelo = detecta_despegue_aterrizaje(
            daDatos, tipo_test, umbral=umbral
        )  # , show=show)
        # reduce la ventana un poco para evitar los rebotes posibles del filtrado
        recorte_ventana = (
            (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
            * pcto_ventana
            / 100
        ).astype("int32")
        vuelo.loc[dict(event="despegue")] += recorte_ventana
        vuelo.loc[dict(event="aterrizaje")] -= recorte_ventana

        offset_vuelo = recorta_ventana_analisis(daDatos, vuelo).mean(dim="time")
        # recorta_ventana_analisis(daDatos, vuelo).sel(axis='x').plot.line(x='time', col='ID', col_wrap=4)
        # offset_vuelo.sel(axis='z').plot.line(col='ID', col_wrap=4, hue='repe')
        # daDatos -= offset_vuelo
        with xr.set_options(keep_attrs=True):
            # datos = daDatos - offset_vuelo
            daDatos = daDatos - offset_vuelo

        if show:
            try:  # comprobar si es necesario cuando hay ejes
                recorta_ventana_analisis(
                    daDatos, vuelo.sel(event=["despegue", "aterrizaje"])
                ).plot.line(col="ID", col_wrap=4, hue="axis", sharey=False)
            except:
                recorta_ventana_analisis(
                    daDatos, vuelo.sel(event=["despegue", "aterrizaje"])
                ).plot.line(col="ID", col_wrap=4, hue="repe", sharey=False)

    return daDatos  # datos


def ajusta_offset_s(
    daDatos, tipo_test="DJ", umbral=20.0, pcto_ventana=5, show=False
) -> xr.DataArray:
    def _último_dato(data, ID):
        # print(ID)
        # plt.plot(data)

        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        return data[~np.isnan(data)][-1]

    """    
    data = daDatos[0,0].data
    """
    if "axis" in daDatos.dims:
        daDatos = daDatos.sel(axis="z")
    daUltimo = xr.apply_ufunc(
        _último_dato,
        daDatos,
        daDatos.ID,
        input_core_dims=[["time"], []],
        # output_core_dims=[['peso']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )
    # (daDatos - daUltimo).plot.line(x='time', col='ID')

    return daDatos - daUltimo


def reset_Fz_vuelo(
    daDatos, tipo_test=None, umbral=20.0, pcto_ventana=5, show=False
) -> xr.DataArray:  # , ventana_vuelo=None):
    """
    Si  tipo_test=None, pone a cero por debajo del umbral
    """
    # daReturn = daDatos.where(daDatos > umbral, 0.0)
    if tipo_test == "DJ2PApart":
        # Para plataforma auxiliar en DJ2Plats. Pone a cero después del despegue inicial
        vuelo = detecta_despegue_aterrizaje(
            daDatos, tipo_test, umbral=umbral
        )  # , show=show)

        def _detect_onset_aux(data, vuel, ID):
            if np.count_nonzero(~np.isnan(data)) == 0:
                return data
            # plt.plot(data)
            # plt.show()
            # print(ID, repe)
            data = data.copy()
            data[int(vuel) :] = 0.0
            return data

        """
        data = daDatos.sel(ID='S07_DJ_30', repe=1).data
        vuel = vuelo.sel(ID='S07_DJ_30', repe=1).data[0]
        """

        daReturn = xr.apply_ufunc(
            _detect_onset_aux,
            daDatos,
            vuelo.sel(event="despegue"),
            daDatos.ID,
            input_core_dims=[["time"], [], []],
            output_core_dims=[["time"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=-umbral, n_above=int(0.1*daDatos.freq), show=show)
        ).drop_vars("event")

    else:  # si no es DJ2PApart
        daReturn = daDatos.where(daDatos > umbral, 0.0)

    if show:
        if "plat" in daDatos.dims:
            if "repe" in daDatos.dims:
                daReturn.sel(axis="z").stack(ID_repe=("ID", "repe")).where(
                    daReturn.sel(axis="z").stack(ID_repe=("ID", "repe")) < umbral * 2.5
                ).plot.line(x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7)
        else:
            daReturn.where(daReturn <= umbral * 2.1).plot.line(
                x="time", col="ID", col_wrap=4
            )

    return daReturn

    """
    #Con ufunc necesario tener despegue y aterrizaje
    def reset_ventana(data, ini, fin):
        dat=data.copy()
        #print(datos.shape, ini,fin)  
        ini=int(ini)
        fin=int(fin)
        dat[ini:fin] = np.full(fin-ini, 0.0)
        return dat
    
    
    # data = daDatos[0,1].sel(axis='z').data
    # ini = daEventos[0,1].sel(event='iniMov')
    # fin = daEventos[0,1].sel(event='finMov')
    
    
    daCortado = xr.apply_ufunc(reset_ventana, daDatos, ventana_vuelo.isel(event=0).sel(ID=daDatos.ID, repe=daDatos.repe), ventana_vuelo.isel(event=1).sel(ID=daDatos.ID, repe=daDatos.repe),
                   input_core_dims=[['time'], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   ).dropna(dim='time', how='all')
    daCortado.attrs = daDatos.attrs
    daCortado.name = daDatos.name
    #daCortado.sel(axis='z').plot.line(x='time', row='ID', col='axis')
    return daCortado
    """


# TODO: SEGUIR PROBANDO ESTA FUNCIÓN PARA AJUSTAR TAMBIÉN LOS EJES X E Y
def reset_F_vuelo_ejes(
    daDatos, tipo_test, umbral=20.0, pcto_ventana=5, show=False
) -> xr.DataArray:  # , ventana_vuelo=None):
    if "axis" in daDatos.dims:
        daDatosZ = daDatos.sel(axis="z")
    else:
        daDatosZ = daDatos

    vuelo = detecta_despegue_aterrizaje(daDatosZ, tipo_test, umbral=umbral)
    # reduce la ventana un poco para evitar los rebotes posibles del filtrado
    recorte_ventana = (
        (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
        * pcto_ventana
        / 100
    ).astype("int32")
    vuelo.loc[dict(event="despegue")] += recorte_ventana
    vuelo.loc[dict(event="aterrizaje")] -= recorte_ventana

    with xr.set_options(keep_attrs=True):
        daDatos = xr.where(
            ~daDatos.isnull(), daDatos.where(daDatos > umbral, 0.0), daDatos
        )
        daDatos.time.attrs["units"] = "s"  # por alguna razón lo cambiaba a newtons
        # daDatos.plot.line(row='ID', col='repe', hue='axis', sharey=False)

    if show:
        recorta_ventana_analisis(daDatos, vuelo + [-50, 50]).plot.line(
            col="ID", col_wrap=4, hue="axis", sharey=False
        )

    return daDatos


def reset_F_vuelo_ejes_convencional(
    daDatos, tipo_test, umbral=20.0, pcto_ventana=5, show=False
) -> xr.DataArray:  # , ventana_vuelo=None):
    if "axis" in daDatos.dims:
        daDatosZ = daDatos.sel(axis="z")
    else:
        daDatosZ = daDatos

    vuelo = detecta_despegue_aterrizaje(daDatosZ, tipo_test, umbral=umbral)
    # reduce la ventana un poco para evitar los rebotes posibles del filtrado
    recorte_ventana = (
        (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
        * pcto_ventana
        / 100
    ).astype("int32")
    vuelo.loc[dict(event="despegue")] += recorte_ventana
    vuelo.loc[dict(event="aterrizaje")] -= recorte_ventana

    with xr.set_options(keep_attrs=True):
        daDatos = xr.where(
            ~daDatos.isnull(), daDatos.where(daDatos > umbral, 0.0), daDatos
        )
        daDatos.time.attrs["units"] = "s"  # por alguna razón lo cambiaba a newtons
        # daDatos.plot.line(row='ID', col='repe', hue='axis', sharey=False)

    if show:
        recorta_ventana_analisis(daDatos, vuelo + [-50, 50]).plot.line(
            col="ID", col_wrap=4, hue="axis", sharey=False
        )

    return daDatos


def calcula_variables(daDatos, daPeso=None, daEventos=None) -> xr.Dataset:
    """
    Calcula variables relacionadas con la fuerza / tiempo: v, s, P, RFD
    daEventos: recive el evento inicial y final para el cálculo. Se puede pasar
               iniMov/finMov (para evitar derivas) o iniAnalisis/finAnalisis para
               gráficas completas variables.
    """

    daBW = daDatos / daPeso.sel(stat="media").drop_vars("stat")

    # se puede integrar directamente con ufunc, pero no deja meter parámetro initial=0 y devuelve con un instante menos
    def _integra(data, time, peso, ini, fin):
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
    ini = daEventos[2,0].sel(event='iniMov').data
    fin = daEventos[2,0].sel(event='finMov').data
    plt.plot(data[int(ini):int(fin)])
    """
    """daV = (
        xr.apply_ufunc(
            _integra,
            daDatos,
            daDatos.time,
            daPeso.sel(stat="media"),
            daEventos.isel(event=0),
            daEventos.isel(
                event=1
            ),  # eventos 0 y 1 para que sirva con reversed, se pasa iniMov y finMov en el orden adecuado
            input_core_dims=[["time"], ["time"], [], [], []],
            output_core_dims=[["time"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            join="exact",
        )
        / (daPeso.sel(stat="media") / g)
    ).drop_vars("stat")
    """
    daV = (
        daDatos.biomxr.integrate_window(
            daEventos, daOffset=daPeso.sel(stat="media"), result_return="continuous"
        )
        / (daPeso.sel(stat="media") / g)
    ).drop_vars("stat")
    # daV.isel(ID=slice(None, 8)).plot.line(x='time', col='ID', col_wrap=4)

    daS = daV.biomxr.integrate_window(daEventos, result_return="continuous")
    """daS = xr.apply_ufunc(
        _integra,
        daV,
        daDatos.time,
        0,
        daEventos.isel(event=0),
        daEventos.isel(event=1),
        input_core_dims=[["time"], ["time"], [], [], []],
        output_core_dims=[["time"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    )
    """
    # daS.isel(ID=slice(None, 8)).plot.line(x='time', col='ID', col_wrap=4)

    daP = daDatos * daV
    daRFD = daDatos.differentiate(coord="time")

    # daV.attrs['units']='m/s'
    # daS.attrs['units']='m'
    # daP.attrs['units']='W'
    # daRFD.attrs['units']='N/s'

    daBW = daBW.assign_attrs({"freq": daDatos.freq, "units": "N/kg"})
    daV = daV.assign_attrs({"freq": daDatos.freq, "units": "m/s"})
    daS = daS.assign_attrs({"freq": daDatos.freq, "units": "m"})
    daP = daP.assign_attrs({"freq": daDatos.freq, "units": "W"})
    daRFD = daRFD.assign_attrs({"freq": daDatos.freq, "units": "N/s"})

    return (
        xr.Dataset(
            {"BW": daBW, "v": daV, "s": daS, "P": daP, "RFD": daRFD}  # F normalizada
        )
        .astype(daDatos.dtype)
        .assign_attrs({"freq": daDatos.freq})
    )


def calcula_results(
    daCinet=None, dsCinem=None, daPeso=None, daResults=None, daEventos=None
) -> xr.DataArray:
    if not isinstance(daResults, xr.DataArray):
        daResults = (
            xr.full_like(daCinet.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {
                    "n_var": [
                        "tVuelo",
                        "tFaseInicioDesc",
                        "tFaseExc",
                        "tFaseConc",
                        "FzMax",
                        "FzMin",
                        "FzTransicion",
                        "vDespegue",
                        "vAterrizaje",
                        "vMax",
                        "vMin",
                        "sIniMov",
                        "sFinMov",
                        "sDespegue",
                        "sAterrizaje",
                        "sDifDespAter",
                        "sMax",
                        "sMin",
                        "hTVuelo",
                        "hVDespegue",
                        "hS",
                        "PMax",
                        "PMin",
                        "RFDMax",
                        "RFDMed",
                        "impNegDescenso",
                        "ImpPositDescenso",
                        "ImpPositAscenso",
                        "ImpNegAscenso",
                        "tFzMax",
                        "tFzMin",
                        "tFzTransicion",
                        "tVMax",
                        "tVMin",
                        "tSMax",
                        "tSMin",
                        "tPMax",
                        "tPMin",
                        "tRFDMax",
                    ]
                },
                axis=-1,
            )
        ).copy()
    daResults.name = "results"
    del daResults.attrs["freq"]
    del daResults.attrs["units"]
    if "freq_ref" in daResults.attrs:
        del daResults.attrs["freq_ref"]

    if "axis" in daCinet.dims:
        daCinet = daCinet.sel(axis="z")  # en principio solo interesa el eje z

    dsBatida = recorta_ventana_analisis(
        dsCinem[["BW", "v", "s", "P", "RFD"]],
        daEventos.sel(event=["iniMov", "despegue"]),
    )
    dsAterrizaje = recorta_ventana_analisis(
        dsCinem[["BW", "v", "s", "P", "RFD"]],
        daEventos.sel(event=["aterrizaje", "finMov"]),
    )

    # Tiempos de fase
    daResults.loc[dict(n_var="tFaseInicioDesc")] = (
        daEventos.sel(event="iniImpPos") - daEventos.sel(event="iniMov")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tFaseExc")] = (
        daEventos.sel(event="maxFlex") - daEventos.sel(event="iniImpPos")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tFaseConc")] = (
        daEventos.sel(event="despegue") - daEventos.sel(event="maxFlex")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tVuelo")] = (
        daEventos.sel(event="aterrizaje") - daEventos.sel(event="despegue")
    ) / dsCinem.freq

    # Fuerzas batida
    daResults.loc[dict(n_var="FzMax")] = dsBatida["BW"].max(
        dim="time"
    )  # recorta_ventana_analisis(daCinet, daEventos.sel(event=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(n_var="FzMin")] = dsBatida["BW"].min(
        dim="time"
    )  # recorta_ventana_analisis(daCinet, daEventos.sel(event=['iniMov', 'despegue'])).min(dim='time')
    daResults.loc[dict(n_var="FzTransicion")] = daCinet.sel(
        time=daEventos.sel(event="maxFlex") / dsCinem.freq, method="nearest"
    )

    # Velocidades
    daResults.loc[dict(n_var="vDespegue")] = dsCinem["v"].sel(
        time=daEventos.sel(event="despegue") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="vAterrizaje")] = dsCinem["v"].sel(
        time=daEventos.sel(event="aterrizaje") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="vMax")] = recorta_ventana_analisis(
        dsCinem["v"], daEventos.sel(event=["iniMov", "despegue"])
    ).max(dim="time")
    daResults.loc[dict(n_var="vMin")] = recorta_ventana_analisis(
        dsCinem["v"], daEventos.sel(event=["iniMov", "despegue"])
    ).min(dim="time")

    # Posiciones
    daResults.loc[dict(n_var="sIniMov")] = dsCinem["s"].sel(
        time=daEventos.sel(event="iniMov") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sFinMov")] = dsCinem["s"].sel(
        time=(daEventos.sel(event="finMov") - 1) / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sDespegue")] = dsCinem["s"].sel(
        time=daEventos.sel(event="despegue") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sAterrizaje")] = dsCinem["s"].sel(
        time=daEventos.sel(event="aterrizaje") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sDifDespAter")] = (
        daResults.loc[dict(n_var="sDespegue")]
        - daResults.loc[dict(n_var="sAterrizaje")]
    )
    daResults.loc[dict(n_var="sMax")] = recorta_ventana_analisis(
        dsCinem["s"], daEventos.sel(event=["despegue", "aterrizaje"])
    ).max(dim="time")
    daResults.loc[dict(n_var="sMin")] = recorta_ventana_analisis(
        dsCinem["s"], daEventos.sel(event=["iniMov", "despegue"])
    ).min(dim="time")

    # Altura salto
    daResults.loc[dict(n_var="hTVuelo")] = (
        g / 8 * daResults.loc[dict(n_var="tVuelo")] ** 2
    )
    daResults.loc[dict(n_var="hVDespegue")] = daResults.loc[
        dict(n_var="vDespegue")
    ] ** 2 / (2 * g)
    daResults.loc[dict(n_var="hS")] = (
        daResults.loc[dict(n_var="sMax")] - daResults.loc[dict(n_var="sDespegue")]
    )

    # Potencias
    daResults.loc[dict(n_var="PMax")] = dsBatida["P"].max(
        dim="time"
    )  # recorta_ventana_analisis(dsCinem['P'], daEventos.sel(event=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(n_var="PMin")] = dsBatida["P"].min(
        dim="time"
    )  # recorta_ventana_analisis(dsCinem['P'], daEventos.sel(event=['iniMov', 'despegue'])).min(dim='time')

    # RFD
    daResults.loc[dict(n_var="RFDMax")] = dsBatida["RFD"].max(
        dim="time"
    )  # recorta_ventana_analisis(dsCinem['RFD'], daEventos.sel(event=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(n_var="RFDMed")] = (
        daCinet.sel(
            time=daEventos.sel(event="maxFlex") / dsCinem.freq, method="nearest"
        )
        - daCinet.sel(
            time=daEventos.sel(event="minFz") / dsCinem.freq, method="nearest"
        )
    ) / ((daEventos.sel(event="maxFlex") - daEventos.sel(event="minFz")) / dsCinem.freq)

    # Impulsos. Como la fuerza viene en BW, el peso que resta es 1. Con fuerza en newtons restar daPeso.sel(stat='media').drop_vars('stat')
    daResults.loc[dict(n_var="impNegDescenso")] = integra_completo(
        daCinet - 1, daEventos=daEventos.sel(event=["iniMov", "iniImpPos"])
    )
    daResults.loc[dict(n_var="ImpPositDescenso")] = integra_completo(
        daCinet - 1, daEventos=daEventos.sel(event=["iniImpPos", "maxFlex"])
    )
    daResults.loc[dict(n_var="ImpPositAscenso")] = integra_completo(
        daCinet - 1, daEventos=daEventos.sel(event=["maxFlex", "finImpPos"])
    )
    daResults.loc[dict(n_var="ImpNegAscenso")] = integra_completo(
        daCinet - 1, daEventos=daEventos.sel(event=["finImpPos", "despegue"])
    )

    # Tiempos de eventos clave
    daResults.loc[dict(n_var="tFzMax")] = (
        dsBatida["BW"].argmax(dim="time", skipna=False) / dsCinem.freq
    )
    daResults.loc[dict(n_var="tFzMin")] = (
        dsBatida["BW"].argmin(dim="time") / dsCinem.freq
    )
    daResults.loc[dict(n_var="tFzTransicion")] = (
        daEventos.sel(event="maxFlex") - daEventos.sel(event="iniMov")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tVMax")] = dsBatida["v"].argmax(dim="time") / dsCinem.freq
    daResults.loc[dict(n_var="tVMin")] = dsBatida["v"].argmin(dim="time") / dsCinem.freq
    daResults.loc[dict(n_var="tSMax")] = dsBatida["s"].argmax(dim="time") / dsCinem.freq
    daResults.loc[dict(n_var="tSMin")] = dsBatida["s"].argmin(dim="time") / dsCinem.freq
    daResults.loc[dict(n_var="tPMax")] = dsBatida["P"].argmax(dim="time") / dsCinem.freq
    daResults.loc[dict(n_var="tPMin")] = dsBatida["P"].argmin(dim="time") / dsCinem.freq
    daResults.loc[dict(n_var="tRFDMax")] = (
        dsBatida["RFD"].argmax(dim="time") / dsCinem.freq
    )

    return daResults


def calcula_results_EMG(daEMG=None, daResults=None, daEventos=None) -> xr.DataArray:
    if not isinstance(daResults, xr.DataArray):
        if "axis" in daEventos.coords:
            daEventos = daEventos.drop_vars("axis")
        daResults = (
            xr.full_like(
                daEMG.isel(time=0).drop_vars(["time", "axis"]), np.nan
            ).expand_dims(
                {
                    "n_var": [
                        "EMGPreIniMean",
                        "EMGExcMean",
                        "EMGConcMean",
                        "EMGVueloMean",
                        "EMGPreIniInteg",
                        "EMGExcInteg",
                        "EMGConcInteg",
                        "EMGVueloInteg",
                        "EMGPreIniRMS",
                        "EMGExcRMS",
                        "EMGConcRMS",
                        "EMGVueloRMS",
                    ]
                },
                axis=-1,
            )
        ).copy()

    daResults.name = "results"
    del daResults.attrs["freq"]
    del daResults.attrs["freq_ref"]
    del daResults.attrs["units"]

    # Medias
    daResults.loc[dict(n_var="EMGPreIniMean")] = recorta_ventana_analisis(
        daDatos=daEMG, daEvents=daEventos.sel(event=["preactiv", "iniMov"])
    ).mean("time")
    daResults.loc[dict(n_var="EMGExcMean")] = recorta_ventana_analisis(
        daEMG, daEventos.sel(event=["iniMov", "maxFlex"])
    ).mean("time")
    daResults.loc[dict(n_var="EMGConcMean")] = recorta_ventana_analisis(
        daEMG, daEventos.sel(event=["maxFlex", "despegue"])
    ).mean("time")
    daResults.loc[dict(n_var="EMGVueloMean")] = recorta_ventana_analisis(
        daEMG, daEventos.sel(event=["despegue", "aterrizaje"])
    ).mean("time")

    # Integrales
    daResults.loc[dict(n_var="EMGPreIniInteg")] = integra_completo(
        daEMG, daEventos=daEventos.sel(event=["preactiv", "iniMov"])
    )
    daResults.loc[dict(n_var="EMGExcInteg")] = integra_completo(
        daEMG, daEventos=daEventos.sel(event=["iniMov", "maxFlex"])
    )
    daResults.loc[dict(n_var="EMGConcInteg")] = integra_completo(
        daEMG, daEventos=daEventos.sel(event=["maxFlex", "despegue"])
    )
    daResults.loc[dict(n_var="EMGVueloInteg")] = integra_completo(
        daEMG, daEventos=daEventos.sel(event=["despegue", "aterrizaje"])
    )

    # RMS
    daResults.loc[dict(n_var="EMGPreIniRMS")] = RMS(
        daDatos=daEMG, ventana=daEventos.sel(event=["preactiv", "iniMov"])
    )
    daResults.loc[dict(n_var="EMGExcRMS")] = RMS(
        daDatos=daEMG, ventana=daEventos.sel(event=["iniMov", "maxFlex"])
    )
    daResults.loc[dict(n_var="EMGConcRMS")] = RMS(
        daEMG, ventana=daEventos.sel(event=["maxFlex", "despegue"])
    )
    daResults.loc[dict(n_var="EMGVueloRMS")] = RMS(
        daEMG, ventana=daEventos.sel(event=["despegue", "aterrizaje"])
    )

    return daResults


# =============================================================================
# PRUEBA METIDO EN UNA CLASE
"""
Clase con funciones para tratar fuerzas de saltos desde archivos de plataforma
de fuerzas.
"""


class trata_fuerzas_saltos:
    def __init__(
        self,
        data: Optional[xr.DataArray] = xr.DataArray(),
        tipo_test: Optional[str] = "CMJ",
        events: Optional[None] = None,
    ):  # funciona Optional[None]???
        self.data = data
        self.tipo_test = tipo_test
        self.events = (
            xr.full_like(self.data.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": eventos_basicos}, axis=-1
            )
        ).copy()
        self.peso = None

    def carga_preprocesados(self, ruta_trabajo, nomArchivoPreprocesado):
        if Path((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix(".nc")).is_file():
            tpo = time.time()
            self.datos = xr.load_dataarray(
                (ruta_trabajo / (nomArchivoPreprocesado)).with_suffix(".nc")
            ).sel(tipo=self.tipo_test)
            print(
                "\nCargado archivo preprocesado ",
                nomArchivoPreprocesado
                + "_Vicon.nc en {0:.3f} s.".format(time.time() - tpo),
            )
        else:
            raise Exception("No se encuentra el archivo Vicon preprocesado")

    def calcula_peso(self, ventana=[100, 600], show=False):
        self.peso = (
            self.datos.sel(axis="z")
            .isel(time=slice(ventana[0], ventana[1]))
            .mean(dim="time")
        )

        if show:

            def dibuja_peso(x, y, **kwargs):  # de momento no funciona
                print(x)  # kwargs['data'])
                # plt.plot()

            g = self.datos.sel(axis="z").plot.line(
                col="ID", col_wrap=4, hue="trial", sharey=False
            )
            # g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
            # g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')

            for h, ax in enumerate(g.axes):  # extrae cada fila
                for i in range(len(ax)):  # extrae cada axis (gráfica)
                    try:
                        idn = g.data.loc[g.name_dicts[h, i]].ID
                        # print('peso=', self.peso.sel(ID=idn).data)#idn)
                        # Rango medida peso
                        # ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.freq)], g.data.time[int(ventana[1]*self.datos.freq)], alpha=0.2, color='C1')
                        ax[i].axvspan(
                            (len(self.datos.time) + ventana[0]) / self.datos.freq,
                            (len(self.datos.time) + ventana[1]) / self.datos.freq,
                            alpha=0.2,
                            color="C1",
                        )
                        # Líneas peso
                        ax[i].hlines(
                            self.peso.sel(ID=idn).data,
                            xmin=self.datos.time[0],
                            xmax=self.datos.time[-1],
                            colors=["C0", "C1", "C2"],
                            lw=1,
                            ls="--",
                            alpha=0.6,
                        )
                    except:
                        print("No va el", h, i)


# =============================================================================


# =============================================================================
# %% PRUEBAS
# =============================================================================
if __name__ == "__main__":
    if False:
        import numpy as np
        import pandas as pd
        import xarray as xr

        from pathlib import Path

        import sys

        sys.path.append(r"F:\Programacion\Python\Mios\Functions")
        from filtrar_Butter import filtrar_Butter

        # ----Pruebas carga archivos
        ruta = Path(
            r"F:\Investigacion\Proyectos\Saltos\2023PreactivacionSJ\DataCollection\S01\FeedbackFuerza"
        )
        file = ruta / "S01_CMJ_000.txt"
        fuerzas_pl = carga_bioware_pl(file)

        # Con C3D
        import sys

        sys.path.append(r"F:\Programacion\Python\Mios\Functions")
        import read_kistler_c3d as rkc3d

        da = rkc3d.read_kistler_c3d_xr(file)
        da = rkc3d.split_plataforms(da)
        da = rkc3d.compute_forces_axes(da)

        daFuerzas = load_merge_bioware_pl(
            ruta,
            n_estudio="preacSJ23",
            n_vars_load=[
                "abs time (s)",
                "Fx",
                "Fy",
                "Fz",
                "Fz1",
                "Fz2",
                "Fz3",
                "Fz4",
                "Ax",
                "Ay",
            ],
        )
        daFuerzasrepe = separa_dim_repe(daFuerzas)
        daFuerzasrepeaxis = separa_dim_axis(daFuerzasrepe.sel(n_var=["Fx", "Fy", "Fz"]))

        daFuerzas = load_merge_bioware_pl(ruta, n_estudio="preacSJ23")
        daFuerzasaxis = separa_dim_repe(daFuerzas)

        daFuerzas = load_merge_bioware_pl(ruta, n_estudio="preacSJ23")
        daFuerzasaxis = separa_dim_plats(daFuerzas, merge_2_plats=1)

        ##

        # =============================================================================
        # %% CMJ del máster trimmed
        # =============================================================================

        # ----Pruebas tratamientos
        xr.set_options(keep_attrs=True)
        visual_bloque_particip = slice(
            None
        )  # [16,17] #nº de participantes para generar gráficas de visualización. EMPIEZA EN CERO

        nom_archivo_preprocesado = Path(
            r"F:/Investigacion/Proyectos/Saltos/MasterPracticas/data-science-template-main_SaltosMasterPracticas/data/processed/SaltosMasterPrac_CMJ_Trimmed.nc"
        )

        sys.path.append(r"F:/Investigacion/Proyectos/Saltos/MasterPracticas")
        daCMJ = xr.load_dataarray(nom_archivo_preprocesado)

        # Filtra
        daCMJ = filtrar_Butter(dat_orig=daCMJ, fr=daCMJ.freq, fc=400)

        # Ajuste offset vuelo, incluso antes del peso---------------------------
        daCMJ = ajusta_offsetFz(
            daDatos=daCMJ,
            tipo_test="CMJ",
            tipo_calculo="vuelo",
            umbral=40,
            pcto_ventana=5,
        )  # , show=True)

        # Sustitución ventana vuelo por cero-------------
        daCMJ = reset_Fz_vuelo(
            daDatos=daCMJ, tipo_test="CMJ", umbral=30, pcto_ventana=5
        )  # , show=True)

        daEventosForces = (
            xr.full_like(daCMJ.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": eventos_basicos}, axis=-1
            )
        ).copy()
        # Estima el ajuste del inicio y final del análisis
        daEventosForces.loc[dict(event=["iniAnalisis", "finAnalisis"])] = (
            estima_inifin_analisis(
                daDatos=daCMJ,
                tipo_test="CMJ",
                daEventos=daEventosForces.sel(event=["iniAnalisis", "finAnalisis"]),
                ventana=[1.5, 1.2],
                umbral=30,
            )
        )
        # Ajuste personalizado inicio-fin análisis
        # daEventosForces.loc[dict(ID='S06_CMJ_2', repe=0, event='iniAnalisis')] = daEventosForces.loc[dict(ID='S06_CMJ_2', repe=0, event='iniAnalisis')] - np.array(0.3) * daCMJ.freq

        # ----Cálculo peso
        # Ventanas donde calcular el peso en cada salto
        daEventosForces.loc[dict(event=["iniPeso", "finPeso"])] = (
            np.array([0.0, 0.5]) * daCMJ.freq
        )
        # daVentanasPeso = daEventosForces.loc[dict(event=['iniPeso', 'finPeso'])]
        """Comprobación
        visual_bloque_particip = slice(10, 20)
        graficas_eventos(daDatos=daCMJ.isel(ID=visual_bloque_particip), daEventos=daEventosForces, show_in_console=True)
        """

        # Ajuste personalizado casos con inicio incorrecto para medir peso
        # daEventosForces.loc[dict(ID='S04_CMJ_2', repe=[1,2], event=['iniPeso', 'finPeso'])] = np.array([0.1, 1.1]) * daCMJ.freq

        # Primero calcula el peso de la media de la zona estable seleccionada
        daPeso_media = calcula_peso(
            daDatos=daCMJ,
            ventana_peso=daEventosForces.sel(event=["iniPeso", "finPeso"]),
        )  # , show=True)
        # Para afinar el peso, primero detecta eventos provisionales
        daEventosForces = detecta_eventos_estandar(
            daDatos=daCMJ,
            daEventos=daEventosForces,
            daPeso=daPeso_media,
            tipo_test="CMJ",
            umbral=30.0,
        )
        daPesoCMJ = afina_peso(
            daDatos=daCMJ,
            daPeso=daPeso_media,
            daEventos=daEventosForces,
            tipo_calculo="iter",
        )  # , show=True)

        """
        #Gráfica para ver la dispersión de los pesos
        daPesoCMJ.sel(stat='media').assign_coords(ID=np.arange(len(daPesoCMJ.ID))).plot.line(x='ID', marker='o')
        daPesoCMJ.sel(stat='resid').assign_coords(ID=np.arange(len(daPesoCMJ.ID))).plot.line(x='ID', marker='o')
        """

        # =============================================================================
        # %%
        # =============================================================================
        ruta_trabajo = Path(r"F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros")
        nom_archivo_preprocesado = "PotenciaDJ_Preprocesado"

        daCMJ = carga_preprocesados(
            ruta_trabajo, nom_archivo_preprocesado, tipo_test="CMJ"
        )

        ##################
        # DJ
        ##################
        daDJ = carga_preprocesados(
            ruta_trabajo, nom_archivo_preprocesado, tipo_test="DJ"
        )

        daVentanasPeso = (
            xr.DataArray(
                data=[6500, 7000], coords=[["ini", "fin"]], dims=("ventana")
            ).expand_dims({"ID": daDJ.coords["ID"], "repe": daDJ.coords["repe"]})
        ).copy()
        daPeso = calcula_peso(daDJ, ventana_peso=daVentanasPeso, show=True)
        daVentanasPeso.loc[dict(ID="13", repe=2)] = [6000, 6500]

        # Ajustes de peso puntuales

        daDJ_norm = daDJ / daPeso
        daDJ_norm.sel(axis="z").plot.line(x="time", col="ID", col_wrap=4)

    """
    # =============================================================================
    # PRUEBAS COMO CLASE
    # =============================================================================
    ruta_trabajo = Path(r'F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros')
    nomArchivoPreprocesado = 'PotenciaDJ_Preprocesado'
    
    dj = trata_fuerzas_saltos(tipo_test='DJ')
    
    dj.carga_preprocesados(ruta_trabajo, nomArchivoPreprocesado)
    dj.data
    dj.tipo_test
    
    dj.calcula_peso(ventana=[-1500, -1000], show=True)
    dj.peso.sel(ID='01', trial='1')
    """
