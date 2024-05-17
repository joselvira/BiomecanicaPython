# -*- coding: utf-8 -*-
"""
Created on Fry Mar 8 13:15:58 2024

@author: jose.lopeze

Lectura de archivos .csv exportados de Noraxon con IMU.
"""
# =============================================================================
# %% Carga librerías
# =============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import polars as pl

import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages #para guardar gráficas en pdf
# import seaborn as sns

from pathlib import Path
import time  # para cuantificar tiempos de procesado


# import sys
# sys.path.append('F:\Programacion\Python\Mios\Functions')
# #sys.path.append('G:\Mi unidad\Programacion\Python\Mios\Functions')

# from readViconCsv import read_vicon_csv


__author__ = "Jose Luis Lopez Elvira"
__version__ = "0.0.1"
__date__ = "08/03/2024"

"""
Modificaciones:    
    08/03/2024, v0.0.1
            - Empezado tomando trozos sueltos.
            
"""

# =============================================================================
# %% Carga archivos iSen
# =============================================================================


def separa_dim_axis(da):
    # Separa el xarray en ejes creando dimensión axis
    if "Accel " in da.n_var.to_series().iloc[0]:
        sensor_type = "Accel "
    elif "Gyro " in da.n_var.to_series().iloc[0]:
        sensor_type = "Gyro "
    elif "Mag " in da.n_var.to_series().iloc[0]:
        sensor_type = "Mag "
    else:
        raise Exception('Sensor type not found, try "Accel", "Gyro" or "Mag"')

    x = da.sel(
        n_var=da.n_var.str.endswith(f"{sensor_type[0]}x")
    )  # .rename({"n_var": "axis"})
    x = x.assign_coords(
        n_var=[
            s[:-3] for s in x.n_var.to_series()
        ]  # [s.split(sensor_type)[1][:-3] for s in x.n_var.to_series()]
    )

    y = da.sel(
        n_var=da.n_var.str.endswith(f"{sensor_type[0]}y")
    )  # .rename({"n_var": "axis"})
    y = y.assign_coords(n_var=[s[:-3] for s in y.n_var.to_series()])

    z = da.sel(
        n_var=da.n_var.str.endswith(f"{sensor_type[0]}z")
    )  # .rename({"n_var": "axis"})
    z = z.assign_coords(n_var=[s[:-3] for s in z.n_var.to_series()])

    da = (
        xr.concat([x, y, z], dim="axis")
        # .assign_coords(n_var="plat1")
        .assign_coords(axis=["x", "y", "z"])
        # .expand_dims({"n_var": 1})
    )

    return da


def separa_dim_lado(da):
    # Separa el xarray en ejes creando dimensión axis

    L = da.sel(n_var=da.n_var.str.contains("izquierda"))
    L = L.assign_coords(n_var=[s.split(" izquierda")[0] for s in L.n_var.to_series()])

    R = da.sel(n_var=da.n_var.str.contains("derecha"))  # .rename({"n_var": "axis"})
    R = R.assign_coords(n_var=[s.split(" derecha")[0] for s in R.n_var.to_series()])

    da = (
        xr.concat([L, R], dim="side")
        # .assign_coords(n_var="plat1")
        .assign_coords(side=["L", "R"])
        # .expand_dims({"n_var": 1})
    )

    return da


def asigna_subcategorias_xr(da, n_estudio=None):

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


def pasa_df_a_da(dfTodos, n_estudio=None, show=False):
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
        dfpd = dfTodos.drop(
            columns=["estudio", "tipo", "subtipo", "particip", "repe"]
        ).melt(id_vars=["ID", "time"], var_name="n_var")

    daTodos = (
        # dfpd.drop(columns=["estudio", "tipo", "subtipo", "particip", "repe"])
        dfpd  # .melt(id_vars=["ID", "time"], var_name="n_var")
        # pd.melt(dfTodosArchivos.to_pandas().drop(columns=['estudio','tipo','subtipo']), id_vars=['ID', 'repe', 'time'], var_name='axis')
        .set_index(["ID", "n_var", "time"])
        .to_xarray()
        .to_array()
        .squeeze("variable")
        .drop_vars("variable")
    )

    # Asigna coordenadas extra
    daTodos = asigna_subcategorias_xr(da=daTodos, n_estudio=n_estudio)
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

    # daTodos.attrs = {
    #     "freq": (np.round(1 / (daTodos.time[1] - daTodos.time[0]), 1)).data,
    #     "units": "N",
    # }
    daTodos.attrs["freq"] = (np.round(1 / (daTodos.time[1] - daTodos.time[0]), 1)).data
    daTodos.time.attrs["units"] = "s"

    return daTodos


def read_noraxon_pd(file, n_vars_load=None, to_dataarray=False):
    # file = Path(r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")
    df = pd.read_csv(
        file, skiprows=3, header=0, engine="c"
    ).drop(  # .astype(np.float64)
        columns=["Activity", "Marker"]
    )

    # df = df.drop(columns=df.filter(regex="(Normal)") + df.filter(regex="(Tiempo.)"))

    #       df.filter(regex="(Normal)").values)).drop(columns=df.filter(regex="(Tiempo.)"))
    # df.filter(regex="(Normal)")+df.filter(regex="(Tiempo.)")
    # df.dropna(axis="columns", how="all")

    # df.filter(regex="Tiempo.")

    if to_dataarray:
        da = xr.DataArray()
        return da

    return df


def read_noraxon_pl(file, n_vars_load=None, to_dataarray=False):
    # file = Path(r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")

    df = (
        pl.read_csv(
            file,
            has_header=True,
            skip_rows=3,
            # skip_rows_after_header=0,
            columns=n_vars_load,
            # separator=",",
        )
        # .select(pl.exclude("^.*_duplicated_.*$"))  # quita columnas de tiempo duplicadas
        .select(pl.exclude(pl.String))  # quita columnas de texto con datos (Normal)
        # .with_columns(pl.all().cast(pl.Float64()))
    )

    """
    cadera = df.select(
            pl.col("^*Flexo-extensión cadera .*::Y$")
        )  # .to_numpy()
    """
    # df = df.to_pandas()

    # ----Transform polars to xarray
    if to_dataarray:

        # # Separa ejes articulares
        # x = df.select(pl.col("^Flexo-extensión.*$")).to_numpy()
        # y = df.select(pl.col("^Aducción-abducción.*$")).to_numpy()
        # z = df.select(pl.col("^Rotación.*$")).to_numpy()
        # data = np.stack([x, y, z])

        freq = 1 / (df[1, "time"] - df[0, "time"])

        # coords = {
        #     "axis": ["x", "y", "z"],
        #     "time": np.arange(data.shape[1]) / freq,
        #     "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        # }
        coords = {
            "time": np.arange(df.shape[0]) / freq,
            "n_var": df.columns,  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=df.to_numpy(),
                dims=coords.keys(),
                coords=coords,
            )
            # .astype(float)
            # .transpose("n_var", "time")
        )
        da.name = "EMG"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "mV"

        return da

    return df


def load_merge_noraxon_csv(
    ruta,
    section="EMG",
    n_estudio=None,
    data_type=None,
    show=False,
):
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
        list(ruta.glob("*.csv"))  # "**/*.csv"
    )  #'**/*.txt' incluye los que haya en subcarpetas
    lista_archivos = [
        x
        for x in lista_archivos
        if "error" not in x.name and "info" not in x.name and "_error" not in x.name
    ]  # selecciona archivos

    todas_vars = [
        "Ultium EMG.EMG 1 (uV)",
        "Ultium EMG.Internal Accel 1 Ax (mG)",
        "Ultium EMG.Internal Accel 1 Ay (mG)",
        "Ultium EMG.Internal Accel 1 Az (mG)",
        "Ultium EMG.Internal Gyro 1 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 1 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 1 Gz (deg/s)",
        "Ultium EMG.Internal Mag 1 Mx (mGauss)",
        "Ultium EMG.Internal Mag 1 My (mGauss)",
        "Ultium EMG.Internal Mag 1 Mz (mGauss)",
        "Ultium EMG.EMG 2 (uV)",
        "Ultium EMG.Internal Accel 2 Ax (mG)",
        "Ultium EMG.Internal Accel 2 Ay (mG)",
        "Ultium EMG.Internal Accel 2 Az (mG)",
        "Ultium EMG.Internal Gyro 2 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 2 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 2 Gz (deg/s)",
        "Ultium EMG.Internal Mag 2 Mx (mGauss)",
        "Ultium EMG.Internal Mag 2 My (mGauss)",
        "Ultium EMG.Internal Mag 2 Mz (mGauss)",
        "Ultium EMG.EMG 3 (uV)",
        "Ultium EMG.Internal Accel 3 Ax (mG)",
        "Ultium EMG.Internal Accel 3 Ay (mG)",
        "Ultium EMG.Internal Accel 3 Az (mG)",
        "Ultium EMG.Internal Gyro 3 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 3 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 3 Gz (deg/s)",
        "Ultium EMG.Internal Mag 3 Mx (mGauss)",
        "Ultium EMG.Internal Mag 3 My (mGauss)",
        "Ultium EMG.Internal Mag 3 Mz (mGauss)",
        "Ultium EMG.EMG 4 (uV)",
        "Ultium EMG.Internal Accel 4 Ax (mG)",
        "Ultium EMG.Internal Accel 4 Ay (mG)",
        "Ultium EMG.Internal Accel 4 Az (mG)",
        "Ultium EMG.Internal Gyro 4 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 4 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 4 Gz (deg/s)",
        "Ultium EMG.Internal Mag 4 Mx (mGauss)",
        "Ultium EMG.Internal Mag 4 My (mGauss)",
        "Ultium EMG.Internal Mag 4 Mz (mGauss)",
        "Ultium EMG.EMG 5 (uV)",
        "Ultium EMG.Internal Accel 5 Ax (mG)",
        "Ultium EMG.Internal Accel 5 Ay (mG)",
        "Ultium EMG.Internal Accel 5 Az (mG)",
        "Ultium EMG.Internal Gyro 5 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 5 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 5 Gz (deg/s)",
        "Ultium EMG.Internal Mag 5 Mx (mGauss)",
        "Ultium EMG.Internal Mag 5 My (mGauss)",
        "Ultium EMG.Internal Mag 5 Mz (mGauss)",
        "Ultium EMG.EMG 6 (uV)",
        "Ultium EMG.Internal Accel 6 Ax (mG)",
        "Ultium EMG.Internal Accel 6 Ay (mG)",
        "Ultium EMG.Internal Accel 6 Az (mG)",
        "Ultium EMG.Internal Gyro 6 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 6 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 6 Gz (deg/s)",
        "Ultium EMG.Internal Mag 6 Mx (mGauss)",
        "Ultium EMG.Internal Mag 6 My (mGauss)",
        "Ultium EMG.Internal Mag 6 Mz (mGauss)",
        "Ultium EMG.Internal Accel 7 Ax (mG)",
        "Ultium EMG.Internal Accel 7 Ay (mG)",
        "Ultium EMG.Internal Accel 7 Az (mG)",
        "Ultium EMG.Internal Gyro 7 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 7 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 7 Gz (deg/s)",
        "Ultium EMG.Internal Mag 7 Mx (mGauss)",
        "Ultium EMG.Internal Mag 7 My (mGauss)",
        "Ultium EMG.Internal Mag 7 Mz (mGauss)",
        "Ultium EMG.Internal Accel 8 Ax (mG)",
        "Ultium EMG.Internal Accel 8 Ay (mG)",
        "Ultium EMG.Internal Accel 8 Az (mG)",
        "Ultium EMG.Internal Gyro 8 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 8 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 8 Gz (deg/s)",
        "Ultium EMG.Internal Mag 8 Mx (mGauss)",
        "Ultium EMG.Internal Mag 8 My (mGauss)",
        "Ultium EMG.Internal Mag 8 Mz (mGauss)",
        "Ultium EMG.Internal Accel 9 Ax (mG)",
        "Ultium EMG.Internal Accel 9 Ay (mG)",
        "Ultium EMG.Internal Accel 9 Az (mG)",
        "Ultium EMG.Internal Gyro 9 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 9 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 9 Gz (deg/s)",
        "Ultium EMG.Internal Mag 9 Mx (mGauss)",
        "Ultium EMG.Internal Mag 9 My (mGauss)",
        "Ultium EMG.Internal Mag 9 Mz (mGauss)",
        "Ultium EMG.Internal Accel 10 Ax (mG)",
        "Ultium EMG.Internal Accel 10 Ay (mG)",
        "Ultium EMG.Internal Accel 10 Az (mG)",
        "Ultium EMG.Internal Gyro 10 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 10 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 10 Gz (deg/s)",
        "Ultium EMG.Internal Mag 10 Mx (mGauss)",
        "Ultium EMG.Internal Mag 10 My (mGauss)",
        "Ultium EMG.Internal Mag 10 Mz (mGauss)",
    ]

    if section == "EMG":
        sect = "EMG "
    else:
        sect = section

    n_vars_load = ["time"] + [s for s in todas_vars if sect in s]

    # file = Path("F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")

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
        print(f"Cargando archivo nº {nf+1}/{len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = read_noraxon_pl(file, n_vars_load)
            n_file = file.stem.split("_")[
                1:
            ]  # para quitar lo primero que pone, que es la fecha
            if len(n_file) == 5:
                n_estudio = n_file[0] if n_estudio is None else n_estudio
                particip = n_file[-4]
                tipo = n_file[-3]
                subtipo = n_file[-2]
            elif len(n_file) == 4:
                n_estudio = n_file[0]
                particip = n_file[1]
                tipo = n_file[2]
                subtipo = "X"
            elif len(n_file) == 3:
                particip = n_file[0]
                tipo = n_file[-2]
                subtipo = "X"
            if n_estudio is None:
                n_estudio = "EstudioX"

            repe = str(int(n_file[-1]))  # int(file.stem.split('.')[0][-1]
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

    # Renombra archivos
    n_vars_load2 = [s.split(".")[-1] for s in n_vars_load[1:] if sect in s]

    # Renombra columnas
    # if section == 'EMG':
    #     renombra = [s.split('Internal ')[-1][:-5] for s in n_vars_load2]
    # elif section == 'Accel':
    #     renombra = [s.split('Internal ')[-1][:-5] for s in n_vars_load2]
    # elif section == 'Gyro':
    renombra = [s.split("Internal ")[-1].split(" (")[0] for s in n_vars_load2]

    dfTodos = dfTodos.rename(
        dict(zip(dfTodos.columns[1 : len(renombra) + 1], renombra))
    )

    daTodos = pasa_df_a_da(dfTodos, n_estudio=n_estudio)
    daTodos.name = section
    daTodos.attrs["units"] = n_vars_load2[0].split("(")[-1].split(")")[0]

    # daTodos = separa_dim_axis(daTodos)

    # daTodos = separa_dim_lado(daTodos)

    return daTodos


def load_merge_iSen_sensores_csv(
    ruta,
    n_vars_load=None,
    n_estudio=None,
    data_type=None,
    show=False,
):
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
        list(ruta.glob("*.csv"))  # "**/*.csv"
    )  #'**/*.txt' incluye los que haya en subcarpetas
    lista_archivos = [
        x
        for x in lista_archivos
        if "error" not in x.name and "aceleración local" in x.name
    ]  # selecciona archivos

    # file = Path("F:/Investigacion/Proyectos/Tesis/TesisCoralPodologa/Registros/PRUEBASiSEN/ANGULOS.csv")

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
        print(f"Cargando archivo nº {nf+1}/{len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = read_isen_pl(file, n_vars_load)

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

    dfTodos = dfTodos.rename({"Tiempo": "time"})  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daTodos = pasa_df_a_da(dfTodos, n_estudio=n_estudio)

    return daTodos


def load_merge_iSen_csvxxxx(ruta):

    print("\nCargando los archivos iSen...")
    timerCarga = time.time()  # inicia el contador de tiempo
    # nomVarsACargar = nomVarsContinuas#+nomVarsDiscretas

    # Selecciona los archivos en la carpeta
    lista_archivos = sorted(
        ruta.glob("*.csv")  # "**/*.csv"
    )  # list((ruta).glob('**/*.csv'))#incluye los que haya en subcarpetas
    lista_archivos = [
        x for x in lista_archivos if "error" not in x.name and "SENSORES" not in x.name
    ]  # selecciona archivos

    print("\nCargando los archivos...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    numArchivosProcesados = 0
    dfTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error

    for nf, file in enumerate(lista_archivos[:]):
        print(f"Cargando archivo nº {nf}/{len(lista_archivos)}: {file.name}")
        """
        #Selecciona archivos según potencia
        if file.parent.parts[-1] not in ['2W', '6W']: #si no es de marcha se lo salta
            #print('{} archivo estático'.format(file.stem))
            continue
        """
        # # Selecciona archivos según variable
        # if file.stem not in [
        #     r"Flexo-extensión caderas",
        #     r"Flexo-extensión rodillas",
        #     r"Dorsiflexión-plantarflexión tobillos",
        # ]:  # si no es de marcha se lo salta
        #     # print('{} archivo estático'.format(file.stem))
        #     continue
        # print(file)

        try:
            timerSub = time.time()  # inicia el contador de tiempo
            dfprovis = pd.read_csv(
                file, sep=","
            )  # .filter(regex="::Y")  # se queda solo con las columnas de datos, descarta las de tiempos

            # Ajusta los nombres R y L
            dfprovis.rename(
                columns={
                    str(dfprovis.filter(regex="derech").columns.values)[2:-2]: "R",
                    str(dfprovis.filter(regex="izquierd").columns.values)[2:-2]: "L",
                },
                inplace=True,
            )
            # dfprovis.columns=['R', 'L']

            freq = (
                1 / dfprovis["Tiempo"].diff().mean()
            )  # (dfprovis["Tiempo"][1] - dfprovis["Tiempo"][0])

            # Añade columna tiempo
            t = np.arange(0, len(dfprovis) / freq)[
                : len(dfprovis)
            ]  # ajuste al cargar el tiempo porque en algunos t sale con un dato de más

            nom = file.parent.parts[-2]
            pot = file.parent.parts[-1]
            artic = file.stem.split()[-1][:-1]
            if "flex" in file.stem.lower():
                eje = "x"

            """
            if file.stem == 'Dorsiflexión-plantarflexión tobillos':
                artic = 'tobillo'
                eje='x'
            elif file.stem == 'Flexo-extensión rodillas':
                artic = 'rodilla'
                eje='x'
            elif file.stem == 'Flexo-extensión caderas':
                artic = 'cadera'
                eje='x'
            """

            # Añade etiquetas
            dfprovis = dfprovis.assign(
                **{
                    "nombre": nom,
                    "potencia": pot,
                    "articulacion": artic,
                    "eje": eje,
                    "time": t,
                }
            ).reset_index(drop=True)

            # Para formato tidy
            dfprovis = dfprovis.reindex(
                columns=["nombre", "potencia", "articulacion", "eje", "time", "R", "L"]
            )

            # Transforma a formato long, mejor dejarlo en tidy para que ocupe menos al guardarlo
            # dfprovis = pd.melt(dfprovis, id_vars=['nombre', 'potencia', 'articulacion', 'eje', 'time'], var_name='lado')#, value_vars=dfprovis[pot].iloc[:, :-4])
            # dfprovis = dfprovis.reindex(columns=['nombre', 'potencia', 'articulacion', 'eje', 'lado', 'time', 'value'])

            dfTodosArchivos.append(dfprovis)

            print(
                "{0} filas y {1} columnas".format(
                    dfTodosArchivos[-1].shape[0], dfTodosArchivos[-1].shape[1]
                )
            )
            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print(
                "\nATENCIÓN. No se ha podido procesar {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            ErroresArchivos.append(file.parent.name + " " + file.name + " " + str(err))
            continue
    dfTodosArchivos = pd.concat(dfTodosArchivos)
    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timerCarga
        )
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    # =============================================================================
    # Lo pasa a DataArray
    # =============================================================================
    # Transforma a formato long
    dfTodosMulti = (
        pd.melt(
            dfTodosArchivos,
            id_vars=["nombre", "potencia", "articulacion", "eje", "time"],
            var_name="lado",
        )
        .reindex(
            columns=[
                "nombre",
                "potencia",
                "articulacion",
                "eje",
                "lado",
                "time",
                "value",
            ]
        )
        .set_index(["nombre", "potencia", "articulacion", "eje", "lado", "time"])
    )

    try:
        daTodosSujetos = dfTodosMulti.to_xarray().to_array()
        daTodosSujetos = daTodosSujetos.sel(time=slice(0, 63))
        del daTodosSujetos["variable"]  # la quita de coordenadas
        daTodosSujetos = daTodosSujetos.squeeze("variable")  # la quita de dimensiones
        daTodosSujetos.attrs["frec"] = frec
        daTodosSujetos.attrs["units"] = "grados"
        daTodosSujetos.time.attrs["units"] = "s"

        if bGraficasComprobaciones:
            daTodosSujetos.sel(
                nombre="Javi", potencia="2W", articulacion="rodilla", eje="x"
            ).plot.line(x="time", hue="lado")
            daTodosSujetos.sel(potencia="2W", articulacion="cadera").plot.line(
                x="time", row="nombre", col="lado"
            )

    except:
        print(
            "Es posible que haya algún archivo duplicado. Buscar el archivo con duración distinta"
        )
        # Si no funciona el data array, comprueba si hay duplicados
        for n, df in dfTodosArchivos.set_index(
            list(dfTodosArchivos.columns[:-2])
        ).groupby(list(dfTodosArchivos.columns[:-2])):
            print(n, len(df))

    if bGuardaGraficasPdf:
        # Comparativa Todas las variables de side L y R juntas en una misma hoja de cada sujeto
        nompdf = ruta_trabajo / "CamparacionVicon_iSen_PorArtics_iSen.pdf"
        with PdfPages(nompdf) as pdf_pages:
            for n, gda in daTodosSujetos.sel(eje="x").groupby("nombre"):
                g = gda.plot.line(
                    x="time",
                    row="potencia",
                    col="articulacion",
                    hue="lado",
                    sharey=False,
                    aspect=1.5,
                )
                for h, ax in enumerate(g.axes):  # extrae cada fila
                    for i in range(len(ax)):  # extrae cada axis (gráfica)
                        nom = str(g.data.loc[g.name_dicts[h, i]].nombre.data)
                        pot = str(g.data.loc[g.name_dicts[h, i]].potencia.data)
                        # print(nom, pot)
                        try:
                            ax[i].axvline(
                                x=dfFrames.loc[(nom, pot), "ini"] / frec, c="r", ls="--"
                            )
                            ax[i].axvline(
                                x=dfFrames.loc[(nom, pot), "fin"] / frec, c="r", ls="--"
                            )
                        except:
                            continue

                g.fig.subplots_adjust(top=0.95)
                g.fig.suptitle(n)
                pdf_pages.savefig(g.fig)
                plt.show()

    # =============================================================================
    #   Guarda archivos cargados
    # =============================================================================
    # Guarda xarray
    tpoGuardar = time.time()
    daTodosSujetos.to_netcdf(
        (ruta_trabajo / (nomArchivoPreprocesado + "_iSen")).with_suffix(".nc")
    )
    print(
        "\nGuardado el Dataframe preprocesado {0} en {1:0.2f} s.".format(
            nomArchivoPreprocesado + "_iSen.nc", time.time() - tpoGuardar
        )
    )

    # Guarda dataframetpoGuardar = time.time()
    tpoGuardar = time.time()
    dfTodosArchivos.to_csv(
        (ruta_trabajo / (nomArchivoPreprocesado + "_iSen_tidy")).with_suffix(".csv"),
        index=False,
    )
    print(
        "\nGuardado el DataArray preprocesado {0} en {1:0.2f} s.".format(
            nomArchivoPreprocesado + "_iSen_tidy.csv", time.time() - tpoGuardar
        )
    )
    # Transforma a formato long
    # dfTodosArchivos = pd.melt(dfTodosArchivos, id_vars=['nombre', 'potencia', 'articulacion', 'eje', 'time'], var_name='lado')#, value_vars=dfprovis[pot].iloc[:, :-4])
    # dfTodosArchivos = dfTodosArchivos.reindex(columns=['nombre', 'potencia', 'articulacion', 'eje', 'lado', 'time', 'value'])


def carga_preprocesados(ruta_trabajo, nomArchivoPreprocesado):
    # CARGA VICON
    if Path(
        (ruta_trabajo / (nomArchivoPreprocesado + "_Vicon")).with_suffix(".nc")
    ).is_file():
        tpo = time.time()
        ds_Vicon = xr.load_dataset(
            (ruta_trabajo / (nomArchivoPreprocesado + "_Vicon")).with_suffix(".nc")
        )
        # daTodosArchivos = xr.load_dataarray((ruta_trabajo / (nomArchivoPreprocesado+'_Vicon')).with_suffix('.nc'))
        print(
            "\nCargado archivo preprocesado ",
            nomArchivoPreprocesado
            + "_Vicon.nc en {0:.3f} s.".format(time.time() - tpo),
        )
    else:
        raise Exception("No se encuentra el archivo Vicon preprocesado")

    # CARGA ISEN
    if Path(
        (ruta_trabajo / (nomArchivoPreprocesado + "_iSen")).with_suffix(".nc")
    ).is_file():
        tpo = time.time()
        da_iSen = xr.load_dataarray(
            (ruta_trabajo / (nomArchivoPreprocesado + "_iSen")).with_suffix(".nc")
        )
        print(
            "\nCargado archivo preprocesado ",
            (ruta_trabajo / (nomArchivoPreprocesado + "_iSen")).with_suffix(".nc").name,
            "en {0:.3f} s.".format(time.time() - tpo),
        )
    else:
        raise Exception("No se encuentra el archivo iSen preprocesado")

    return ds_Vicon, da_iSen


# =============================================================================
# %% Prueba las funciones
# =============================================================================
if __name__ == "__main__":
    ###################################
    # PROBAR CÁLCULOS ÁNGULOS IMUS CON PYKINEMATICS
    # pip install pykinematics
    # TAMBIÉN PROBAR CON pip install pyjamalib, conda install conda-forge::pyjamalib
    # TAMBIÉN SCIKIT-KINEMATICS http://work.thaslwanter.at/skinematics/html/
    # pip install scikit-kinematics
    ###################################
    # ruta_trabajo = Path('F:\Programacion\Python\Mios\TratamientoDatos\EjemploBikefitting')
    ruta_trabajo = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000"
    )

    # Carga EMG
    daEMG = load_merge_noraxon_csv(ruta=ruta_trabajo, section="EMG")
    # daEMG.attrs['units'] = 'uV'
    daEMG.plot.line(x="time", col="n_var", row="ID", sharey=False)

    # Carga Acelerómetro
    daAcc = load_merge_noraxon_csv(ruta=ruta_trabajo, section="Accel")
    daAcc = separa_dim_axis(da=daAcc)
    daAcc.plot.line(x="time", col="n_var", row="ID", sharey=False)

    # Carga Giroscopio
    daGyro = load_merge_noraxon_csv(ruta=ruta_trabajo, section="Gyro")
    daGyro = separa_dim_axis(da=daGyro)
    daGyro.plot.line(x="time", col="n_var", row="ID", sharey=False)
    daGyro.where(daGyro.tipo == "salto", drop=True).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4, sharey=False
    )
    daGyro.sel(ID=daGyro.ID.str.contains("salto")).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4, sharey=False
    )

    # Carga Magnetómetro
    daMag = load_merge_noraxon_csv(ruta=ruta_trabajo, section="Mag")
    daMag = separa_dim_axis(da=daMag)
    daMag.plot.line(x="time", col="n_var", row="ID", sharey=False)
    daMag.sel(ID=daMag.ID.str.contains("salto")).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4, sharey=False
    )

    """
    file = Path(r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")
    #Sigue siendo más rápido con polars que pd engine='c'
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(10):
        dfProvis = read_noraxon_pd(file)
    print(time.perf_counter()-t)
    
    
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(10):
        dfProvis = read_noraxon_pl(file, n_vars_load=['Ultium EMG.Internal Accel 1 Ax (mG)'], to_dataarray=False)
    print(time.perf_counter()-t)
    
    read_noraxon_pl(file, to_dataarray=False)
    """
