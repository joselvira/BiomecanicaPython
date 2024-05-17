# -*- coding: utf-8 -*-
"""
Created on Fry Mar 1 13:15:58 2024

@author: jose.lopeze

Lectura de archivos .csv exportados de iSen.
Basado en el usado para registros 2021 para TFM de María Aracil.
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


# =============================================================================
# %% Carga archivos iSen
# =============================================================================


def separa_dim_axis(da):
    # Separa el xarray en ejes creando dimensión axis

    x = da.sel(
        n_var=da.n_var.str.contains("Flexo-extensión")
    )  # .rename({"n_var": "axis"})
    x = x.assign_coords(
        n_var=[s.split("Flexo-extensión ")[1][:-3] for s in x.n_var.to_series()]
    )

    y = da.sel(
        n_var=da.n_var.str.contains("Abducción-aducción")
    )  # .rename({"n_var": "axis"})
    y = y.assign_coords(
        n_var=[s.split("Abducción-aducción ")[1][:-3] for s in y.n_var.to_series()]
    )

    z = da.sel(n_var=da.n_var.str.contains("Rotación"))  # .rename({"n_var": "axis"})
    z = z.assign_coords(
        n_var=[s.split("Rotación ")[1][:-3] for s in z.n_var.to_series()]
    )

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

    # Renombra columnas
    # dfTodos = dfTodos.rename({'abs time (s)':'time', 'Fx':'x', 'Fy':'y', 'Fz':'z'})

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
    daTodos.attrs = {
        "freq": (np.round(1 / (daTodos.time[1] - daTodos.time[0]), 1)).data,
        "units": "N",
    }
    daTodos.time.attrs["units"] = "s"
    daTodos.name = "Forces"

    return daTodos


def read_isen_pd(file, n_vars_load=None, to_dataarray=False):
    # file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-CaderasRodillas_00.csv')

    df = pd.read_csv(file, dtype=np.float64, engine="c")  # .astype(np.float64)
    # df = df.drop(columns=df.filter(regex="(Normal)") + df.filter(regex="(Tiempo.)"))

    #       df.filter(regex="(Normal)").values)).drop(columns=df.filter(regex="(Tiempo.)"))
    # df.filter(regex="(Normal)")+df.filter(regex="(Tiempo.)")
    # df.dropna(axis="columns", how="all")

    # df.filter(regex="Tiempo.")

    if to_dataarray:
        da = xr.DataArray()
        return da

    return df


def read_isen_pl(file, n_vars_load=None, to_dataarray=False):
    # file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-CaderasRodillas_00.csv')
    # file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-Sensores_00.csv')
    df = (
        pl.read_csv(
            file,
            # has_header=True,
            # skip_rows=0,
            # skip_rows_after_header=0,
            columns=n_vars_load,
            # separator=",",
        )  # , columns=nom_vars_cargar)
        .select(pl.exclude("^.*_duplicated_.*$"))  # quita columnas de tiempo duplicadas
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

        # Separa ejes articulares
        x = df.select(pl.col("^Flexo-extensión.*$")).to_numpy()
        y = df.select(pl.col("^Aducción-abducción.*$")).to_numpy()
        z = df.select(pl.col("^Rotación.*$")).to_numpy()
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


def load_merge_iSen_angulos_csv(
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
        if "error" not in x.name and "Sensores" not in x.name and "Actual" not in x.name
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

    daTodos = separa_dim_axis(daTodos)

    daTodos = separa_dim_lado(daTodos)

    # TODO: SEGUIR HACIENDO QUE SI CARGA VARIABLES DISTINTAS DE LAS CONVENCIONALES LAS PASE A DAARRAY PERO SIN EJES
    # if  dfTodos.columns == ['abs time (s)', 'Fx', 'Fy', 'Fz'] or n_vars_load == ['abs time (s)', 'Fx', 'Fy', 'Fz', 'Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0']:
    #     daTodos = pasa_dfpl_a_da(dfTodos, merge_2_plats=merge_2_plats, show=show)

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

            HAY QUE ADAPTAR LAS ETIQUETAS A LAS CARPETAS
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

    # Vicon
    # ruta_trabajo = Path('F:\Programacion\Python\Mios\TratamientoDatos\EjemploBikefitting')
    ruta_trabajo = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen"
    )
    # dfTodosArchivos, daTodosArchivos = carga_preprocesa_vicon(ruta=ruta_trabajo)

    # guardar_preprocesados_vicon(ruta=ruta_trabajo , nom_archivo='Prueba', df= dfTodosArchivos, da=daTodosArchivos)
    daData = load_merge_iSen_angulos_csv(ruta=ruta_trabajo)
    daData.sel(axis="x", time=slice(30, None)).plot.line(
        x="time", col="n_var", row="ID"
    )


    n_vars_load = None #['Tiempo', 'BOLTWOODITE X::Y', 'BOLTWOODITE Y::Y', 'BOLTWOODITE Z::Y',
                   'LEO X::Y', 'LEO Y::Y', 'LEO Z::Y',
                   'GASPEITE X::Y', 'GASPEITE Y::Y', 'GASPEITE Z::Y',
                   'MIARGYRITE X::Y', 'MIARGYRITE Y::Y', 'MIARGYRITE Z::Y',
                   'KARENAI X::Y', 'KARENAI Y::Y', 'KARENAI Z::Y'
    ]
    daSensors = load_merge_iSen_sensores_csv(ruta=ruta_trabajo / 'SensoresSeparado', n_vars_load=n_vars_load)
    daSensors.sel(time=slice(30, None)).isel(ID=0).plot.line(x="time", col="n_var", col_wrap=4)
    
    
    """
    file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-CaderasRodillas_00.csv')
    #Sigue siendo más rápido con polars que pd engine='c'
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(100):
        dfProvis = read_isen_pd(file)
    print(time.perf_counter()-t)
    
    
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(100):
        dfProvis = read_isen_pl(file)
    print(time.perf_counter()-t)
    """
