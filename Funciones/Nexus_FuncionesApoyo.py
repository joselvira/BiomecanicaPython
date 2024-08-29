# -*- coding: utf-8 -*-


"""
Created on Tue Jun 07 13:44:46 2022

Funciones comunes para tratar archivos del Nexus.
Se pueden utilizar desde Nexus y desde fuera.
Utiliza los csv exportados de las MVC. Calcula el máximo de cada canal de EMG
en todos los archivos con 'MVC' en su nombre. Guarda los máximos en un archivo
para no tener que volver a cargarlos.
El máximo lo utiliza para normalizar los canales EMG del archivo actual.

@author: Jose Luis López Elvira
"""

# =============================================================================
# %% INICIA
# =============================================================================


__filename__ = "Nexus_FuncionesApoyo"
__version__ = "0.4.0"
__company__ = "CIDUMH"
__date__ = "26/07/2024"
__author__ = "Jose L. L. Elvira"

"""
Modificaciones:
    26/07/2024, v0.3.0
        - Corrección importante en el cálculo de AngSegRETROPIE con modelo
          antiguo (a partir de metas).
    
    21/05/2024, v0.3.0
        - Ahora importa funciones útiles desde el package instalable biomdp.
    
    11/05/2024, v0.2.2
        - Incluido escribir variables en Nexus de fuerzas y EMG.
    
    16/03/2024, v0.2.1
        - TODO: PROBAR CÁLCULO ÁNGULOS CON ViconUtils.EulerFromMatrix()
        - Adaptado para que funcione con modelo pie con Meta (central) o con Meta1 y Meta5.
    
    12/01/2024, v0.2.0
        - Trasladadas funciones propias del bikefitting al archivo bikefitting_funciones_apoyo.py.
    
    11/01/2024, v0.1.1
        - Mejorada función cálculo ángulos a partir de matrices rotación.
        - Cambiada nomenclatura de dimensiones:
          daTodos = daTodos.rename({'Archivo':'ID', 'nom_var':'n_var', 'lado':'side', 'eje':'axis'})

    23/04/2023, v0.1.0
        - Incluidas funciones de carga de datos directamente desde Nexus
          (Trajectories, Forces, EMG).
          
    09/06/2022, v0.0.1
                - 
"""

# =============================================================================
# DEFINE OPCIONES DE PROCESADO
# =============================================================================
bCargarMVCsPreprocesados = True
bCrearGraficas = False  # crea las gráficas de ángulos y posiciones
formatoImagenes = ".pdf"  #'.svg' #'.png'
bEnsembleAvg = (
    True  # puede ser True, False o 'completo' (la media con cada repe de fondo)
)
bSesionNexus = False  # si está abierto el Nexus con registro cargado

umbral_onset = 10.0  # Umbral para detectar el onset de músculo activo
delay_EMG = int(-0.650 * 2000)  # Retraso en señal EMG, en fotogramas
filtro_MVC = "todos"  # Para determinar qué grupo de archivos de MVC se usan para calcular los máximos en cada músculo
# Las opciones son:
#'todo' : coge todos los archivos con nombre MVC.
#'auto' : coge el máximo del propio archivo dinámico activo.
# Una cadena de texto común a un grupo de archivos (sprint, 200W, standar, etc.)
bSoloMVC = False  # para que procese solo las MVCs
bComparaLadosGraf = True  # para que compare lados con SPM1D
# =============================================================================

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

import os
import sys

from biomdp.readViconCsv import read_vicon_csv, read_vicon_csv_pl_xr
from biomdp.read_vicon_c3d import read_vicon_c3d_xr
import biomdp.slice_time_series_phases as stsp

# from biomdp.slice_time_series_phases import SliceTimeSeriesPhases as stsp

r"""
# Para que intente cargar antes la versión de mi carpeta más actualizada
carpeta_funciones = Path(r"F:\Programacion\Python\Mios\Functions")
if carpeta_funciones.exists():
    sys.path.append(carpeta_funciones.as_posix())
    # print("Cargadas funciones nexus de apoyo de mi carpeta")
else:
    sys.path.append(
        r"C:\Users\Public\Documents\Vicon\Nexus2.x\ModelTemplates\Functions"
    )

from readViconCsv import read_vicon_csv, read_vicon_csv_pl_xr
from read_vicon_c3d import read_vicon_c3d_xr
from slice_time_series_phases import SliceTimeSeriesPhases as stsp
"""
# from cortar_repes_ciclicas import CortaTimeSeries as cts
# from cortar_repes_ciclicas import corta_repes, corta_repes_xr

# from calculaEulerAngles import euler_angles_from_rot_xyz  # para calcular el ángulo entre 2 matrices de rotación

# from cortar_repes_ciclicas import corta_repes as cts

# from psd import psd #funcion para visualizar Power Spectral Density

# from viconnexusapi import ViconNexus #, ViconUtils
# vicon = ViconNexus.ViconNexus()

# from detect_peaks import detect_peaks
# from detect_onset import detect_onset

# from detecta import detect_peaks
# from detecta import detect_onset

# =============================================================================
# %% CONSTANTES
# =============================================================================

N_VARS_BILATERAL = [
    "AngSegPELVIS",
    "AngBiela",
    "vAngBiela",
    "discr_FrecPedal",
    "Ant_Cabeza",
    "Post_Cabeza",
    "C7",
    "T6",
    "L1",
]

nomVarsContinuas250 = [
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LHJC",
    "RHJC",
    "LKJC",
    "RKJC",
    "LAJC",
    "RAJC",
    "Left_KneeInt",
    "Left_KneeExt",
    "Right_KneeInt",
    "Right_KneeExt",
    "Left_AnkleInt",
    "Left_AnkleExt",
    "Right_AnkleInt",
    "Right_AnkleExt",
    "Left_TalonSup",
    "Left_TalonInf",
    "Right_TalonSup",
    "Right_TalonInf",
    "Left_Pedal_A",
    "Left_Pedal_P",
    "Right_Pedal_A",
    "Right_Pedal_P",
    "Left_Meta",
    "Right_Meta",
]

nomVarsContinuas250_completo = nomVarsContinuas250 + [
    "Left_Hombro",
    "Left_Codo",
    "Left_Muneca",
    "Right_Hombro",
    "Right_Codo",
    "Right_Muneca",
    "Post_Cabeza",
    "Ant_Cabeza",
    "Right_Cabeza",
    "Left_Cabeza",
]
# Las variables _LR del modelo completo las carga aparte
nomVarsCentrales = [
    "L1",
    "T6",
    "C7",
]

renombrar_vars = {
    "LASI": "ASI_L",
    "RASI": "ASI_R",
    "LPSI": "PSI_L",
    "RPSI": "PSI_R",
    "LHJC": "HJC_L",
    "RHJC": "HJC_R",
    "LKJC": "KJC_L",
    "RKJC": "KJC_R",
    "LAJC": "AJC_L",
    "RAJC": "AJC_R",
    "Left_KneeInt": "KneeleInt_L",
    "Right_KneeInt": "KneeInt_R",
    "Left_KneeExt": "KneeleExt_L",
    "Right_KneeExt": "KneeExt_R",
    "Left_AnkleInt": "AnkleInt_L",
    "Right_AnkleInt": "AnkleInt_R",
    "Left_AnkleExt": "AnkleExt_L",
    "Right_AnkleExt": "AnkleExt_R",
    "Left_TalonSup": "TalonSup_L",
    "Right_TalonSup": "TalonSup_R",
    "Left_Pedal_A": "Pedal_A_L",
    "Right_Pedal_P": "Pedal_P_R",
    "Left_TalonInf": "TalonInf_L",
    "Right_TalonInf": "TalonInf_R",
    "Left_Meta": "Meta_L",
    "Right_Meta": "Meta_R",
    "Left_Hombro": "Hombro_L",
    "Right_Hombro": "Hombro_R",
    "Left_Codo": "Codo_L",
    "Right_Codo": "Codo_R",
    "Left_Muneca": "Muneca_L",
    "Right_Muneca": "Muneca_R",
}


# =============================================================================
# %% Funciones varias
# =============================================================================
def crea_marcador(
    vicon,
    num_fot: int,
    n_marcador: str,
    n_marcador2: str = None,
    offset: list = None,
    fot_ref: int = None,
) -> None:
    """
    Ejemplos de uso:
    crea_marcador(vicon, num_fot=0, n_marcador='Right_MusloAS', offset=[914,29,755])
    crea_marcador(vicon, num_fot=0, n_marcador='Right_MusloAS', n_marcador2='Right_MusloAI', offset=[0,0,80])

    """
    n_subject = vicon.GetSubjectNames()[0]
    num_frames = vicon.GetTrialRange()[1]

    region_of_interest = (
        np.array(vicon.GetTrialRegionOfInterest()) - 1
    )  # corrección para que ajuste a la escala empezando en cero
    exists = np.full(
        (num_frames), False, dtype=bool
    )  # pone a cero toda la variable de si existe
    # activa solo en la región de interés del trial
    exists[region_of_interest[0] : region_of_interest[1] + 1] = False

    try:
        marker = np.array([vicon.GetTrajectory(n_subject, n_marcador)][0][:3]).T
        exists = np.array([vicon.GetTrajectory(n_subject, n_marcador)][0][3])
    except:
        marker = np.zeros((num_frames, 3))

    if n_marcador == n_marcador2:
        marker = np.array([vicon.GetTrajectory(n_subject, n_marcador2)][0][:3]).T
        exists = np.array([vicon.GetTrajectory(n_subject, n_marcador)][0][3])

    if offset is not None:
        if n_marcador2 is None:
            marker[num_fot, :] = marker[num_fot, :] + np.array(offset)
        else:
            if fot_ref is None:  # carga todo el registro
                marker2 = np.array(
                    [vicon.GetTrajectory(n_subject, n_marcador2)][0][:3]
                ).T
                marker[num_fot, :] = marker2[num_fot, :] + np.array(offset)
            else:  # carga sólo el fotograma especificado
                marker2 = np.array(
                    [vicon.GetTrajectory(n_subject, n_marcador2)][0][:3]
                ).T[fot_ref]
                marker[num_fot, :] = marker2 + np.array(offset)

    exists[num_fot] = True

    # Escribe el marcador modificado
    vicon.SetTrajectory(
        n_subject,
        n_marcador,
        marker[:, 0].tolist(),
        marker[:, 1].tolist(),
        marker[:, 2].tolist(),
        exists.tolist(),
    )


def presenta_base(vicon, origen, matrizRot, nombre, exists, escala) -> None:
    n_subject = vicon.GetSubjectNames()[0]
    ##VISUALIZA LOS MARCADORES DEL BASE
    modeledName = "BaseOrig" + nombre
    if modeledName not in vicon.GetModelOutputNames(n_subject):
        vicon.CreateModeledMarker(n_subject, modeledName)
    vicon.SetModelOutput(n_subject, modeledName, (origen).T, exists)

    modeledName = "Base" + nombre + "X"
    if modeledName not in vicon.GetModelOutputNames(n_subject):
        vicon.CreateModeledMarker(n_subject, modeledName)
    vicon.SetModelOutput(
        n_subject, modeledName, ((matrizRot[0] * escala) + origen).T, exists
    )

    modeledName = "Base" + nombre + "Y"
    if modeledName not in vicon.GetModelOutputNames(n_subject):
        vicon.CreateModeledMarker(n_subject, modeledName)
    vicon.SetModelOutput(
        n_subject, modeledName, ((matrizRot[1] * escala) + origen).T, exists
    )

    modeledName = "Base" + nombre + "Z"
    if modeledName not in vicon.GetModelOutputNames(n_subject):
        vicon.CreateModeledMarker(n_subject, modeledName)
    vicon.SetModelOutput(
        n_subject, modeledName, ((matrizRot[2] * escala) + origen).T, exists
    )


# Función para calcular el centro de 3 puntos
def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return (cx, cy)  # ((cx, cy), radius)


# Función para normalizar los vectores unidad, para no depender de scikit
def normaliza_vector(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def normaliza_vectores(x, y, z):
    def normaliza_vector_aux(a):
        # Función intermedia para usar normaliza_vector con xarray con más dimensiones
        return normaliza_vector(a.T)

    x = xr.apply_ufunc(
        normaliza_vector_aux,
        x,
        input_core_dims=[["axis", "time"]],
        output_core_dims=[["time", "axis"]],
        dask="parallelized",
        vectorize=True,
    )
    y = xr.apply_ufunc(
        normaliza_vector_aux,
        y,
        input_core_dims=[["axis", "time"]],
        output_core_dims=[["time", "axis"]],
        dask="parallelized",
        vectorize=True,
    )
    z = xr.apply_ufunc(
        normaliza_vector_aux,
        z,
        input_core_dims=[["axis", "time"]],
        output_core_dims=[["time", "axis"]],
        dask="parallelized",
        vectorize=True,
    )
    return x, y, z


def calcula_euler_angles_aux(rot_matrix):
    # TODO: PROBAR viconUtils.EulerFromMatrix()

    # rot_matrix = RlGPelvisxr
    # print(rot_matrix)
    # print(rot_matrix.shape)
    # plt.plot(rot_matrix.T)
    R = np.array(rot_matrix, dtype=np.float64, copy=False)[:3, :3]
    angles = np.zeros(3)

    angles[0] = np.arctan2(-R[2, 1], R[2, 2])
    angles[1] = np.arctan2(R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    angles[2] = np.arctan2(-R[1, 0], R[0, 0])

    # Devuelve el ángulo en radianes
    return np.rad2deg(angles)


def calcula_bases(daData, modelo_completo=False) -> xr.Dataset:
    """
    Calcula las matrices de rotación de cada segmento a partir de sus marcadores.
    Recibe trayectorias marcadores ya separadas en lados o sin separar.
    TODO: SIMPLIFICAR TODO SIN SEPARAR EN LADOS.
    """

    timer_procesa = time.perf_counter()

    dsRlG = xr.Dataset()  # guarda matrices de rotación de cada segmento

    # Versión separada por lados
    if "side" in daData.coords:

        # PELVIS
        # datos_model=np.zeros((len(daDatos.time), 3))
        try:
            origen = daData.sel(n_var="ASI").sum(dim="side", skipna=False) * 0.5

            x = daData.sel(n_var="ASI", side="R") - origen
            vprovis = (
                origen
                - (
                    daData.sel(n_var="PSI", side="L")
                    + daData.sel(n_var="ASI", side="R")
                )
                * 0.5
            )
            z = xr.cross(x, vprovis, dim="axis")
            y = xr.cross(z, x, dim="axis")

            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat(
                [x, y, z], dim="axis_base"
            )  # .transpose('ID', 'axis_base', 'time', 'axis')
            # RlG = xr.apply_ufunc(normaliza_vector, RlG)
            # RlG = RlG.T.groupby('axis_base').map(normaliza_vector)

            dsRlG["Pelvis_LR"] = RlG

        except:
            print("No se ha podido calcular el segmento PELVIS")

        # modelado['nombre'].append('AngSegPELVIS_LR')
        # modelado['datos'].append(datos_model)
        # modeled_name.append('AngSegPELVIS_LR')
        # modeled_data.append(modelado)

        # ----MUSLO_L
        try:
            origen = daData.sel(n_var="KJC", side="L")
            z = daData.sel(n_var="HJC", side="L") - daData.sel(n_var="KJC", side="L")
            vprovis = daData.sel(n_var="KneeInt", side="L") - daData.sel(
                n_var="KneeExt", side="L"
            )
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Muslo_L"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento MUSLO_L")

        # ----MUSLO_R
        try:
            origen = daData.sel(n_var="KJC", side="R")
            z = daData.sel(n_var="HJC", side="R") - daData.sel(n_var="KJC", side="R")
            vprovis = daData.sel(n_var="KneeExt", side="R") - daData.sel(
                n_var="KneeInt", side="R"
            )
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Muslo_R"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento MUSLO_R")

        # ----PIERNA_L
        try:
            origen = daData.sel(n_var="AJC", side="L")
            z = daData.sel(n_var="KJC", side="L") - daData.sel(n_var="AJC", side="L")
            vprovis = daData.sel(n_var="AnkleInt", side="L") - daData.sel(
                n_var="AnkleExt", side="L"
            )
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Pierna_L"] = RlG

            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento PIERNA_L")

        # ----PIERNA_R
        try:
            origen = daData.sel(n_var="AJC", side="R")
            z = daData.sel(n_var="KJC", side="R") - daData.sel(n_var="AJC", side="R")
            vprovis = daData.sel(n_var="AnkleExt", side="R") - daData.sel(
                n_var="AnkleInt", side="R"
            )
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Pierna_R"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento PIERNA_R")

        # ----RETROPIE_L
        if "Meta" in daData.n_var:
            daMeta = daData.sel(n_var="Meta")
        else:
            daMeta = (
                daData.sel(n_var=["Meta1", "Meta5"]).sum(dim="n_var", skipna=False)
                * 0.5
            )

        try:

            origen = daData.sel(n_var="TalonInf", side="L")
            y = daMeta.sel(side="L") - daData.sel(n_var="TalonSup", side="L")
            vprovis = daData.sel(n_var="TalonSup", side="L") - daData.sel(
                n_var="TalonInf", side="L"
            )
            x = xr.cross(y, vprovis, dim="axis")
            z = xr.cross(x, y, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Retropie_L"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento RETROPIE_L")

        # ----RETROPIE_R
        try:
            origen = daData.sel(n_var="TalonInf", side="R")
            y = daMeta.sel(side="R") - daData.sel(n_var="TalonSup", side="R")
            vprovis = daData.sel(n_var="TalonSup", side="R") - daData.sel(
                n_var="TalonInf", side="R"
            )
            x = xr.cross(y, vprovis, dim="axis")
            z = xr.cross(x, y, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Retropie_R"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento RETROPIE_R")

        # =============================================================================
        # Modelo parte superior
        # =============================================================================
        if modelo_completo:
            # ----LUMBAR_LR
            try:
                origen = (
                    daData.sel(n_var="PSI", side="L")
                    + daData.sel(n_var="PSI", side="R")
                ) * 0.5
                x = daData.sel(n_var="PSI", side="R") - origen
                vprovis = daData.sel(n_var="L1", side="R") - origen
                y = xr.cross(vprovis, x, dim="axis")
                z = xr.cross(x, y, dim="axis")
                x, y, z = normaliza_vectores(x, y, z)

                RlG = xr.concat([x, y, z], dim="axis_base")
                dsRlG["Lumbar_LR"] = RlG
                # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

            except:
                print("No se ha podido calcular el segmento LUMBAR")

            # ----TORAX_LR
            try:
                origen = daData.sel(n_var="C7", side="R")
                z = daData.sel(n_var="C7", side="R") - daData.sel(
                    n_var="T6", side="R"
                )  # .expand_dims({"n_var": ["C7_T6"]})
                vprovis = daData.sel(n_var="Hombro", side="R") - daData.sel(
                    n_var="Hombro", side="L"
                )
                y = xr.cross(z, vprovis, dim="axis").drop_vars("n_var")
                x = xr.cross(y, z, dim="axis")
                x, y, z = normaliza_vectores(x, y, z)

                RlG = xr.concat([x, y, z], dim="axis_base")
                dsRlG["Torax_LR"] = RlG
                # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

            except:
                print("No se ha podido calcular el segmento TORAX")

            # ----CABEZA_LR
            try:
                origen = daData.sel(n_var="Post_Cabeza")
                y = daData.sel(n_var="Ant_Cabeza") - daData.sel(n_var="Post_Cabeza")
                # DÓNDE SE CARGA LADOS DE CABEZA????
                vprovis = daData.sel(n_var="Cabeza", side="R") - daData.sel(
                    n_var="Cabeza", side="L"
                )
                z = xr.cross(vprovis, y, dim="axis").drop_vars("n_var")
                x = xr.cross(y, z, dim="axis")
                x, y, z = normaliza_vectores(x, y, z)

                RlG = xr.concat([x, y, z], dim="axis_base")
                dsRlG["Cabeza_LR"] = RlG
                # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

            except:
                print("No se ha podido calcular el segmento CABEZA")

        dsRlG = dsRlG.drop_vars(["n_var", "side"])

    else:  # versión sin separar por side
        # PELVIS
        # datos_model=np.zeros((len(daDatos.time), 3))
        try:
            origen = (daData.sel(n_var="ASI_L") + daData.sel(n_var="ASI_R")) * 0.5
            x = daData.sel(n_var="ASI_R") - origen
            vprovis = (
                origen - (daData.sel(n_var="PSI_L") + daData.sel(n_var="ASI_R")) * 0.5
            )
            z = xr.cross(x, vprovis, dim="axis")
            y = xr.cross(z, x, dim="axis")

            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat(
                [x, y, z], dim="axis_base"
            )  # .transpose('ID', 'axis_base', 'time', 'axis')
            # RlG = xr.apply_ufunc(normaliza_vector, RlG)
            # RlG = RlG.T.groupby('axis_base').map(normaliza_vector)

            dsRlG["Pelvis_LR"] = RlG

        except:
            print("No se ha podido calcular el segmento PELVIS")

        # modelado['nombre'].append('AngSegPELVIS_LR')
        # modelado['datos'].append(datos_model)
        # modeled_name.append('AngSegPELVIS_LR')
        # modeled_data.append(modelado)

        # ----MUSLO_L
        try:
            origen = daData.sel(n_var="KJC_L")
            z = daData.sel(n_var="HJC_L") - daData.sel(n_var="KJC_L")
            vprovis = daData.sel(n_var="KneeInt_L") - daData.sel(n_var="KneeExt_L")
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Muslo_L"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento MUSLO_L")

        # ----MUSLO_R
        try:
            origen = daData.sel(n_var="KJC_R")
            z = daData.sel(n_var="HJC_R") - daData.sel(n_var="KJC_R")
            vprovis = daData.sel(n_var="KneeExt_R") - daData.sel(n_var="KneeInt_R")
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Muslo_R"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento MUSLO_R")

        # ----PIERNA_L
        try:
            origen = daData.sel(n_var="AJC_L")
            z = daData.sel(n_var="KJC_L") - daData.sel(n_var="AJC_L")
            vprovis = daData.sel(n_var="AnkleInt_L") - daData.sel(n_var="AnkleExt_L")
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Pierna_L"] = RlG

            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento PIERNA_L")

        # ----PIERNA_R
        try:
            origen = daData.sel(n_var="AJC_R")
            z = daData.sel(n_var="KJC_R") - daData.sel(n_var="AJC_R")
            vprovis = daData.sel(n_var="AnkleExt_R") - daData.sel(n_var="AnkleInt_R")
            y = xr.cross(z, vprovis, dim="axis")
            x = xr.cross(y, z, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Pierna_R"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento PIERNA_R")

        # ----RETROPIE_L
        try:
            origen = daData.sel(n_var="TalonInf_L")
            y = daData.sel(n_var="Meta_L") - daData.sel(n_var="TalonSup_L")
            vprovis = daData.sel(n_var="TalonSup_L") - daData.sel(n_var="TalonInf_L")
            x = xr.cross(y, vprovis, dim="axis")
            z = xr.cross(x, y, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Retropie_L"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento RETROPIE_L")

        # ----RETROPIE_R
        try:
            origen = daData.sel(n_var="TalonInf_R")
            y = daData.sel(n_var="Meta_R") - daData.sel(n_var="TalonSup_R")
            vprovis = daData.sel(n_var="TalonSup_R") - daData.sel(n_var="TalonInf_R")
            x = xr.cross(y, vprovis, dim="axis")
            z = xr.cross(x, y, dim="axis")
            x, y, z = normaliza_vectores(x, y, z)

            RlG = xr.concat([x, y, z], dim="axis_base")
            dsRlG["Retropie_R"] = RlG
            # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

        except:
            print("No se ha podido calcular el segmento RETROPIE_R")

        # =============================================================================
        # Modelo parte superior
        # =============================================================================
        if modelo_completo:
            # ----LUMBAR_LR
            try:
                origen = (daData.sel(n_var="PSI_L") + daData.sel(n_var="PSI_R")) * 0.5
                x = daData.sel(n_var="PSI_R") - origen
                vprovis = daData_completo.sel(n_var="L1") - origen
                y = xr.cross(vprovis, x, dim="axis")
                z = xr.cross(x, y, dim="axis")
                x, y, z = normaliza_vectores(x, y, z)

                RlG = xr.concat([x, y, z], dim="axis_base")
                dsRlG["Lumbar_LR"] = RlG
                # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

            except:
                print("No se ha podido calcular el segmento LUMBAR")

            # ----TORAX_LR
            try:
                origen = daData_completo.sel(n_var="C7")
                z = daData_completo.sel(n_var="C7") - daData_completo.sel(n_var="T6")
                vprovis = daData.sel(n_var="Hombro", side="R") - daData.sel(
                    n_var="Hombro", side="L"
                )
                y = xr.cross(z, vprovis, dim="axis")
                x = xr.cross(y, z, dim="axis")
                x, y, z = normaliza_vectores(x, y, z)

                RlG = xr.concat([x, y, z], dim="axis_base")
                dsRlG["Torax_LR"] = RlG
                # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

            except:
                print("No se ha podido calcular el segmento TORAX")

            # ----CABEZA_LR
            try:
                origen = daData_completo.sel(n_var="Post_Cabeza")
                y = daData_completo.sel(n_var="Ant_Cabeza") - daData_completo.sel(
                    n_var="Post_Cabeza"
                )
                # DÓNDE SE CARGA LADOS DE CABEZA????
                vprovis = daData_completo.sel(
                    n_var="Cabeza", side="R"
                ) - daData_completo.sel(n_var="Cabeza", side="L")
                z = xr.cross(vprovis, y, dim="axis")
                x = xr.cross(y, z, dim="axis")
                x, y, z = normaliza_vectores(x, y, z)

                RlG = xr.concat([x, y, z], dim="axis_base")
                dsRlG["Cabeza_LR"] = RlG
                # datos_model.plot.line(x='time', col='ID', col_wrap=4, hue='axis')

            except:
                print("No se ha podido calcular el segmento CABEZA")

        dsRlG = dsRlG.drop_vars(["n_var"])

    print(
        f"Bases de {len(daData.ID)} registros calculados en {time.perf_counter() - timer_procesa:.3f} s."
    )

    return dsRlG


def calcula_angulos_segmentos(dsRlG, verbose=False) -> xr.Dataset:
    timer_procesa = time.perf_counter()

    dsAngSeg = xr.Dataset()

    for RlG in dsRlG:
        if verbose:
            print(f"Calculando {RlG}...")
        if dsRlG[RlG].isnull().all():  # si viene vacío se lo salta
            if verbose:
                print("vacío")
            dsAngSeg[f"AngSeg{RlG.upper()}"] = xr.full_like(
                dsRlG[RlG].isel(axis_base=0), np.nan
            )
            continue
        dsAngSeg[f"AngSeg{RlG.upper()}"] = calcula_angulo_xr(RlGChild=dsRlG[RlG])
        if verbose:
            print("calculado")
    # RlGChild=dsRlG[RlG]

    print(
        f"Procesados {len(dsRlG.ID)} registros en {time.perf_counter() - timer_procesa:.3f} s."
    )

    return dsAngSeg


def calcula_angulos_articulaciones(
    dsRlG, daTrajec=None, modelo_completo=False, verbose=False
) -> xr.Dataset:
    """
    Calcula ángulos de articulaciones a partir de las matrices de rotación
    daTrajec necesario sólo para el modelo completo
    """
    if modelo_completo and daTrajec is None:
        raise ValueError("Datos de trayectoria necesarios para el modelo completo")
        return

    timer_procesa = time.perf_counter()

    dsAngles = xr.Dataset()

    """
    child = dsRlG['Muslo_L'][:,0,0,:]
    parent = dsRlG['Pelvis_LR'][:,0,0,:]
    """
    modeled_names = [
        "AngArtHip_L",
        "AngArtHip_R",
        "AngArtKnee_L",
        "AngArtKnee_R",
        "AngArtAnkle_L",
        "AngArtAnkle_R",
    ]
    child_parents = [
        ["Muslo_L", "Pelvis_LR"],
        ["Muslo_R", "Pelvis_LR"],
        ["Pierna_L", "Muslo_L"],
        ["Pierna_R", "Muslo_R"],
        ["Retropie_L", "Pierna_L"],
        ["Retropie_R", "Pierna_R"],
    ]

    if modelo_completo:
        modeled_names += [
            "AngArtLumbar_LR",
            "AngArtToracoLumbar_LR",
            #'AngArtL1_LR', 'AngArtT6_LR',
            "AngArtCuello_LR",
            #'AngArtElbow_L', 'AngArtElbow_R',
        ]
        child_parents += [
            ["Pelvis_LR", "Lumbar_LR"],
            ["Lumbar_LR", "Torax_LR"],
            ["Torax_LR", "Cabeza_LR"],
        ]

    for modeled_name, child_parent in zip(modeled_names, child_parents):
        if verbose:
            print(f"Calculando {modeled_name}...")

        try:
            child = dsRlG[child_parent[0]]
            parent = dsRlG[child_parent[1]]
            if (
                child.isnull().all() or parent.isnull().all()
            ):  # si viene vacío se lo salta
                if verbose:
                    print("vacío")
                dsAngles[modeled_name] = xr.full_like(child.isel(axis_base=0), np.nan)
                continue

            dsAngles[modeled_name] = calcula_angulo_xr(RlGChild=child, RlGParent=parent)
            if verbose:
                print("calculado")
            """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                            input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                            output_core_dims=[['axis']],
                            vectorize=True
                            )
            """
        except:
            print("No se ha podido calcular el ángulo", modeled_name)

    # ----Correcciones signos específicas
    # HIP (# flex +, abd +, rot ext +)
    if "AngArtHip_R" in dsAngles.variables:
        dsAngles["AngArtHip_R"].loc[dict(axis="y")] = -dsAngles["AngArtHip_R"].loc[
            dict(axis="y")
        ]  # invierte el signo de la abd-add
    if "AngArtHip_R" in dsAngles.variables:
        dsAngles["AngArtHip_R"].loc[dict(axis="z")] = -dsAngles["AngArtHip_R"].loc[
            dict(axis="z")
        ]  # invierte el signo de la rot
    # KNEE
    if "AngArtKnee_L" in dsAngles.variables:
        dsAngles["AngArtKnee_L"].loc[dict(axis="x")] = -dsAngles["AngArtKnee_L"].loc[
            dict(axis="x")
        ]  # solo se invierte el signo de la flexoext
    if "AngArtKnee_R" in dsAngles.variables:
        dsAngles[
            "AngArtKnee_R"
        ] *= -1  # dsAngles[modeled_name].loc[dict(axis=['x', 'y', 'z'])
    # ANKLE
    if "AngArtAnkle_L" in dsAngles.variables:
        dsAngles["AngArtAnkle_L"].loc[dict(axis="y")] = -dsAngles["AngArtAnkle_L"].loc[
            dict(axis="y")
        ]  # solo se invierte el signo de la pronosup tobillo izq
    if "AngArtAnkle_R" in dsAngles.variables:
        dsAngles["AngArtAnkle_R"].loc[dict(axis="z")] = -dsAngles["AngArtAnkle_R"].loc[
            dict(axis="z")
        ]  # solo se invierte el signo de la rotación tobillo der

    # ----ÁNGULOS LINEALES
    if modelo_completo:
        # TODO: FALTA DIFERENCIAR SI VIENE CON SIDE O NO
        # ----L1
        modeled_name = "AngArtL1_LR"
        try:
            # Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var="T6")
            p2 = daTrajec.sel(n_var="L1")
            p3 = (
                daTrajec.sel(n_var="PSI", side="L")
                + daTrajec.sel(n_var="PSI", side="R")
            ) * 0.5
            dsAngles[modeled_name] = 180 - (
                np.arctan2(
                    np.linalg.norm(xr.cross(p1 - p2, p3 - p2, dim="axis")),
                    xr.dot(p1 - p2, p3 - p2, dim="axis"),
                )
                * 180
                / np.pi
            )

            # dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print("No se ha podido calcular el ángulo", modeled_name)

        # ----T6
        modeled_name = "AngArtT6_LR"
        try:
            # Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var="C7")
            p2 = daTrajec.sel(n_var="T6")
            p3 = daTrajec.sel(n_var="L1")
            dsAngles[modeled_name] = 180 - (
                np.arctan2(
                    np.linalg.norm(xr.cross(p1 - p2, p3 - p2, dim="axis")),
                    xr.dot(p1 - p2, p3 - p2, dim="axis"),
                )
                * 180
                / np.pi
            )

            # dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print("No se ha podido calcular el ángulo", modeled_name)

        # ----ELBOW_L
        modeled_name = "AngArtElbow_L"
        try:
            # Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var="Muneca", side="L")
            p2 = daTrajec.sel(n_var="Codo", side="L")
            p3 = daTrajec.sel(n_var="Hombro", side="L")
            dsAngles[modeled_name] = 180 - (
                np.arctan2(
                    np.linalg.norm(xr.cross(p1 - p2, p3 - p2, dim="axis")),
                    xr.dot(p1 - p2, p3 - p2, dim="axis"),
                )
                * 180
                / np.pi
            )

            # dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print("No se ha podido calcular el ángulo", modeled_name)

        # ----ELBOW_R
        modeled_name = "AngArtElbow_R"
        try:
            # Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var="Muneca", side="R")
            p2 = daTrajec.sel(n_var="Codo", side="R")
            p3 = daTrajec.sel(n_var="Hombro", side="R")
            dsAngles[modeled_name] = 180 - (
                np.arctan2(
                    np.linalg.norm(xr.cross(p1 - p2, p3 - p2, dim="axis")),
                    xr.dot(p1 - p2, p3 - p2, dim="axis"),
                )
                * 180
                / np.pi
            )

            # dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print("No se ha podido calcular el ángulo", modeled_name)

    '''
    #----HIP_L
    modeled_name = 'AngArtHip_L'
    try:
        child = dsRlG['Muslo_L']
        parent = dsRlG['Pelvis_LR']
                
        dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
        """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                        input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                        output_core_dims=[['axis']],
                        vectorize=True
                        )
        """
        # flex +, abd +, rot ext +
        #dsAngles[modeled_name].plot.line(x='time', col='ID', col_wrap=4, hue='axis')
    except:
        print('No se ha podido calcular el ángulo', modeled_name)

    #----HIP_R
    modeled_name = 'AngArtHip_R'
    try:
        child = dsRlG['Muslo_R']
        parent = dsRlG['Pelvis_LR']
        dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
        """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                        input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                        output_core_dims=[['axis']],
                        vectorize=True
                        )
        """
        #Ajusta signos
        dsAngles[modeled_name].loc[dict(axis='y')] = -dsAngles[modeled_name].loc[dict(axis='y')] #invierte el signo de la abd-add
        dsAngles[modeled_name].loc[dict(axis='z')] = -dsAngles[modeled_name].loc[dict(axis='z')] #invierte el signo de la rot
        #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
    except:
        print('No se ha podido calcular el ángulo', modeled_name)

    #----KNEE_L
    modeled_name = 'AngArtKnee_L'
    try:
        child = dsRlG['Pierna_L']
        parent = dsRlG['Muslo_L']        
        dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
        """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                        input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                        output_core_dims=[['axis']],
                        vectorize=True
                        )
        """
        #Ajusta signos
        dsAngles[modeled_name].loc[dict(axis='x')] = -dsAngles[modeled_name].loc[dict(axis='x')] #solo se invierte el signo de la flexoext        
        #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
    except:
        print('No se ha podido calcular el ángulo', modeled_name)
    
    #----KNEE_R
    modeled_name = 'AngArtKnee_R'
    try:
        child = dsRlG['Pierna_R']
        parent = dsRlG['Muslo_R']        
        dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
        """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                        input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                        output_core_dims=[['axis']],
                        vectorize=True
                        )
        """
        #Ajusta signos
        dsAngles[modeled_name] *= -1 #dsAngles[modeled_name].loc[dict(axis=['x', 'y', 'z'])
        #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
    except:
        print('No se ha podido calcular el ángulo', modeled_name)

    #----ANKLE_L
    modeled_name = 'AngArtAnkle_L'
    try:
        child = dsRlG['Retropie_L']
        parent = dsRlG['Pierna_L']        
        dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
        """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                        input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                        output_core_dims=[['axis']],
                        vectorize=True
                        )
        """
        #Ajusta signos
        dsAngles[modeled_name].loc[dict(axis='y')] = -dsAngles[modeled_name].loc[dict(axis='y')] #solo se invierte el signo de la pronosup tobillo izq
        #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
    except:
        print('No se ha podido calcular el ángulo', modeled_name)

    #----ANKLE_R
    modeled_name = 'AngArtAnkle_R'
    try:
        child = dsRlG['Retropie_R']
        parent = dsRlG['Pierna_R']        
        dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
        """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                        input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                        output_core_dims=[['axis']],
                        vectorize=True
                        )
        """
        #Ajusta signos
        dsAngles[modeled_name].loc[dict(axis='z')] = -dsAngles[modeled_name].loc[dict(axis='z')] #solo se invierte el signo de la rotación tobillo der
        #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
    except:
        print('No se ha podido calcular el ángulo', modeled_name)
    
    if modelo_completo:
        #----LUMBAR
        modeled_name = 'AngArtLumbar_LR'
        try:
            child = dsRlG['Pelvis_LR']
            parent = dsRlG['Lumbar_LR']        
            dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
            """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                            input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                            output_core_dims=[['axis']],
                            vectorize=True
                            )
            """
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)

        #----TORACOLUMBAR
        modeled_name = 'AngArtToracoLumbar_LR'
        try:
            child = dsRlG['Lumbar_LR']
            parent = dsRlG['Torax_LR']        
            dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
            """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                            input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                            output_core_dims=[['axis']],
                            vectorize=True
                            )
            """
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)
        
        #----L1
        modeled_name = 'AngArtL1_LR'
        try:
            #Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var='T6')
            p2 = daTrajec.sel(n_var='L1')
            p3 = (daTrajec.sel(n_var='PSI', side='L') + daTrajec.sel(n_var='PSI', side='R')) * 0.5
            dsAngles[modeled_name] = 180 - (np.arctan2(np.linalg.norm(xr.cross(p1-p2, p3-p2, dim='axis')), xr.dot(p1-p2, p3-p2, dim='axis'))*180/np.pi)
          
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)

        #----T6
        modeled_name = 'AngArtT6_LR'
        try:
            #Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var='C7')
            p2 = daTrajec.sel(n_var='T6')
            p3 = daTrajec.sel(n_var='L1')
            dsAngles[modeled_name] = 180 - (np.arctan2(np.linalg.norm(xr.cross(p1-p2, p3-p2, dim='axis')), xr.dot(p1-p2, p3-p2, dim='axis'))*180/np.pi)
          
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)

        #----CUELLO
        modeled_name = 'AngArtCuello_LR'
        try:
            child = dsRlG['Torax_LR']
            parent = dsRlG['Cabeza_LR']        
            dsAngles[modeled_name] = calcula_angulo_xr(child, parent)
            """dsAngles[modeled_name] = xr.apply_ufunc(calcula_ang_artic_aux, child, parent,
                            input_core_dims=[['axis_base', 'axis'], ['axis_base', 'axis']],
                            output_core_dims=[['axis']],
                            vectorize=True
                            )
            """
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)
        
        #----ELBOW_L
        modeled_name = 'AngArtElbow_L'
        try:
            #Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var='Muneca', side='L')
            p2 = daTrajec.sel(n_var='Codo', side='L')
            p3 = daTrajec.sel(n_var='Hombro', side='L')
            dsAngles[modeled_name] = 180 - (np.arctan2(np.linalg.norm(xr.cross(p1-p2, p3-p2, dim='axis')), xr.dot(p1-p2, p3-p2, dim='axis'))*180/np.pi)
          
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)

        #----ELBOW_R
        modeled_name = 'AngArtElbow_R'
        try:
            #Probar con esto en xr, si no, hacer función aux y usar ufunc
            p1 = daTrajec.sel(n_var='Muneca', side='R')
            p2 = daTrajec.sel(n_var='Codo', side='R')
            p3 = daTrajec.sel(n_var='Hombro', side='R')
            dsAngles[modeled_name] = 180 - (np.arctan2(np.linalg.norm(xr.cross(p1-p2, p3-p2, dim='axis')), xr.dot(p1-p2, p3-p2, dim='axis'))*180/np.pi)
          
            #dsAngles[modeled_name].plot.line(x='time', row='ID', hue='axis')
        except:
            print('No se ha podido calcular el ángulo', modeled_name)
    '''
    print(
        f"Calculados ángulos de {len(dsRlG.ID)} registros en {time.perf_counter() - timer_procesa:.3f} s."
    )

    return dsAngles


def calcula_angulos_desde_trajec(
    daData, modelo_completo=False, tipo="all", verbose=False
) -> xr.DataArray:
    """
    Calcula ángulos de articulaciones a partir de las trayectorias.
    Paso intermedio de calcular matrices de rotación
    tipo: 'artic', 'segment' o 'all'
    """
    timer_procesa = time.perf_counter()  # inicia el contador de tiempo

    print("\nCalculando matrices de rotación...")
    dsRlG = calcula_bases(daData, modelo_completo)  # daData=daDatos.isel(ID=0))

    if tipo in ["segment", "all"]:
        print("\nCalculando ángulos de segmentos...")
        dsAngSegments = calcula_angulos_segmentos(dsRlG, verbose=verbose)

    if tipo in ["artic", "all"]:
        print("\nCalculando ángulos articulares...")
        dsAngArtics = calcula_angulos_articulaciones(
            dsRlG, modelo_completo, verbose=verbose
        )
    # dsAngArtics['AngArtHip_L'].plot.line(x='time', row='ID', hue='axis')

    if tipo == "all":
        daAngles = xr.merge([dsAngSegments, dsAngArtics])
    elif tipo == "artic":
        daAngles = dsAngArtics
    elif tipo == "segment":
        daAngles = dsAngSegments

    daAngles = daAngles.to_array().rename({"variable": "n_var"})  # .to_dataarray()

    if "side" in daData.coords:
        daAngles = separa_trayectorias_lados(daAngles)

    daAngles.name = "Angles"
    daAngles.attrs["units"] = "deg"
    daAngles.attrs["freq"] = daData.freq

    # daAngles.sel(n_var='AngArtHip').plot.line(x='time', row='ID', col='axis', hue='side')

    print(
        f"Procesados {len(daAngles.ID)} registros en {time.perf_counter() - timer_procesa:.3f} s."
    )

    return daAngles


def ajusta_etiquetas_lado_final(daData) -> xr.DataArray:
    labels = daData["n_var"].values

    # Busca variablel bilaterales
    n_var_nuevo = [f"{i}_LR" if i in N_VARS_BILATERAL else i for i in labels]

    # Ajusta las etiquetas a formato lados L, R
    n_var_nuevo = [
        f'{i.split("Left_")[-1]}_L' if "Left" in i else i for i in n_var_nuevo
    ]
    n_var_nuevo = [
        f'{i.split("Right_")[-1]}_R' if "Right" in i else i for i in n_var_nuevo
    ]
    n_var_nuevo = [
        i[1:] + "_L" if i[0] == "L" and i[0:6] != "Length" and i != "L1_LR" else i
        for i in n_var_nuevo
    ]
    n_var_nuevo = [i[1:] + "_R" if i[0] == "R" else i for i in n_var_nuevo]

    daData = daData.assign_coords(n_var=n_var_nuevo)
    return daData


def traduce_variables(nomvar) -> str:
    if nomvar == "GLU":
        nom = "Glúteo"
    elif nomvar == "BIC":
        nom = "Bíceps femoral"
    elif nomvar == "REC":
        nom = "Recto femoral"
    elif nomvar == "VME":
        nom = "Vasto interno"
    elif nomvar == "GAS":
        nom = "Gemelo"
    elif nomvar == "TIB":
        nom = "Tibial"

    elif nomvar == "AngArtHip":
        nom = "Ang Cadera"
    elif nomvar == "AngArtKnee":
        nom = "Ang Rodilla"
    elif nomvar == "AngArtAnkle":
        nom = "Ang Tobillo"
    elif nomvar == "AngSegPELVIS":
        nom = "Pelvis"
    elif nomvar == "x" or nomvar == ["y", "z"]:
        nom = "sagital"
    elif nomvar == "y" or nomvar == ["x", "z"]:
        nom = "frontal"
    elif nomvar == ["x", "y"]:
        nom = "cenital"
    elif nomvar == "z":
        nom = "rotación"

    elif nomvar == "Eje x":
        nom = "mediolateral"
    elif nomvar == "Eje y":
        nom = "anteroposterior"
    elif nomvar == "Eje z":
        nom = "vertical"

    elif nomvar == "HJC":
        nom = "Eje Cadera"
    elif nomvar == "KJC":
        nom = "Eje Rodilla"
    elif nomvar == "AJC":
        nom = "Eje Tobillo"
    elif nomvar == "vAngBiela":
        nom = "Velocidad Ang biela"
    else:
        nom = "desconocido"
    return nom


def calcula_angulo_xr(RlGChild, RlGParent=None) -> xr.DataArray:
    """
    Recibe matrices de rotación. Si llega una sola calcula el ángulo del segmento,
    si llegan dos calcula el ángulo entre los dos segmentos (articulación).
    """
    # Comprueba si hay que calcular para segmento o articulación
    if RlGParent is None:
        RlGParent = RlGChild
        bSegment = True
    else:
        bSegment = False

    # TODO: PROBAR FUNCIÓN NEXUS ViconUtils.EulerFromMatrix()
    #  from viconnexusapi import ViconUtils
    def calc_ang_aux(child, parent):
        if bSegment:
            R = child
        else:
            R = np.dot(child, parent.T)
        x = np.arctan2(-R[2, 1], R[2, 2])
        y = np.arctan2(R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        z = np.arctan2(-R[1, 0], R[0, 0])

        return np.rad2deg(np.array([x, y, z]))

    """
    RlGChild = dsRlG[child_parent[0]]
    RlGParent = dsRlG[child_parent[1]]

    child = RlGChild[:,0].isel(time=0).data
    parent = RlGParent[:,0].isel(time=0).data
    
    ViconUtils.EulerFromMatrix
    """

    """
    #PROBAR CON XR.DOT Y calcula_euler_angles_aux
    xr.dot(RlGChild, RlGParent.T, dim=['axis_base', 'axis'])
    Rxr = xr.dot(RlGChild, RlGParent.T, dim='axis') #['axis_base', 'axis'])
    Rxr[0,0]
    """
    angulo = xr.apply_ufunc(
        calc_ang_aux,
        RlGChild,
        RlGParent,
        input_core_dims=[["axis_base", "axis"], ["axis_base", "axis"]],
        output_core_dims=[["axis"]],
        # exclude_dims=set(('axis',)),
        dask="parallelized",
        vectorize=True,
    )
    # angulo.sel(axis='z').plot.line(x='time')

    """
    datos_model = xr.apply_ufunc(calcula_euler_angles_aux, RlG,
                                 input_core_dims=[['axis_base', 'axis']],
                                 output_core_dims=[['axis']],
                                 vectorize=True,
                                 )
    """
    return angulo


# =============================================================================


# =============================================================================
# %% Extrae variables del Nexus directamente
# =============================================================================
def carga_variables_nexus_trajectories(vicon=None, n_vars=None) -> xr.DataArray:
    print("Cargando datos Trayectorias...")
    timer = time.time()  # inicia el contador de tiempo

    if n_vars is None:
        n_vars = vicon.GetMarkerNames(vicon.GetSubjectNames()[0])
        n_vars = n_vars + [
            "LASI",
            "RASI",
            "LHJC",
            "RHJC",
            "LKJC",
            "RKJC",
            "LAJC",
            "RAJC",
            "Left_KneeExt",
            "Left_KneeInt",
            "Right_KneeExt",
            "Right_KneeInt",
            "Left_AnkleExt",
            "Left_AnkleInt",
            "Right_AnkleExt",
            "Right_AnkleInt",
        ]  # vicon.GetModelOutputNames(vicon.GetSubjectNames()[0])

    # Borra si hay nombres repetidos
    n_vars = np.unique(n_vars).tolist()

    markers = []
    for nom in n_vars:
        # print(f'Intentando cargar la variable {nom}')

        dat = vicon.GetTrajectory(vicon.GetSubjectNames()[0], nom)
        dat = np.where(np.array(dat[3]), dat[:3], np.nan)
        # dat = np.array([vicon.GetTrajectory(vicon.GetSubjectNames()[0], nom)][0])
        # np.where(dat[3,:]==1., dat[:3], np.nan)
        # dat = np.array([vicon.GetTrajectory(vicon.GetSubjectNames()[0], nom)][0][:3])
        if dat.size == 0:
            dat = vicon.GetModelOutput(vicon.GetSubjectNames()[0], nom)
            dat = np.where(np.array(dat[1]), np.array(dat[0]), np.nan)

            # dat = np.array(
            #     [vicon.GetModelOutput(vicon.GetSubjectNames()[0], nom)[0][:3]]
            # )[
            #     0
            # ]  # .reshape(3, vicon.GetTrialRange()[1]).T
            print(f"{nom} cargada desde Modeled Markers")
        if dat.size == 0:
            dat = np.full((3, vicon.GetTrialRange()[1]), np.nan, dtype=float)
            print(f"No se ha podido cargar la variable {nom}")

        markers.append(dat)

    # print(f'Cargada la variable {nom}')
    # Añade una dimensión para el ID
    data = np.expand_dims(np.array(markers), axis=0)
    coords = {
        "ID": [vicon.GetSubjectNames()[0]],
        "n_var": n_vars,
        "axis": ["x", "y", "z"],
        "time": np.arange(0, data.shape[-1]) / vicon.GetFrameRate(),
    }

    da = xr.DataArray(
        data=data,
        dims=coords.keys(),
        coords=coords,
    )

    da.name = "Trajectories"
    da.attrs["freq"] = float(vicon.GetFrameRate())
    da.attrs["units"] = "mm"
    da.time.attrs["units"] = "s"

    print("Cargados los datos en {0:.3f} s \n".format(time.time() - timer))

    return da


def carga_variables_nexus_force(vicon=None, n_plate=None) -> xr.DataArray:
    print("Cargando datos Fuerza...")
    timer = time.time()  # inicia el contador de tiempo

    deviceForce = [x for x in vicon.GetDeviceNames() if n_plate in x][0]
    deviceID = vicon.GetDeviceIDFromName(deviceForce)
    _, _, freqForce, outputIDs, _, _ = vicon.GetDeviceDetails(deviceID)

    # Coge los nombres de los canales
    cols = [
        vicon.GetDeviceOutputDetails(deviceID, outputIDs[i])
        for i in range(len(outputIDs))
    ]

    # Carga las fuerzas de la plataforma
    ejes = []
    for n, eje in enumerate(["x", "y", "z"], start=1):
        ejes.append(vicon.GetDeviceChannel(deviceID, outputIDs[0], n)[0])

    data = np.expand_dims(np.array(ejes), axis=0)  # añade una dimensión para el ID
    coords = {
        "ID": [vicon.GetSubjectNames()[0]],
        "axis": ["x", "y", "z"],
        "time": np.arange(0, data.shape[-1]) / freqForce,
    }
    da = xr.DataArray(
        data=data,
        dims=coords.keys(),
        coords=coords,
    )
    da.loc[dict(axis="z")] *= -1
    # da.plot.line(x='time')

    da.name = "Force"
    da.attrs["freq"] = float(freqForce)
    da.attrs["freq_ref"] = vicon.GetFrameRate()  # frequency of markers
    da.attrs["units"] = "N"
    da.time.attrs["units"] = "s"

    print("Cargados los datos en {0:.3f} s \n".format(time.time() - timer))
    return da


def carga_variables_nexus_EMG(vicon=None, n_vars=None) -> xr.DataArray:
    print("Cargando datos EMG...")
    timer = time.time()  # inicia el contador de tiempo

    deviceEMG = [x for x in vicon.GetDeviceNames() if "EMG" in x][0]
    deviceID = vicon.GetDeviceIDFromName(deviceEMG)

    _, _, freqEMG, outputIDs, _, _ = vicon.GetDeviceDetails(deviceID)

    # dir(ViconNexus.ViconNexus)
    # help(ViconNexus.ViconNexus.GetDeviceChannelIDFromName)
    # help(ViconNexus.ViconNexus.GetDeviceChannel)

    # nom_canal,_,_,_,_,channelIDs = vicon.GetDeviceOutputDetails(deviceID,outputIDs[1])

    # Coge los nombres de los canales
    cols = [
        vicon.GetDeviceOutputDetails(deviceID, outputIDs[i])[0]
        for i in range(len(outputIDs))
    ]

    # data, ready, rate = vicon.GetDeviceChannel(deviceID,cols.index('EMG2')+1,1)
    # plt.plot(data)

    channel = []
    for n, nom in enumerate(cols):
        channel.append(vicon.GetDeviceChannel(deviceID, outputIDs[n], 1)[0])

    data = np.expand_dims(np.array(channel), axis=0)  # añade una dimensión para el ID
    coords = {
        "ID": [vicon.GetSubjectNames()[0]],
        "channel": cols,
        "time": np.arange(0, data.shape[-1]) / freqEMG,
    }
    da = xr.DataArray(
        data=data,
        dims=coords.keys(),
        coords=coords,
    )

    # Ajusta a mano los sensores de cada lado
    """renombrar_vars={'EMG9':'GLU', 'EMG10':'REC', 'EMG11':'BIV',
                    'EMG12':'VAE', 'EMG13':'VAI',
                    'EMG14':'GAS', 'EMG15':'TIB',
                    }
    
    # renombrar_vars={'EMG1':'GLU_R', 'EMG2':'BIC_R', 'EMG3':'REC_R', 'EMG4':'VME_R', 'EMG5':'GAS_R', 'EMG6':'TIB_R', 
    #                 'EMG7':'GLU_L', 'EMG8':'BIC_L', 'EMG9':'REC_L', 'EMG10':'VME_L', 'EMG11':'GAS_L', 'EMG12':'TIB_L',
    #                 }
    """
    # Se queda las variables seleccionadas
    if n_vars is not None:
        # daEMG = daEMG.sel(channel=daEMG.channel.str.contains('EMG'))
        da = da.sel(channel=n_vars)

    da = da * 1000  # pasa a milivoltios
    da.name = "EMG"
    da.attrs["freq"] = float(freqEMG)
    da.attrs["freq_ref"] = vicon.GetFrameRate()  # frequency of markers
    da.attrs["units"] = "mV"
    da.time.attrs["units"] = "s"

    print("Cargados los datos en {0:.3f} s \n".format(time.time() - timer))
    return da


def escribe_variables_en_nexus_kinem(da, vicon) -> None:
    import itertools

    n_subject = vicon.GetSubjectNames()[0]
    num_frames = vicon.GetTrialRange()[1]
    region_of_interest = (
        np.array(vicon.GetTrialRegionOfInterest()) - 1
    )  # corrección para que ajuste a la escala empezando en cero
    exists = np.full(
        (num_frames), False, dtype=bool
    )  # pone a cero toda la variable de si existe
    # activa solo en la región de interés del trial
    exists[region_of_interest[0] : region_of_interest[1] + 1] = True

    for var, lado in itertools.product(da.n_var.values, da.side.values):
        if var in N_VARS_BILATERAL:
            lado = "LR"
        n_modeled = f"{var}_{lado}"

        # print(n_modeled)

        if lado == "LR":  # si es bilateral lo vuelve a cambiar para poder seleccionar
            lado = "L"

        if "ID" in da.dims:
            insert_values = da.isel(ID=0).sel(n_var=var, side=lado)
        else:
            insert_values = da.sel(n_var=var, side=lado)

        # Si no hay datos se sale
        if np.isnan(insert_values).all():
            continue

        # Se asegura de que los ejes estén bien orientados
        insert_values = insert_values.transpose(..., "axis", "time")

        if n_modeled not in vicon.GetModelOutputNames(n_subject):
            vicon.CreateModelOutput(
                n_subject,
                n_modeled,
                "Modeled Angles",
                ["x", "y", "z"],
                ["Angle", "Angle", "Angle"],
            )
        vicon.SetModelOutput(n_subject, n_modeled, insert_values.values, exists)


def escribe_variables_en_nexus_forces(da=None, vicon=None) -> None:
    # Escribe Fuerzas tratadas de vuelta al Nexus
    # Get ModelOutput List
    n_subject = vicon.GetSubjectNames()[0]
    num_frames = vicon.GetTrialRange()[1]
    region_of_interest = (
        np.array(vicon.GetTrialRegionOfInterest()) - 1
    )  # corrección para que ajuste a la escala empezando en cero
    exists = np.full(
        (num_frames), False, dtype=bool
    )  # pone a cero toda la variable de si existe
    # activa solo en la región de interés del trial
    exists[region_of_interest[0] : region_of_interest[1] + 1] = True

    full_model_output_list = vicon.GetModelOutputNames(n_subject)

    x2 = da.time.data[
        :: int(da.freq / da.freq_ref)
    ]  # list(np.array(np.linspace(region_of_interest[0], len(x1), num_frames)))
    da_subsamp = da.interp(time=x2, method="cubic")

    # musc=['GLU', 'BIC', 'VME', 'REC', 'TIB', 'GAS']

    modeled_name = "Forces"
    var_model = da_subsamp.isel(ID=0).data

    if modeled_name not in vicon.GetModelOutputNames(n_subject):
        vicon.CreateModelOutput(
            n_subject,
            modeled_name,
            "Modeled Forces",
            ["x", "y", "z"],
            ["Force", "Force", "Force"],
        )
    vicon.SetModelOutput(n_subject, modeled_name, var_model, exists)


def escribe_variables_en_nexus_emg(da=None, vicon=None) -> None:
    # Escribe EMG tratada de vuelta al Nexus
    # Get ModelOutput List

    n_subject = vicon.GetSubjectNames()[0]
    num_frames = vicon.GetTrialRange()[1]
    region_of_interest = (
        np.array(vicon.GetTrialRegionOfInterest()) - 1
    )  # corrección para que ajuste a la escala empezando en cero
    exists = np.full(
        (num_frames), False, dtype=bool
    )  # pone a cero toda la variable de si existe
    # activa solo en la región de interés del trial
    exists[region_of_interest[0] : region_of_interest[1] + 1] = True

    full_model_output_list = vicon.GetModelOutputNames(n_subject)

    x2 = da.time.data[
        :: int(da.freq / da.freq_ref)
    ]  # list(np.array(np.linspace(region_of_interest[0], len(x1), num_frames)))
    da_subsamp = da.interp(time=x2, method="cubic")

    # musc=['GLU', 'BIC', 'VME', 'REC', 'TIB', 'GAS']

    for modeled_name in da.channel.data:
        var_model = (
            da_subsamp.isel(ID=0).sel(channel=modeled_name).data / 1000
        )  # vuelve a pasarlo a voltios, que es la unidad del Nexus

        if modeled_name not in vicon.GetModelOutputNames(n_subject):
            vicon.CreateModelOutput(
                n_subject,
                modeled_name,
                "Modeled EMG (raw)",
                ["EMG"],
                ["Electric Potential"],
            )
        vicon.SetModelOutput(n_subject, modeled_name, [var_model], exists)


# =============================================================================
# %% Extrae variables del Nexus desde csv o c3d
# =============================================================================
def pasa_df_a_da_EMG(data) -> xr.DataArray:
    # Pasa de df a da
    if isinstance(data, pd.DataFrame):
        da = (
            data.set_index(["ID", "time"])
            .stack()
            .to_xarray()
            .rename({"level_2": "n_var"})
        )  # .transpose('ID', 'n_var', 'axis', 'time')
    elif isinstance(data, xr.DataArray):
        da = data

    L = da.sel(
        n_var=da.n_var.str.endswith("_L")
    )  # [i for i in da.n_var.data if '_L' in i and '_LR' not in i])#nomVarsContinuas_L) #el L es distinto porque incluye _L y _LR
    R = da.sel(n_var=da["n_var"].str.endswith("_R"))

    # Quita las terminaciones después de _
    L["n_var"] = ("n_var", L["n_var"].str.rstrip(to_strip="_L").data)
    R["n_var"] = ("n_var", R["n_var"].str.rstrip(to_strip="_R").data)

    da = xr.concat([L, R], pd.Index(["L", "R"], name="side")).transpose(
        "ID", "n_var", "side", "time"
    )
    return da


# =============================================================================


# =============================================================================
# %% Carga trayectorias desde c3d
# =============================================================================
def carga_trayectorias_c3d(lista_archivos, nom_vars_cargar=None):
    raise Exception(
        "Deprecation warning. Mejor utilizar carga_c3d_generico_xr(data, section='Trajectories')"
    )
    # return

    import c3d

    print("Cargando los archivos...")
    timer = time.time()  # inicia el contador de tiempo

    daTodosArchivos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    numArchivosProcesados = 0
    for file in lista_archivos:
        # se asegura de que la extensión es c3d
        file = file.with_suffix(".c3d")
        print(file.name)
        try:
            timerSub = time.time()  # inicia el contador de tiempo
            print("Cargando archivo: {0:s}".format(file.name))

            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freq = reader.point_rate

                points = []
                for i, (_, p, _) in enumerate(reader.read_frames()):
                    points.append(p)
                    # analog.append(a)
                    if not i % 10000 and i:
                        print("Extracted %d point frames", len(points))

                labels = [s.replace(" ", "") for s in reader.point_labels]
            data = np.asarray(points)[:, :, :3]

            # Ajusta las etiquetas a formato lados L, R
            n_var_nuevo = [
                i.split("Left_")[-1] + "_L" if "Left" in i else i for i in labels
            ]
            n_var_nuevo = [
                i.split("Right_")[-1] + "_R" if "Right" in i else i for i in n_var_nuevo
            ]
            n_var_nuevo = [i[1:] + "_L" if i[0] == "L" else i for i in n_var_nuevo]
            n_var_nuevo = [i[1:] + "_R" if i[0] == "R" else i for i in n_var_nuevo]

            da = xr.DataArray(
                data=np.expand_dims(data, axis=0) / 10,  # pasado a centímetros
                dims=("ID", "time", "n_var", "axis"),
                coords={
                    "ID": [file.parent.parts[-2] + "_" + file.stem],
                    "time": (np.arange(0, data.shape[0]) / freq),
                    "n_var": (n_var_nuevo),
                    "axis": (["x", "y", "z"]),
                },
                name="Trayectorias",
                attrs={
                    "freq": freq,
                    "units": "cm",
                },
            ).transpose("ID", "n_var", "axis", "time")
            da.time.attrs["units"] = "s"
            # Se queda solo con las trayectorias
            da = da.sel(n_var=~da.n_var.str.contains("USERMO"))
            # da.isel(ID=0, axis=0).plot.line(x='time', hue='n_var')

            daTodosArchivos.append(da)
            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(os.path.basename(file.name) + " " + str(err))
            continue

    daTodosArchivos = xr.concat(
        daTodosArchivos, dim="ID"
    )  # .transpose('ID', 'n_var', 'side', 'axis', 'time')

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timer
        )
    )
    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido procesar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])
    # *******************************************************

    if nom_vars_cargar:
        daTodosArchivos = daTodosArchivos.sel(n_var=nom_vars_cargar)

    # daTodosArchivos.isel(ID=0).sel(n_var=['HJC', 'KJC', 'AJC']).plot.line(x='time', col='side', hue='n_var')

    """
    #Añade coordenada con nombre del tipo de test. Diferencia entre MVC y dinámicos, por el criterio de nombrado
    if daTodosArchivos.ID[0].str.contains('MVC'):
        lista_coords = daTodosArchivos.ID.to_series().str.split('_').str[-1].str.split('-').str[-2]
    else:
        lista_coords = daTodosArchivos.ID.to_series().str.split('_').str[-1].str.split('-').str[0]
    daTodosArchivos = daTodosArchivos.assign_coords(test=('ID', lista_coords))
    """

    return daTodosArchivos


def separa_trayectorias_lados(daData) -> xr.DataArray:
    L = daData.sel(n_var=daData.n_var.str.endswith("_L"))
    R = daData.sel(n_var=daData["n_var"].str.endswith("_R"))
    LR = daData.sel(n_var=daData["n_var"].str.endswith("_LR"))
    # [i for i in da.n_var.data if '_L' in i and '_LR' not in i])#nomVarsContinuas_L) #el L es distinto porque incluye _L y _LR
    # daTodosArchivos.sel(n_var=list(daTodosArchivos.n_var[daTodosArchivos['n_var'].str.endswith('_LR').data].data))

    # Quita las terminaciones después de _
    L["n_var"] = ("n_var", L["n_var"].str.rstrip(to_strip="_L").data)
    R["n_var"] = ("n_var", R["n_var"].str.rstrip(to_strip="_R").data)
    LR["n_var"] = ("n_var", LR["n_var"].str.rstrip(to_strip="_LR").data)

    # Integra LR en L y R (para no gastar memoria en dimensión side coordenada LR)
    L = xr.concat([L, LR], dim="n_var")
    R = xr.concat([R, LR], dim="n_var")

    """daTodos = (xr.concat([L, R, LR], dim='side')
               .assign_coords(side=['L', 'R', 'LR'])
               .transpose('ID', 'n_var', 'side', 'axis', 'time')
               )"""
    daData = xr.concat(
        [L, R], pd.Index(["L", "R"], name="side")  # compat='no_conflicts'
    )  # .transpose('ID', 'n_var', 'side', 'axis', 'time')
    # xr.merge([L, R])
    return daData


# =============================================================================
# %%Carga en un mismo dataframe todos los archivos csv, variante cinem y EMG
# =============================================================================
def carga_csv_generico_pl_xr(
    lista_archivos, nom_vars_cargar=None, section=None
) -> xr.DataArray:
    # Versión con Polars
    print("\nCargando los archivos...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    num_archivos_procesados = 0
    dfTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daTodos = []
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(lista_archivos[:]):
        print(f"Cargando archivo nº {nf+1} / {len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            daProvis = read_vicon_csv_pl_xr(
                file, section=section, n_vars_load=nom_vars_cargar
            ).expand_dims(
                {
                    "ID": [
                        f"{file.parent.parts[-2].replace('_','')}_{'_'.join(file.stem.split('-'))}"
                    ]
                },
                axis=0,
            )  # Añade columna ID
            daTodos.append(daProvis)

            print(f"{len(daProvis.time)} filas y {len(daProvis.n_var)} columnas")
            print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))
            num_archivos_procesados += 1

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
    # dfTodos = pl.concat(dfTodos)

    print(
        f"Cargados {num_archivos_procesados} archivos en {time.perf_counter()-timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    return daTodos


def carga_c3d_generico_xr(
    lista_archivos, nom_vars_cargar=None, section=None
) -> xr.DataArray:
    print("\nCargando los archivos...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    num_archivos_procesados = 0
    dfTodos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daTodos = []
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(lista_archivos[:]):
        print(f"Cargando archivo nº {nf+1} / {len(lista_archivos)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            daProvis = read_vicon_c3d_xr(
                file, section=section, n_vars_load=nom_vars_cargar
            ).expand_dims(
                {
                    "ID": [
                        f"{file.parent.parts[-2].replace('_','')}_{'_'.join(file.stem.split('-'))}"
                    ]
                },
                axis=0,
            )  # Añade columna ID
            # daProvis.isel(ID=0).sel(n_var='AngArtHip_R', axis='y').plot.line(x='time')
            daTodos.append(daProvis)

            print(f"{len(daProvis.time)} filas y {len(daProvis.n_var)} columnas")
            print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))
            num_archivos_procesados += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATENCIÓN. No se ha podido procesar {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            ErroresArchivos.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    print(
        f"Cargados {num_archivos_procesados} archivos en {time.perf_counter()-timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido cargar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])

    daTodos = xr.concat(daTodos, dim="ID", coords="minimal").astype("float")

    return daTodos


def carga_preprocesa_csv_cinem(
    listaArchivos, nom_vars_cargar=None, nomArchivoPreprocesado=None
) -> xr.DataArray:
    # if nomArchivoPreprocesado==None:
    #     raise Exception('Debes indicar el nombre de los archivos preprocesados')

    # nomVarsDiscretas240 = ['FrecuenciaPedaleo_y',
    #                     'AngArtLHipPedalAnt_x', 'AngArtRHipPedalAnt_x',
    #                     'AngArtLKneePedalAnt_x', 'AngArtRKneePedalAnt_x',
    #                     'AngArtLKneeMaxExt_x', 'AngArtRKneeMaxExt_x']

    # En versiones del modelo anteriores a la 2.5.0 se ajustan los nombres de las variables para hacer facil distinción de lateralidad
    renombrar_vars_coords = {
        "AngArtLHip_x": "AngArtHip_L_x",
        "AngArtRHip_x": "AngArtHip_R_x",
        "AngArtLHip_y": "AngArtHip_L_y",
        "AngArtRHip_y": "AngArtHip_R_y",
        "AngArtLHip_z": "AngArtHip_L_z",
        "AngArtRHip_z": "AngArtHip_R_z",
        "AngArtLKnee_x": "AngArtKnee_L_x",
        "AngArtRKnee_x": "AngArtKnee_R_x",
        "AngArtLKnee_y": "AngArtKnee_L_y",
        "AngArtRKnee_y": "AngArtKnee_R_y",
        "AngArtLKnee_z": "AngArtKnee_L_z",
        "AngArtRKnee_z": "AngArtKnee_R_z",
        "AngArtLAnkle_x": "AngArtAnkle_L_x",
        "AngArtLAnkle_y": "AngArtAnkle_L_y",
        "AngArtLAnkle_z": "AngArtAnkle_L_z",
        "AngArtRAnkle_x": "AngArtAnkle_R_x",
        "AngArtRAnkle_y": "AngArtAnkle_R_y",
        "AngArtRAnkle_z": "AngArtAnkle_R_z",
        "AngArtLumbar_x": "AngArtLumbar_LR_x",
        "AngArtLumbar_y": "AngArtLumbar_LR_y",
        "AngArtLumbar_z": "AngArtLumbar_LR_z",
        "AngSegPELVIS_x": "AngSegPELVIS_LR_x",
        "AngSegPELVIS_y": "AngSegPELVIS_LR_y",
        "AngSegPELVIS_z": "AngSegPELVIS_LR_z",
        "AngSegTORAX_x": "AngSegTORAX_LR_x",
        "AngSegTORAX_y": "AngSegTORAX_LR_y",
        "AngSegTORAX_z": "AngSegTORAX_LR_z",
        "AngArtCuello_x": "AngArtCuello_LR_x",
        "AngArtCuello_y": "AngArtCuello_LR_y",
        "AngArtCuello_z": "AngArtCuello_LR_z",
        "AngArtL1_x": "AngArtL1_LR_x",
        "AngArtL1_y": "AngArtL1_LR_y",
        "AngBiela_y": "AngBiela_LR_y",
        "vAngBiela": "vAngBiela_LR_x",
    }
    renombrar_vars = {
        "AngArtLHip": "AngArtHip_L",
        "AngArtRHip": "AngArtHip_R",
        "AngArtLKnee": "AngArtKnee_L",
        "AngArtRKnee": "AngArtKnee_R",
        "AngArtLAnkle": "AngArtAnkle_L",
        "AngArtRAnkle": "AngArtAnkle_R",
        "AngArtLumbar": "AngArtLumbar_LR",
        "AngSegPELVIS": "AngSegPELVIS_LR",
        "AngSegTORAX": "AngSegTORAX_LR",
        "AngArtCuello": "AngArtCuello_LR",
        "AngArtL1": "AngArtL1_LR",
        "AngBiela": "AngBiela_LR",
        "vAngBiela": "vAngBiela_LR",
        "LHJC": "HJC_L",
        "RHJC": "HJC_R",
        "LKJC": "KJC_L",
        "RKJC": "KJC_R",
        "LAJC": "AJC_L",
        "RAJC": "AJC_R",
        "LPosPedal": "PosPedal_L",
        "RPosPedal": "PosPedal_R",
        "X": "x",
        "Y": "y",
        "Z": "z",
    }

    print("Cargando los archivos...")
    timer = time.time()  # inicia el contador de tiempo
    # nomVarsACargar = nomVarsContinuas#+nomVarsDiscretas

    dfTodosArchivos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    numArchivosProcesados = 0
    for file in listaArchivos:
        # print(file.name)
        try:
            timerSub = time.time()  # inicia el contador de tiempo
            print("Cargando archivo: {0:s}".format(file.name))
            dfprovis, freq = read_vicon_csv(
                file, nomBloque="Model Outputs", returnFreq=True, header_format="noflat"
            )
            # dfprovis, daprovis, freq = read_vicon_csv(file, nomBloque='Model Outputs', returnFreq=True, formatoxArray=True)

            dfprovis = (
                dfprovis.loc[
                    :, ~dfprovis.columns.duplicated()
                ]  # quita duplicados (aparecen en centros articulares)
                .rename(columns=renombrar_vars)  # , inplace=True)
                .sort_index()
            )

            if nom_vars_cargar:
                dfprovis = dfprovis[nom_vars_cargar]

            # Duplica AngBiela para _L, _R y _LR
            dfprovis = pd.concat(
                [
                    dfprovis,
                    dfprovis[["AngBiela_LR"]].rename(
                        columns={"AngBiela_LR": "AngBiela_L"}
                    ),
                    dfprovis[["AngBiela_LR"]].rename(
                        columns={"AngBiela_LR": "AngBiela_R"}
                    ),
                ],
                axis=1,
            )

            # Ajusta el AngBiela R para ser como el L pero con diferencia de 180º
            dfprovis.loc[:, ("AngBiela_R", "x")] = (
                dfprovis.loc[:, ("AngBiela_R", "x")].where(
                    dfprovis.loc[:, ("AngBiela_R", "x")] < np.pi,
                    dfprovis.loc[:, ("AngBiela_R", "x")] - 2 * np.pi,
                )
                + np.pi
            )
            dfprovis.loc[:, ("AngBiela_R", "y")] = (
                dfprovis.loc[:, ("AngBiela_R", "x")] - np.pi
            )  # dfprovis.loc[:, ('AngBiela_R', 'y')].where(dfprovis.loc[:, ('AngBiela_R', 'y')]<0.0, dfprovis.loc[:, ('AngBiela_R', 'y')]-2*np.pi)+np.pi
            dfprovis.loc[:, ("AngBiela_R", "z")] = (
                dfprovis.loc[:, ("AngBiela_R", "z")].where(
                    dfprovis.loc[:, ("AngBiela_R", "z")] < 0.0,
                    dfprovis.loc[:, ("AngBiela_R", "z")] - 2 * np.pi,
                )
                + np.pi
            )

            ######
            """
            #calcula velocidad angular biela y la añade
            #primero calcula el ángulo unwrapeado        
            if dfprovis['AngBiela']['y'].isnull()[0]: #comprueba si el primer valor es nulo, y le asigna un valor siguiendo la tendencia
                dfprovis.loc[0,('AngBiela','y')] = dfprovis.loc[1,('AngBiela','y')] - (dfprovis.loc[2,('AngBiela','y')]-dfprovis.loc[1,('AngBiela','y')])
            AngBielaUnwrap = np.unwrap(dfprovis[('AngBiela','y')])
            vAngBiela = np.gradient(AngBielaUnwrap)/(1/freq)
            """

            # Añade columna ID y time
            dfprovis.insert(
                0, "ID", [file.parent.parts[-2] + "_" + file.stem] * len(dfprovis)
            )
            dfprovis.insert(
                1, "time", np.arange(len(dfprovis))[0 : len(dfprovis)] / freq
            )  # la parte final es para asegurarse de que se queda con el tamaño adecuado

            dfTodosArchivos.append(dfprovis)

            # dfTodosArchivos.append(dfprovis.assign(**{'ID' : file.parent.parts[-2]+'_'+file.stem, #adaptar esto según la estructura de carpetas
            #                                           #'vAngBiela' : vAngBiela,
            #                                           'time' : np.arange(0, len(dfprovis)/freq, 1/freq)[0:len(dfprovis)] #la parte final es para asegurarse de que se queda con el tamaño adecuado
            #                                          }))#.reset_index(drop=True))

            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(os.path.basename(file.name) + " " + str(err))
            continue

    dfTodosArchivos = pd.concat(dfTodosArchivos)

    # dfTodosArchivos.loc[:,(slice(None), '')]

    # dfTodosArchivos.rename(columns={'':'x'}, inplace=True)

    """
    dfTodosArchivos.loc[:, ('vAngBiela','')].rename({'':'x'})#, inplace=True)
    dfTodosArchivos.loc[:, ('vAngBiela','')].columns=pd.MultiIndex.from_frame(('vAngBiela','x'))
    dfTodosArchivos['vAngBiela'].name=('vAngBiela','x')
    dfTodosArchivos.loc[:, ('vAngBiela','')].rename(('vAngBiela','x'), inplace=True)
    """
    # dfTodosArchivos['SujID'] = dfTodosArchivos['ID'].str.split('_', expand=True)[0]
    # dfTodosArchivos['Grupo'] = dfTodosArchivos['ID'].str.split('_', expand=True)[2]

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timer
        )
    )
    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido procesar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])
    # *******************************************************

    # Reordena columnas
    # if nom_vars_cargar==None: #Se queda con todas las variables
    #     nom_vars_cargar = dfTodosArchivos.columns.to_list()

    # dfTodosArchivos = dfTodosArchivos.reindex(columns=['ID', 'time'] + nom_vars_cargar, level=0)
    # -------------------------------
    # Ahora traspasa todos los datos a dadaArray separando en L, R y LR. Se podría hacer que vaya haciendo los cortes antes de juntar L, R y LR
    # df = dfTodosArchivos.set_index([('ID','x'), ('time','x')]).stack()
    # df = dfTodosArchivos.set_index(['ID', 'time']).stack().to_xarray().to_array().rename({'variable':'n_var'})
    # df.index.rename(['ID', 'time' , 'axis'], inplace=True)
    # df = df.reorder_levels([0,2,1], axis='index')

    # ----------------
    """
    #Duplica la variable usada para hacer los cortes para que funcione el segmenta_xr
    import itertools
    dfprovis = dfTodosArchivos.copy()
    for bloque in itertools.product(['AngBiela_L', 'AngBiela_R', 'AngBiela_LR'], ['x', 'y', 'z']):
        dfprovis.loc[:, bloque] = dfTodosArchivos.loc[:, ('AngBiela_LR', 'y')]
    """
    # ----------------

    daTodos = (
        dfTodosArchivos.set_index(["ID", "time"])
        .stack()
        .to_xarray()
        .to_array()
        .rename({"variable": "n_var"})
    )  # .transpose('ID', 'n_var', 'axis', 'time')
    # daTodos.isel(ID=0).sel(n_var='AngBiela_LR')

    # da = df.to_xarray().to_array().rename({'variable':'n_var'})#.transpose('ID', 'variable', 'axis', 'time').rename({'variable':'n_var'})
    L = daTodos.sel(
        n_var=daTodos.n_var.str.endswith("_L")
    )  # [i for i in da.n_var.data if '_L' in i and '_LR' not in i])#nomVarsContinuas_L) #el L es distinto porque incluye _L y _LR
    R = daTodos.sel(n_var=daTodos["n_var"].str.endswith("_R"))
    LR = daTodos.sel(
        n_var=list(daTodos.n_var[daTodos["n_var"].str.endswith("_LR").data].data)
    )

    # Quita las terminaciones después de _
    L["n_var"] = ("n_var", L["n_var"].str.rstrip(to_strip="_L").data)
    R["n_var"] = ("n_var", R["n_var"].str.rstrip(to_strip="_R").data)
    LR["n_var"] = ("n_var", LR["n_var"].str.rstrip(to_strip="_LR").data)

    """daTodos = (xr.concat([L, R, LR], dim='side')
               .assign_coords(side=['L', 'R', 'LR'])
               .transpose('ID', 'n_var', 'side', 'axis', 'time')
               )"""
    daTodos = xr.concat([L, R, LR], pd.Index(["L", "R", "LR"], name="side")).transpose(
        "ID", "n_var", "side", "axis", "time"
    )
    try:
        daTodos.loc[
            dict(n_var=["HJC", "KJC", "AJC"])
        ] /= 10  # pasa las posiciones de los ejes a cm
    except:
        pass
    # daTodos.isel(ID=0).sel(n_var='AngBiela')

    # Calcula vAngBiela
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
    # Añade coordenada con nombre del tipo de test
    daTodos = daTodos.assign_coords(
        test=("ID", daTodos.ID.to_series().str.split("_").str[-1].str.split("-").str[0])
    )

    daTodos.name = "Cinem"
    daTodos.attrs["freq"] = float(freq)
    daTodos.attrs["units"] = "deg"
    daTodos.time.attrs["units"] = "s"

    if False:
        vAngBiela.sel(n_var="vAngBiela").plot.line(
            x="time", row="ID", col="side", hue="axis", sharey=False
        )
        daTodos.sel(n_var="vAngBiela").plot.line(
            x="time", row="ID", col="side", hue="axis", sharey=False
        )

        daTodos.isel(ID=slice(0, -1)).sel(
            n_var="AngArtKnee", axis="x", side=["L", "R"]
        ).plot.line(x="time", col="side")

        daTodos.isel(ID=slice(0, -1)).sel(n_var="AngBiela", side="LR").plot.line(
            x="time", col="axis", hue="ID"
        )
        daTodos.isel(ID=slice(2, 4)).sel(n_var="vAngBiela", side="LR").plot.line(
            x="time", col="axis", hue="ID", sharey=False
        )
    # -------------------------------

    # Pone el df en formato 1 nivel encabezados
    dfTodosArchivos.columns = dfTodosArchivos.columns.map("_".join).str.strip()
    dfTodosArchivos = dfTodosArchivos.rename(columns={"ID_": "ID", "time_": "time"})
    # EL DATAFRAME VA SIN vAngBiela

    return dfTodosArchivos, daTodos


def carga_preprocesa_c3d_cinem(listaArchivos, nom_vars_cargar=None) -> xr.DataArray:
    import c3d

    print("Cargando los archivos...")
    timer = time.time()  # inicia el contador de tiempo

    daTodosArchivos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    numArchivosProcesados = 0
    for file in listaArchivos:
        # cambia extensión de csv a c3d
        file = file.with_suffix(".c3d")
        print(file.name)
        try:
            timerSub = time.time()  # inicia el contador de tiempo
            print("Cargando archivo: {0:s}".format(file.name))

            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freq = reader.point_rate

                points = []
                for i, (_, p, _) in enumerate(reader.read_frames()):
                    points.append(p)
                    # analog.append(a)
                    if not i % 10000 and i:
                        print("Extracted %d point frames", len(points))

                labels = [s.replace(" ", "") for s in reader.point_labels]
            data = np.concatenate(points, axis=1)

            # data=np.asarray(analog)#[:,:,:3]

            da = xr.DataArray(
                data=np.expand_dims(data, axis=0) / 10,
                dims=("ID", "n_var", "time"),
                coords={
                    "ID": [file.parent.parts[-2] + "_" + file.stem],
                    "n_var": (labels),
                    "time": (np.arange(0, data.shape[1]) / freq),
                },
                name="Cinem",
                attrs={
                    "freq": float(freq),
                    "units": "deg",
                },
            )
            da.time.attrs["units"] = "s"
            # da.isel(ID=0).plot.line(x='time', hue='n_var')

            if nom_vars_cargar:
                da = da.sel(n_var=nom_vars_cargar)

            renombrar_vars = {
                "EMG1": "GLU_R",
                "EMG2": "BIC_R",
                "EMG3": "REC_R",
                "EMG4": "VME_R",
                "EMG5": "GAS_R",
                "EMG6": "TIB_R",
                "EMG7": "GLU_L",
                "EMG8": "BIC_L",
                "EMG9": "REC_L",
                "EMG10": "VME_L",
                "EMG11": "GAS_L",
                "EMG12": "TIB_L",
            }
            # TODO: De momento no deja reemplazar nombre por nombre, hay que confiar en que respete el orden
            da = da.assign_coords(
                n_var=(
                    [
                        "GLU_R",
                        "BIC_R",
                        "REC_R",
                        "VME_R",
                        "GAS_R",
                        "TIB_R",
                        "GLU_L",
                        "BIC_L",
                        "REC_L",
                        "VME_L",
                        "GAS_L",
                        "TIB_L",
                    ]
                )
            )

            daTodosArchivos.append(da)
            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(os.path.basename(file.name) + " " + str(err))
            continue

    daTodosArchivos = xr.concat(
        daTodosArchivos, dim="ID"
    )  # .transpose('ID', 'n_var', 'side', 'axis', 'time')

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timer
        )
    )
    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido procesar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])
    # *******************************************************

    daTodosArchivos = pasa_df_a_da_EMG(daTodosArchivos)
    # Añade coordenada con nombre del tipo de test. Diferencia entre MVC y dinámicos, por el criterio de nombrado
    if daTodosArchivos.ID[0].str.contains("MVC"):
        lista_coords = (
            daTodosArchivos.ID.to_series().str.split("_").str[-1].str.split("-").str[-2]
        )
    else:
        lista_coords = (
            daTodosArchivos.ID.to_series().str.split("_").str[-1].str.split("-").str[0]
        )
    daTodosArchivos = daTodosArchivos.assign_coords(test=("ID", lista_coords))

    return daTodosArchivos


# ------------------------------
def carga_preprocesa_csv_EMG_pl_xr(
    listaArchivos, nom_vars_cargar=None, nomBloque="Devices", freqEMG=None
) -> xr.DataArray:
    # Versión Polars SIN TERMINAR DE ADAPTAR!
    print("Cargando los archivos...")
    timer = time.time()  # inicia el contador de tiempo

    dfTodosArchivos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    numArchivosProcesados = 0
    for file in listaArchivos:
        print(file.name)
        try:
            timerSub = time.time()  # inicia el contador de tiempo
            print("Cargando archivo: {0:s}".format(file.name))

            dfprovis = read_vicon_csv_pl_xr(file, section=nomBloque, to_dataarray=False)

            freq = 1 / dfprovis[2, "time"] - dfprovis[0, "time"]

            if nom_vars_cargar:
                dfprovis = dfprovis[nom_vars_cargar]

            if nomBloque == "Devices":
                renombrar_vars = {
                    "EMG1": "GLU_R",
                    "EMG2": "BIC_R",
                    "EMG3": "REC_R",
                    "EMG4": "VME_R",
                    "EMG5": "GAS_R",
                    "EMG6": "TIB_R",
                    "EMG7": "GLU_L",
                    "EMG8": "BIC_L",
                    "EMG9": "REC_L",
                    "EMG10": "VME_L",
                    "EMG11": "GAS_L",
                    "EMG12": "TIB_L",
                }
                dfprovis = (
                    dfprovis.rename(columns=renombrar_vars)
                    .sort_index()
                    .iloc[:, : len(renombrar_vars)]
                )

            elif nomBloque == "Model Outputs":
                # Añade el ángulo de la biela para lado L y R interpolando de marcadores a EMG
                angBiela = dfprovis["AngBiela"]["y"]
                angBiela = np.interp(
                    np.arange(len(dfprovis) * freqEMG / freq) / freqEMG,
                    np.arange(len(angBiela)) / freq,
                    angBiela,
                )

                dPedal_z = dfprovis["PosPedal_R"]["z"] - dfprovis["PosPedal_L"]["z"]
                dPedal_z = np.interp(
                    np.arange(len(dfprovis) * freqEMG / freq) / freqEMG,
                    np.arange(len(dPedal_z)) / freq,
                    dPedal_z,
                )

                dfprovis = pd.DataFrame(
                    np.asarray([angBiela, angBiela, dPedal_z, dPedal_z]).T,
                    columns=["AngBiela_L", "AngBiela_R", "dPedal_z_L", "dPedal_z_R"],
                )
                freq = freqEMG

            # Añade columna ID y time
            dfprovis.insert(
                0, "ID", [file.parent.parts[-2] + "_" + file.stem] * len(dfprovis)
            )
            # dfprovis.insert(0, 'ID', [file.stem]*len(dfprovis))
            if "time" not in dfprovis.columns:
                dfprovis.insert(
                    1, "time", np.arange(len(dfprovis))[0 : len(dfprovis)] / freq
                )  # la parte final es para asegurarse de que se queda con el tamaño adecuado

            dfTodosArchivos.append(dfprovis)

            # dfTodosArchivos.append(dfprovis.assign(**{'ID' : file.parent.parts[-2]+'_'+file.stem, #adaptar esto según la estructura de carpetas
            #                                           #'vAngBiela' : vAngBiela,
            #                                           'time' : np.arange(0, len(dfprovis)/freq, 1/freq)[0:len(dfprovis)] #la parte final es para asegurarse de que se queda con el tamaño adecuado
            #                                          }))#.reset_index(drop=True))

            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(os.path.basename(file.name) + " " + str(err))
            continue

    dfTodosArchivos = pd.concat(dfTodosArchivos)

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timer
        )
    )
    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido procesar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])
    # *******************************************************
    # ----------------
    daTodos = pasa_df_a_da_EMG(dfTodosArchivos)
    # Añade coordenada con nombre del tipo de test
    daTodos = daTodos.assign_coords(
        test=("ID", daTodos.ID.to_series().str.split("_").str[-1].str.split("-").str[0])
    )

    daTodos = daTodos * 1000  # pasa a milivoltios
    daTodos.name = "EMG"
    daTodos.attrs["freq"] = float(freq)
    daTodos.attrs["units"] = "mV"
    daTodos.time.attrs["units"] = "s"
    # daTodos.isel(ID=0).sel(n_var='GLU')

    # daTodos.isel(ID=slice(0,-1)).plot.line(x='time', col='side', row='n_var', sharey=False)

    # -------------------------------

    return daTodos  # , dfTodosArchivos, float(freq)


def carga_preprocesa_csv_EMG(
    listaArchivos, nom_vars_cargar=None, nomBloque="Devices", freqEMG=None
) -> xr.DataArray:

    print("Cargando los archivos...")
    timer = time.time()  # inicia el contador de tiempo

    dfTodosArchivos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    numArchivosProcesados = 0
    for file in listaArchivos:
        print(file.name)
        try:
            timerSub = time.time()  # inicia el contador de tiempo
            print("Cargando archivo: {0:s}".format(file.name))

            dfprovis, freq = read_vicon_csv(
                file, nomBloque=nomBloque, returnFreq=True, header_format="noflat"
            )
            # dfprovis, daprovis, freq = read_vicon_csv(file, nomBloque='Model Outputs', returnFreq=True, formatoxArray=True)
            if nom_vars_cargar:
                dfprovis = dfprovis[nom_vars_cargar]

            if nomBloque == "Devices":
                renombrar_vars = {
                    "EMG1": "GLU_R",
                    "EMG2": "BIC_R",
                    "EMG3": "REC_R",
                    "EMG4": "VME_R",
                    "EMG5": "GAS_R",
                    "EMG6": "TIB_R",
                    "EMG7": "GLU_L",
                    "EMG8": "BIC_L",
                    "EMG9": "REC_L",
                    "EMG10": "VME_L",
                    "EMG11": "GAS_L",
                    "EMG12": "TIB_L",
                }
                dfprovis = (
                    dfprovis.rename(columns=renombrar_vars)
                    .sort_index()
                    .iloc[:, : len(renombrar_vars)]
                )

            elif nomBloque == "Model Outputs":
                # Añade el ángulo de la biela para lado L y R interpolando de marcadores a EMG
                angBiela = dfprovis["AngBiela"]["y"]
                angBiela = np.interp(
                    np.arange(len(dfprovis) * freqEMG / freq) / freqEMG,
                    np.arange(len(angBiela)) / freq,
                    angBiela,
                )

                dPedal_z = dfprovis["PosPedal_R"]["z"] - dfprovis["PosPedal_L"]["z"]
                dPedal_z = np.interp(
                    np.arange(len(dfprovis) * freqEMG / freq) / freqEMG,
                    np.arange(len(dPedal_z)) / freq,
                    dPedal_z,
                )

                dfprovis = pd.DataFrame(
                    np.asarray([angBiela, angBiela, dPedal_z, dPedal_z]).T,
                    columns=["AngBiela_L", "AngBiela_R", "dPedal_z_L", "dPedal_z_R"],
                )
                freq = freqEMG

            # Añade columna ID y time
            dfprovis.insert(
                0, "ID", [file.parent.parts[-2] + "_" + file.stem] * len(dfprovis)
            )
            # dfprovis.insert(0, 'ID', [file.stem]*len(dfprovis))
            if "time" not in dfprovis.columns:
                dfprovis.insert(
                    1, "time", np.arange(len(dfprovis))[0 : len(dfprovis)] / freq
                )  # la parte final es para asegurarse de que se queda con el tamaño adecuado

            dfTodosArchivos.append(dfprovis)

            # dfTodosArchivos.append(dfprovis.assign(**{'ID' : file.parent.parts[-2]+'_'+file.stem, #adaptar esto según la estructura de carpetas
            #                                           #'vAngBiela' : vAngBiela,
            #                                           'time' : np.arange(0, len(dfprovis)/freq, 1/freq)[0:len(dfprovis)] #la parte final es para asegurarse de que se queda con el tamaño adecuado
            #                                          }))#.reset_index(drop=True))

            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(os.path.basename(file.name) + " " + str(err))
            continue

    dfTodosArchivos = pd.concat(dfTodosArchivos)

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timer
        )
    )
    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido procesar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])
    # *******************************************************
    # ----------------
    daTodos = pasa_df_a_da_EMG(dfTodosArchivos)
    # Añade coordenada con nombre del tipo de test
    daTodos = daTodos.assign_coords(
        test=("ID", daTodos.ID.to_series().str.split("_").str[-1].str.split("-").str[0])
    )

    daTodos = daTodos * 1000  # pasa a milivoltios
    daTodos.name = "EMG"
    daTodos.attrs["freq"] = float(freq)
    daTodos.attrs["units"] = "mV"
    daTodos.time.attrs["units"] = "s"
    # daTodos.isel(ID=0).sel(n_var='GLU')

    # daTodos.isel(ID=slice(0,-1)).plot.line(x='time', col='side', row='n_var', sharey=False)

    # -------------------------------

    return daTodos  # , dfTodosArchivos, float(freq)


def carga_preprocesa_c3d_EMG(
    listaArchivos, nom_vars_cargar=None, nomBloque="Devices", freqEMG=None
) -> xr.DataArray:
    import c3d

    print("Cargando los archivos...")
    timer = time.time()  # inicia el contador de tiempo

    daTodosArchivos = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    ErroresArchivos = (
        []
    )  # guarda los nombres de archivo que no se pueden abrir y su error
    numArchivosProcesados = 0
    for file in listaArchivos:
        # cambia extensión de csv a c3d
        file = file.with_suffix(".c3d")
        print(file.name)
        try:
            timerSub = time.time()  # inicia el contador de tiempo
            print("Cargando archivo: {0:s}".format(file.name))

            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freqEMG = reader.analog_rate
                n_var = reader.analog_labels

                analog = []
                for i, (_, _, a) in enumerate(reader.read_frames()):
                    # points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print("Extracted %d point frames", len(analog))

                labels = [
                    s.split(".")[0].replace(" ", "") for s in reader.analog_labels
                ]
            data = np.concatenate(analog, axis=1) * 1000  # pasa a milivoltios

            # data=np.asarray(analog)#[:,:,:3]

            da = xr.DataArray(
                data=np.expand_dims(data, axis=0),
                dims=("ID", "n_var", "time"),
                coords={
                    "ID": [file.parent.parts[-2] + "_" + file.stem],
                    "n_var": (labels),
                    "time": (np.arange(0, data.shape[1]) / freqEMG),
                },
                name="EMG",
                attrs={
                    "freq": freqEMG,
                    "units": "mV",
                },
            )
            da.time.attrs["units"] = "s"
            # da.isel(ID=0).plot.line(x='time', hue='n_var')

            if nom_vars_cargar:
                da = da.sel(n_var=nom_vars_cargar)

            if nomBloque == "Devices":
                renombrar_vars = {
                    "EMG1": "GLU_R",
                    "EMG2": "BIC_R",
                    "EMG3": "REC_R",
                    "EMG4": "VME_R",
                    "EMG5": "GAS_R",
                    "EMG6": "TIB_R",
                    "EMG7": "GLU_L",
                    "EMG8": "BIC_L",
                    "EMG9": "REC_L",
                    "EMG10": "VME_L",
                    "EMG11": "GAS_L",
                    "EMG12": "TIB_L",
                }
                # TODO: De momento no deja reemplazar nombre por nombre, ahy que confiar en que respete el orden
                da = da.assign_coords(
                    n_var=(
                        [
                            "GLU_R",
                            "BIC_R",
                            "REC_R",
                            "VME_R",
                            "GAS_R",
                            "TIB_R",
                            "GLU_L",
                            "BIC_L",
                            "REC_L",
                            "VME_L",
                            "GAS_L",
                            "TIB_L",
                        ]
                    )
                )

            daTodosArchivos.append(da)
            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            numArchivosProcesados += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")
            ErroresArchivos.append(os.path.basename(file.name) + " " + str(err))
            continue

    daTodosArchivos = xr.concat(
        daTodosArchivos, dim="ID"
    )  # .transpose('ID', 'n_var', 'side', 'axis', 'time')

    print(
        "Cargados {0:d} archivos en {1:.3f} s \n".format(
            numArchivosProcesados, time.time() - timer
        )
    )
    # Si no ha podido cargar algún archivo, lo indica
    if len(ErroresArchivos) > 0:
        print("\nATENCIÓN. No se ha podido procesar:")
        for x in range(len(ErroresArchivos)):
            print(ErroresArchivos[x])
    # *******************************************************

    daTodosArchivos = pasa_df_a_da_EMG(daTodosArchivos)
    # Añade coordenada con nombre del tipo de test. Diferencia entre MVC y dinámicos, por el criterio de nombrado
    if daTodosArchivos.ID[0].str.contains("MVC"):
        lista_coords = (
            daTodosArchivos.ID.to_series().str.split("_").str[-1].str.split("-").str[-2]
        )
    else:
        lista_coords = (
            daTodosArchivos.ID.to_series().str.split("_").str[-1].str.split("-").str[0]
        )
    daTodosArchivos = daTodosArchivos.assign_coords(test=("ID", lista_coords))

    return daTodosArchivos


# =============================================================================
# %% Procesa EMG
# =============================================================================
# Función para detectar onsets
"""
Ref: Solnik, S., Rider, P., Steinweg, K., Devita, P., & Hortobágyi, T. (2010). Teager-Kaiser energy operator signal conditioning improves EMG onset detection. European Journal of Applied Physiology, 110(3), 489–498. https://doi.org/10.1007/s00421-010-1521-8

Función sacada de Duarte (https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb)
The Teager-Kaiser Energy operator to improve onset detection
The Teager-Kaiser Energy (TKE) operator has been proposed to increase the accuracy of the onset detection by improving the SNR of the EMG signal (Li et al., 2007).
"""


def tkeo(x) -> np.ndarray:
    r"""Calculates the Teager-Kaiser Energy operator.

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


def procesaEMG(daEMG, fr=None, fc_band=[10, 400], fclow=8, btkeo=False) -> xr.DataArray:
    from filtrar_Butter import filtrar_Butter, filtrar_Butter_bandpass

    if fr == None:
        fr = daEMG.freq
    # Filtro band-pass
    daEMG_proces = filtrar_Butter_bandpass(
        daEMG, fr=fr, fclow=fc_band[0], fchigh=fc_band[1]
    )
    # Centra, ¿es necesario?
    daEMG_proces = daEMG_proces - daEMG_proces.mean(dim="time")

    if btkeo:
        daEMG_proces = xr.apply_ufunc(
            tkeo,
            daEMG_proces,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
        )
    # Rectifica
    daEMG_proces = abs(daEMG_proces)
    # filtro low-pass
    daEMG_proces = filtrar_Butter(daEMG_proces, fr=fr, fc=fclow, kind="low")

    # daEMG_proces.attrs['freq'] = daEMG.attrs['freq']
    # daEMG_proces.attrs['units'] = daEMG.attrs['units']
    # daEMG_proces.time.attrs['units'] = daEMG.time.attrs['units']
    daEMG_proces.attrs = daEMG.attrs
    daEMG_proces.name = "EMG"

    return daEMG_proces


# =============================================================================
# %% Segmenta por repeticiones, versión cinem y EMG
# =============================================================================


def segmenta_ModeloBikefitting_xr_cinem(
    daDatos, num_cortes=12, graficas=False
) -> xr.DataArray:
    # from detecta import detect_peaks
    # from cortar_repes_ciclicas import detect_onset_aux

    print(f"Segmentando {len(daDatos.ID)} archivos.")
    # Es necesario separar lado L y R porque usan criterios distintos de corte
    ###CORTES A PARTIR DE ANG BIELA
    print("Cortando lado L...")
    daL = stsp(
        data=daDatos.sel(side="L"),
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela", axis="y"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=1, show=graficas),
    ).slice_time_series()
    print("Cortando lado R...")
    daR = stsp(
        data=daDatos.sel(side="R"),
        func_events=stsp.detect_onset_detecta_aux,
        reference_var=dict(n_var="AngBiela", axis="y"),
        discard_phases_ini=1,
        discard_phases_end=0,
        n_phases=num_cortes,
        include_first_next_last=True,
        **dict(threshold=0.0, n_above=2, event_ini=0, show=graficas),
    ).slice_time_series()
    """
    print('Cortando lados R y LR...')
    daR = stsp(data=daDatos.sel(side=['R', 'LR']), func_events=stsp.detect_onset_detecta_aux,
              reference_var=dict(n_var='AngBiela', side='LR', axis='y'),
              discard_phases_ini=1, discard_phases_end=0, n_phases=num_cortes,
              include_first_next_last=True, 
              **dict(threshold=0.0, n_above=2, event_ini=0, show=graficas)
              ).slice_time_series()
    """
    # daL = cts(daDatos.sel(side='L'), func_cortes=cts.detect_onset_aux, var_referencia=dict(n_var='AngBiela', axis='y'),
    #           descarta_corte_ini=1, descarta_corte_fin=0, num_cortes=num_cortes,
    #           **dict(threshold=0.0, n_above=2, corte_ini=1, show=graficas)).corta_repes()
    # daR = cts(daDatos.sel(side=['R', 'LR']), func_cortes=cts.detect_onset_aux, var_referencia=dict(n_var='AngBiela', side='LR', axis='y'),
    #           descarta_corte_ini=1, num_cortes=num_cortes, descarta_corte_fin=0, **dict(threshold=0.0, n_above=2, corte_ini=0, show=graficas)).corta_repes()

    # daL = corta_repes_xr(daDatos.sel(side='L'), func_cortes=detect_onset_aux, var_referencia=dict(n_var='AngBiela', axis='y'), descarta_rep_ini=1, descarta_rep_fin=0, num_repes=num_repes, **dict(threshold=0.0, n_above=2, corte_ini=1, show=graficas))
    # daR = corta_repes_xr(daDatos.sel(side=['R', 'LR']), func_cortes=detect_onset_aux, var_referencia=dict(n_var='AngBiela', side='LR', axis='y'), descarta_rep_ini=1, num_repes=num_repes, descarta_rep_fin=0, **dict(threshold=0.0, n_above=2, corte_ini=0, show=graficas))

    # #Mete variables LR en lado R (antiguo)
    # daLR = daR.sel(side='LR')
    # daR = daR.sel(side='R')

    # daL.isel(ID=0).sel(n_var=['AngArtHip', 'AngArtKnee', 'AngArtAnkle']).plot.line(x='time', col='axis', row='n_var', sharey=False)
    # daR.isel(ID=0).sel(n_var=['AngArtHip', 'AngArtKnee', 'AngArtAnkle']).plot.line(x='time', col='axis', row='n_var', sharey=False)
    # daLR.isel(ID=0).dropna(dim='n_var', how='all').plot.line(x='time', col='axis', row='n_var', sharey=False)

    """
    ####CORTES A PARTIR DE DIFERENCIA PEDALES
    #Incorpora variable distancia entre pedales
    dPedales = ((daDatos.sel(n_var='PosPedal', side='R') - daDatos.sel(n_var='PosPedal', side='L'))
                .expand_dims(dim=['n_var', 'side'])
                .assign_coords(dict(n_var=['dPedales'], side=['LR']))
              )
    daDatos = xr.concat([daDatos, dPedales], dim='n_var', join='left') #join left para que guarde el orden de coords lado
    
    daL = (corta_repes_xr(daDatos.sel(side=['L', 'LR']), func_cortes=detect_peaks, var_referencia=dict(n_var='dPedales', side='LR', axis='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=True, mph=-100, mpd=50, show=graficas))
           .sel(side='L')
           )
    daR = corta_repes_xr(daDatos.sel(side=['R', 'LR']), func_cortes=detect_peaks, var_referencia=dict(n_var='dPedales', side='LR', axis='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=False, mph=100, mpd=50, show=graficas))#.sel(side='R')
    """

    """    
    daSegment = (xr.concat([daL, daR, daLR], pd.Index(['L','R', 'LR'], name='side'))
                 .transpose('ID', 'n_var', 'side', 'axis', 'phase', 'time')
                 #.isel(n_var=slice(0,-1))#quita la variable creada distancia pedales
                 )
    """

    daSegment = (
        xr.concat([daL, daR], pd.Index(["L", "R"], name="side")).transpose(
            "ID", "n_var", "side", "axis", "phase", "time"
        )
        # .isel(n_var=slice(0,-1))#quita la variable creada distancia pedales
    )

    return daSegment


# #Versión con dataframe
# def ApilaFactores_Segmenta_ModeloBikefitting_cinem(dfArchivo, graficas=False):
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
#     dfL = corta_repes(dfL, func_cortes=detect_peaks, frec=freq,  col_factores='ID', col_referencia='AngBiela_LR_y', col_variables=nomVarsContinuas_L_coord+['vAngBiela_LR_x'], descarta_rep_ini=1, descarta_rep_fin=0, incluye_primero_siguiente=True, **dict(valley=True, show=graficas))

#     #Corta derecho y LR a la vez
#     dfR = dfArchivo.drop(dfArchivo.filter(regex='|'.join(['_L_'])).columns, axis=1) #al derecho no hace falta añadir bielas
#     dfR = corta_repes(dfArchivo, func_cortes=detect_onset_aux, frec=freq,  col_factores='ID', col_referencia='AngBiela_LR_y', col_variables=nomVarsContinuas_R_coord+nomVarsContinuas_LR_coord+['vAngBiela_LR_x'], descarta_rep_ini=1, descarta_rep_fin=0, **dict(threshold=0.0, corte_ini=0, n_above=2, show=graficas))
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


# =============================================================================
# %%
# =============================================================================
if __name__ == "__main__":
    if False:
        """
        sys.path.append(r'F:\Programacion\Python\Mios\Functions')
        import Nexus_FuncionesApoyo as nfa
        """
        # Cargando directamente desde c3d
        ruta_trabajo = Path(
            r"F:\Investigacion\Proyectos\BikeFitting\Bikefitting\EstudioEMG_MVC\Registros\17_Eduardo"
        )

        lista_archivos = list(
            ruta_trabajo.glob("**/*.c3d")
        )  # incluye los que haya en subcarpetas
        lista_archivos = [
            x
            for x in lista_archivos
            if "MVC-" not in x.name and "Estatico" not in x.name
        ]
        lista_archivos.sort()

        daDatos = carga_c3d_generico_xr(
            lista_archivos[:10], section="Trajectories"
        )  # , nom_vars_cargar=['HJC', 'KJC', 'AJC'])
        daDatos = ajusta_etiquetas_lado_final(daDatos)
        daDatos = separa_trayectorias_lados(daDatos)
        # daAngles = calcula_angulos_desde_trajec(daDatos)
        # daPos = calcula_variables_posicion(daDatos)
        # daCinem = xr.concat([daAngles, daPos], dim='n_var')
