# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:05:37 2019

@author: josel
"""
from __future__ import division, print_function

"""Lee archivos de datos exportados del Vicon Nexus"""

import numpy as np
import pandas as pd
import xarray as xr
#import scipy.signal

__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.2.2.0'
__date__ = '29/03/2021'

"""
Modificaciones:
    29/03/2021, v2.1.1
            - Incluido parámetro 'header_format' para que devuelva el encabezado como 'flat' en una sola línea (variable_x, variable_y, ...) o en dos líneas ((variable,x), (variable,y), ...).
            
    28/03/2021, v2.1.1
            - Mejorada lectura con Pandas. Ahora puede cargar archivos que empiezan sin datos en las primeras líneas.

	21/03/2021, v2.1.0
            - Cambiado lector del bloque de archivos por pd.read_csv con número de columnas delimitado a los que carga en las variables (quitando los de velocidad y aceleración)
            - Solucionado fallo al leer frecuencia cuando terminaba la línea rellenando con separadores (como al exportar en Excel)

    10/01/2021, v2.0.1
            - Ajustado para que pueda devolver xArray con Model Outputs

    13/12/2020, v2.0.0
            - Con el argumento formatoxArray se puede pedir que devuelva los datos en formato xArray
"""

def read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', separador=',', returnFrec=False, formatoxArray=False, header_format='flat'):
    """    
    Parameters
    ----------
    versión : v2.2.0
    nombreArchivo : string
        ruta del archivo a abrir
        
    nomBloque : string
        tipo de datos a leer en el archivo original.
        'Model Outputs', 'Trajectories' o 'Devices'
    
    separador : string
        caracter separador de los datos
    
    returnFrec : bool
        si es True devuelve un int con la frecuencia de muestreo
    
    formatoxArray : bool
        si es true devuelve los datos en formato xArray    
    
    header_format : str
        'flat': devuelve el encabezado en una línea (por defecto)
        otra cosa: devuelve el encabezaco en dos líneas (var y coord)
        
    Returns
    -------
    data : datos leidos en formato DataFrame de Pandas o DataArray de xArray.
    frec: frecuencia de registro de los datos.
    
    
    
    Examples
    --------
    >>> dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    >>> dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    
    >>> #Con formato dataarray de xArray
    >>> daDatos = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', formatoxArray=True)

    """   
    
    with open(nombreArchivo, mode='rt') as f:
        numLinea=0
        #busca etiqueta del inicio del bloque
        linea = f.readline()
        while nomBloque not in linea:
            if linea == '':        
                raise Exception('No se ha encontrado el encabezado')
                
            numLinea+=1
            linea = f.readline()
        
        
        inicioBloque = numLinea
               
        #Lo que viene detrás de la etiqueta es la frecuencia
        linea = f.readline()
        frecuencia = int(linea.replace(separador,'')) #quita el separador para los casos en los que el archivo ha sido guardado con Excel (completa línea con separador)
        
        #Carga el nombre de las columnas
        #linea = f.readline()
        nomColsVar = str(f.readline()[:-1]).split(separador) #nombreVariables
        nomCols = str(f.readline()[:-1]).split(separador) #nombre coordenadas X,Y,Z.
        #nomCols = [s.lower() for s in nomCols] # Lo fuerza a minúsculas
        
        #busca etiqueta del final del bloque
        while linea!='\n':
            if linea == '':         
                raise Exception('No se ha encontrado el final del bloque')
                
            numLinea+=1
            #print('Linea '+ str(numLinea))
            linea = f.readline()
          
    finBloque = numLinea-1 #quita 1 para descontar la línea vacía
    
    #Cuenta el nº de líneas totales
    finArchivo=0
    with open(nombreArchivo, mode='rt') as f:
        for i in f:
            finArchivo+=1
    
    
    #primero asigna los nombres según el propio archivo
    nomVars=['Frame', 'Sub Frame']
    for i in range(2,len(nomCols),3):
        if "'" not in nomCols[i] and "''" not in nomCols[i]: #elimina las posibles columnas de velocidad y aceleración
            nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i])#X
            nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+1])#Y
            nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+2])#Z
    
    # [i for i in nomColsVar if "'" in i]
    # nomColsVar = [i for i in nomColsVar if "'" not in i]
        
   
    #carga todos los datos
    #CON GENFROMTXT FALLA SI NO EMPIEZA LA PRIMERA LÍNEA CON DATOS
    #provisional= np.genfromtxt(nombreArchivo, skip_header= inicioBloque+5, max_rows=finBloque-inicioBloque-1, delimiter=separador, missing_values='', filling_values=np.nan, invalid_raise=True)
    #provisional=provisional[:, :len(nomVars)] #recorta solo hasta las variables 
    
    #Convierte los datos en pandas dataframe. Pasa solo los que no son de velocidad o aceleración
    #dfReturn = pd.DataFrame(provisional[:, :len(nomVars)], columns=nomVars)
    #dfReturn = dfReturn.iloc[:, :len(nomVars)] #se queda solo con las columnas de las variables, quita las de velocidad si las hay
    
    
    
    #Con pandas directamente funciona (para evitar error si primera línea no son datos, lee la fina de las unidades y luego la quita)
    dfReturn = pd.read_csv(nombreArchivo, delimiter=separador, header=None, skiprows=inicioBloque+4, skipfooter=finArchivo-finBloque-5, usecols=range(len(nomVars)), engine='python')
    dfReturn = dfReturn.drop(index=0).reset_index(drop=True).astype(float) #borra la primera fila, que contiene las unidades
    
    #Nombra encabezado
    
    var=['_'.join(s.split('_')[:-1]) for s in nomVars[:len(nomVars)]] #gestiona si la variable tiene separador '_', lo mantiene
    coord=[s.split(':')[-1] for s in nomCols[:len(nomVars)]]
    dfReturn.columns=pd.MultiIndex.from_tuples(list(zip(*[var,coord])), names=['Variable', 'Coord'])
    #dfReturn.columns=[var, coord]
    #dfReturn.columns.set_names(names=['Variable', 'Coord'], level=[0,1], inplace=True)

    if header_format=='flat':
        dfReturn.columns = dfReturn.columns.map('_'.join).str.strip()
        
    # #Elimina las columnas de velocidad y aceleración, si las hay
    # borrarColsVA = dfReturn.filter(regex='|'.join(["'", "''"])).columns
    # dfReturn = dfReturn.drop(columns=borrarColsVA)
    
    #Si hace falta lo pasa a xArray
    if formatoxArray:
        daReturn=xr.DataArray()
    
        #transforma los datos en xarray
        x=dfReturn.filter(regex='|'.join(['_x','_X'])).to_numpy().T
        y=dfReturn.filter(regex='|'.join(['_y','_Y'])).to_numpy().T
        z=dfReturn.filter(regex='|'.join(['_z','_Z'])).to_numpy().T
        data=np.stack([x,y,z])
        
        #Quita el identificador de la coordenada del final
        canales = dfReturn.filter(regex='|'.join(['_x','_X'])).columns.str.rstrip('|'.join(['_x','_X']))
               
        n_frames = x.shape[1]
        channels = canales
        time = np.arange(start=0, stop=n_frames / frecuencia, step=1 / frecuencia)
        coords = {}
        coords['axis'] = ['x', 'y', 'z']
        coords['channel'] = channels
        coords['time'] = time
        
        daReturn=xr.DataArray(
                    data=data,
                    dims=('axis', 'channel', 'time'),
                    coords=coords,
                    name=nomBloque,
                    attrs={'Frec':frecuencia}
                    #**kwargs,
                )
        
            
    
    if formatoxArray and returnFrec:
        return dfReturn, daReturn, frecuencia
    elif formatoxArray:
        return dfReturn, daReturn
    elif returnFrec:
        return dfReturn, frecuencia
    else:   
        return dfReturn

# =============================================================================
# %%        
# =============================================================================
if __name__ == '__main__':
    from pathlib import Path
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN.csv'
    nombreArchivo = Path(ruta_Archivo)    
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    
        
    #Con Models al final
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN_ModeloAlFinal.csv'
    nombreArchivo = Path(ruta_Archivo)
    
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
       
    
    #Sin fila inicial en blanco
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN_SinFilaBlancoInicial.csv'
    nombreArchivo = Path(ruta_Archivo)
    
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
        
    
    #Solo bloque modelos
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN_2.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    
    
    #Con hueco muy grande al inicio
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconConHuecoInicio_S27_WHT_T2_L01.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    dfDatos['R5Meta_Z'].plot()

    #Con formato dataarray de xArray    
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN.csv'
    nombreArchivo = Path(ruta_Archivo)    
    dfDatos, daDatos = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', formatoxArray=True)
    dfDatos['Right_Toe_Z'].plot()
    daDatos.sel(channel='Right_Toe', axis='z').plot.line()
    
    
    dfDatos, daDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', formatoxArray=True)
    dfDatos['AngArtLKnee_x'].plot()
    daDatos.sel(channel='AngArtLKnee', axis='x').plot.line()
    
    
    #Archivo con huecos
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconConHuecos_S01_WHF_T1_L04.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    dfDatos.plot()
    
    #prueba con encabezado multiindex
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN.csv'
    nombreArchivo = Path(ruta_Archivo)    
    dfDatosFlat = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatosMulti = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', header_format='multi')
    
    dfDatosFlat[['AngArtLKnee_x','AngArtLKnee_y','AngArtLKnee_z']].plot()
    dfDatosMulti['AngArtLKnee'].plot()
    
    dfDatosMulti.loc[:, (slice(None), 'x')].plot() #todas las variables de una misma coordenada


    dfDatosFlat = read_vicon_csv(nombreArchivo, nomBloque='Trajectories')
    dfDatosMulti = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', header_format='multi')

    dfDatosFlat[['Right_Toe_X','Right_Toe_Y','Right_Toe_Z']].plot()
    dfDatosMulti['Right_Toe'].plot()
    
    dfDatosMulti.loc[:, (slice(None), 'Z')].plot() #todas las variables de una misma coordenada
