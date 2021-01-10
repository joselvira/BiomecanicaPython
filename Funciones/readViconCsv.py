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
__version__ = 'v.2.0.0'
__date__ = '13/12/2020'

"""
Modificaciones:
13/12/2020, v2.0.0
            - con el argumento formatoxArray se puede pedir que devuelva los datos en formato xArray
"""

def read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', separador=',', returnFrec=False, formatoxArray=False):
    """    
    Parameters
    ----------
    nombreArchivo : string
        ruta del archivo a abrir.
        
    nomBloque : string
        tipo de datos a leer en el archivo original.
        'Model Outputs', 'Trajectories' o 'Devices'
    
    separador : string
        caracter separador de los datos
    
    returnFrec : bool
        si es True devuelve un int con la frecuencia de muestreo
    
    formatoxArray : bool
        si es true devuelve los datos en formato xArray    
    
    
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
    daReturn=xr.DataArray()
    
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
        frecuencia= int(linea)
        
        #Carga el nombre de las columnas
        #linea = f.readline()
        nomColsVar = str(f.readline()[:-1]).split(separador) #nombreVariables
        nomCols = str(f.readline()[:-1]).split(separador) #nombre coordenadas X,Y,Z
        
        #busca etiqueta del final del bloque
        while linea!='\n':
            if linea == '':         
                raise Exception('No se ha encontrado el final del bloque')
                
            numLinea+=1
            #print('Linea '+ str(numLinea))
            linea = f.readline()
          
    finBloque = numLinea-1 #quita 1 para descontar la línea vacía
    
    
    
    #primero asigna los nombres según el propio archivo
    nomVars=['Frame', 'Sub Frame']
    for i in range(2,len(nomCols),3):
        if "'" not in nomCols[i] and "''" not in nomCols[i]:            
            nomVars.append(nomColsVar[i].split(':')[1]+'_'+nomCols[i])#X
            nomVars.append(nomColsVar[i].split(':')[1]+'_'+nomCols[i+1])#Y
            nomVars.append(nomColsVar[i].split(':')[1]+'_'+nomCols[i+2])#Z
    
    # [i for i in nomColsVar if "'" in i]
    # nomColsVar = [i for i in nomColsVar if "'" not in i]
        
    
    
    
    #carga todos los datos
    provisional= np.genfromtxt(nombreArchivo, skip_header= inicioBloque+5, max_rows=finBloque-inicioBloque-1, delimiter=separador)
    #provisional=provisional[:, :len(nomVars)] #recorta solo hasta las variables 
    
    #Convierte los datos en pandas dataframe. Pasa solo los que no son de velocidad o aceleración
    dfReturn = pd.DataFrame(provisional[:, :len(nomVars)], columns=nomVars)
    
    # #Elimina las columnas de velocidad y aceleración, si las hay
    # borrarColsVA = dfReturn.filter(regex='|'.join(["'", "''"])).columns
    # dfReturn = dfReturn.drop(columns=borrarColsVA)
    
    #Si hace falta lo pasa a xArray
    if formatoxArray:
        #transforma los datos en xarray
        x=dfReturn.filter(regex='|'.join(['_x','_X'])).to_numpy().T
        y=dfReturn.filter(regex='|'.join(['_y','_Y'])).to_numpy().T
        z=dfReturn.filter(regex='|'.join(['_z','_Z'])).to_numpy().T
        data=np.stack([x,y,z])
        
        #Quita el identificador de la coordenada del final
        canales = [i.split(':')[-1] for i in nomColsVar[2:len(nomCols):3]] #se pone hasta len(nomCols) porque a veces queda una línea al final suelta que descuadra
                        
        n_frames = x.shape[1]
        channels = canales
        time = np.arange(start=0, stop=n_frames / frecuencia, step=1 / frecuencia)
        coords = {}
        coords["axis"] = ['x', 'y', 'z']
        coords["channel"] = channels
        coords["time"] = time
        
        daReturn=xr.DataArray(
                    data=data,
                    dims=('axis', 'channel', 'time'),
                    coords=coords,
                    name=nomBloque,
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
    ruta_Archivo = r'G:\Mi unidad\Investigacion\Proyectos\BikeFitting\Fatiga\Datos Cinematica Congreso Estudiantes\JUNTOS\01_Carrillo_FIN.csv'
    nombreArchivo = Path(ruta_Archivo)
    
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    
    #Con formato dataarray de xArray
    dfDatos, daDatos = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', formatoxArray=True)
    
    dfDatos['Right_Toe_Z'].plot()
    daDatos.sel(channel='Right_Toe', axis='z').plot.line()
