# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:05:37 2019

@author: josel
"""
from __future__ import division, print_function

"""Lee archivos de datos exportados del Vicon Nexus"""

import numpy as np
import pandas as pd
import scipy.signal

__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.1.0'
__date__ = '17/06/2019'

def read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', returnFrec=False, separador=','):
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
        nomVars.append(nomColsVar[i].split(':')[1]+'_'+nomCols[i])#X
        nomVars.append(nomColsVar[i].split(':')[1]+'_'+nomCols[i+1])#Y
        nomVars.append(nomColsVar[i].split(':')[1]+'_'+nomCols[i+2])#Z
            
    #carga los datos
    provisional= np.genfromtxt(nombreArchivo, skip_header= inicioBloque+5, max_rows=finBloque-inicioBloque-1, delimiter=separador)
    #provisional=provisional[:, :len(nomVars)] #recorta solo hasta las variables 
    
    if returnFrec:
        return pd.DataFrame(provisional, columns=nomVars), frecuencia
    else:
        return pd.DataFrame(provisional, columns=nomVars)

# =============================================================================
# %%        
# =============================================================================
if __name__ == '__main__':
    nombreArchivo = r"G:\Mi unidad\Investigacion\Proyectos\BikeFitting\Fatiga\Datos Cinematica Congreso Estudiantes\JUNTOS\01_Carrillo_FIN.csv"
    
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', returnFrec=True)
    