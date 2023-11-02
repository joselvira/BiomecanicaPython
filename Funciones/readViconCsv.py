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
try:
    import polars as pl
except:
    print('Si quieres usar la versión con Polars, debes instalarlo')
    pass
#import scipy.signal

__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.4.0.4'
__date__ = '07/05/2023'

"""
Modificaciones:
    07/05/2023, v.4.0.4
        - Mejorada lectura con polars
    
    22/04/2023, v.4.0.3
        - Corregido que n Polars actual (0.17.2) no coindicen los saltos de
          línea cuando empieza con línea en blanco.
        - Optimizado código de lectura con Polars.
          
    17/04/2023, v.4.0.2
        - Corregido error nombre variable archivo en read_vicon_csv_pl_xr.
        - Optimizada parte lectura con open en versión Polars.
        - Puede cargar por separado datos EMG o Forces, aunque vayan en el
          mismo archivo.
        - Cambiada nombre variables a n_var por conflicto en xarray, .var
          calcula la varianza.
        - En función con Polars, La selección de variables la hace en el df
          Polars, no en el dataarray.

    27/03/2023, v.4.0.0
        - Incluida una versión que lee con Polars y lo pasa a DataArray
          directamente (read_vicon_csv_pl_xr), mucho más rápida.
        - La versión que lee con Polars es capaz de leer csv con encabezados
          repetidos.
          
    07/06/2022, v.3.0.0
        - Intento de corrección cuando tiene que leer csv con variables EMG
          modeladas, que se interfieren entre las xyz.
        
    09/04/2022, v.2.3.0
        - Habilitado para cargar como dataframe y dataArray EMG Noraxon en modo Devices.
        - Incluye el tiempo en una columna.

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

def read_vicon_csv_pl_xr(file, section='Model Outputs', n_vars_load=None, coincidence='similar', sep=','):
    """
    Parameters
    ----------
    file : string or path of the file
        DESCRIPTION.
    section : string, optional
        Kind of data variables to load.
        Options: 'Trajectories', 'Model Outputs', 'Forces', 'EMG'
        The default is 'Model Outputs'.
    n_vars_load : list, optional
        DESCRIPTION. The default is None.
    coincidence: string
        When selecting which variables to load, allows strings containing the
        indicated or forces to be exact.
        Options: 'similar', 'exact'. The default is 'similar'.
    sep : string, optional
        Separator used in the csv file. The default is ','.

    Returns
    -------
    Xarray DataArray.

    """
    if section in ['Forces', 'EMG']:
        n_block = 'Devices'
    elif section == 'Model Outputs EMG':
        n_block = 'Model Outputs'
    else:
        n_block = section
    
    #Ensures that file is a csv file
    file = file.with_suffix('.csv')

    #----Check for blank lines. In current Polars (0.17.2) line breaks do not match when starting with blank line
    with open(file, mode='rt') as f:
        offset_blank_ini = 0
        
        while f.readline() in [u'\r\n', u'\n', 'ï»¿\n']:
            offset_blank_ini += 1
    
    #----Search section position and length
    with open(file, mode='rt') as f:
        num_lin = 0
        ini_bloque = None
        fin_bloque = None        
                    
        #Scrolls through the entire file to find the start and end of the section and the number of lines
        for linea in f:
            #Search for section start label
            if ini_bloque is None and n_block in linea:
                ini_bloque = num_lin
                #The frequency is below the section label
                freq = int(f.readline().replace(sep,'')) #removes the separator for cases where the file has been saved with Excel (full line with separator)
                
                # #Load columns names
                # n_head = str(f.readline()[:-1]).split(sep) #variable names
                # n_subhead = str(f.readline()[:-1]).lower().split(sep) #coordinate names (x,y,z)
                # num_lin+=3
                num_lin+=1
            #When start found, search the end
            if ini_bloque is not None and fin_bloque is None and linea=='\n':
                fin_bloque = num_lin-1
                break
        
            num_lin+=1            
        # fin_archivo = num_lin
        
    if ini_bloque is None:
        raise Exception('Section header not found')
        return
        
    
    #----Load data from file
    pl.read_csv(file, truncate_ragged_lines=True)
    df = pl.read_csv(file, has_header=True, skip_rows=ini_bloque+2-offset_blank_ini,
                     n_rows=fin_bloque-ini_bloque-2,
                     truncate_ragged_lines=True
                     #columns=range(len(n_vars_merged)),
                     #new_columns=n_vars_merged, separator=sep
                     )
    n_head = df.columns
    n_subhead = list(np.char.lower(df.slice(0,1).to_numpy()[0].astype(str)))
    
    #TODO: SIPLIFY WITH skip_rows_after_header=1?
    #Remove subhead and units rows
    df = df.slice(2, None)
    
    if n_block in ['Trajectories', 'Model Outputs', 'Model Outputs EMG']:
        n_vars_merged = ['Frame', 'Sub Frame']            
        for i in range(2,len(n_subhead)):
            if n_subhead[i] in 'x' and "'" not in n_subhead[i] and "''" not in n_subhead[i]:
                #print(n_subhead[i], n_head[i])
                n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i])#X
                n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i+1])#Y
                n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i+2])#Z
            elif 'emg' in n_subhead[i]:
                 #print(n_subhead[i], n_head[i])
                 n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i])
        
        #Rename headers
        df = df.rename(dict(zip(n_head, n_vars_merged)))
        
        if section == 'Model Outputs EMG':
            df = df.select(pl.col('^*_emg.*$'))
            df = (df.rename(dict(zip(df.columns, [n[:-3] for n in df.columns])))
                  .select(pl.exclude('^*_duplicate.*$')) #remove duplicates
                  #.slice(2, None) #remove subhead and units rows
                  )
        else: #Trajectories and Model Outputs
            df = (df.select(pl.exclude(['Frame', 'Sub Frame']))
                    .select(pl.exclude('^*_duplicate.*$')) #remove duplicates
                    .select(pl.exclude('^*_emg.*$')) #remove 1D EMG variables
                    #.slice(2, None) #remove subhead and units rows
                  )
    
    elif n_block == 'Devices':
        if section == 'EMG':
            # n_head = n_head[:len(n_subhead)] #ajusta el número de variables
            # n_vars_merged = rename_duplicates(n_head)
            # selection = [s for s in n_vars_merged[2:] if 'EMG' in s]
            # selection2 = selection
           
            df = (df.select(pl.col('^*EMG.*$'))                      
                    .select(pl.exclude('^*_duplicate.*$')) #remove duplicates                    
                    #.slice(2, None) #remove subhead and units rows
                  )
                        
            
        elif section == 'Forces':
            n_vars_merged = ['Frame', 'Sub Frame']            
            for i in range(2,len(n_subhead)):
                if 'x' in n_subhead[i] and "'" not in n_subhead[i] and "''" not in n_subhead[i]:
                    #print(n_subhead[i], n_head[i])
                    n_vars_merged.append(n_head[i] + '_' + n_subhead[i])#X
                    n_vars_merged.append(n_head[i] + '_' + n_subhead[i+1])#Y
                    n_vars_merged.append(n_head[i] + '_' + n_subhead[i+2])#Z
                elif 'v' in n_subhead[i]:
                    n_vars_merged.append(n_head[i] + '_' + n_subhead[i])
                    
            df = (df.rename(dict(zip(df.columns, n_vars_merged)))
                      .select(pl.exclude('^*EMG.*$'))                      
                      .select(pl.exclude('^*_duplicate.*$')) #remove duplicates                    
                    #.slice(2, None) #remove subhead and units rows
                  )
            
                       
    #----Filter variables
    if n_vars_load is not None:
        if not isinstance(n_vars_load, list):
            n_vars_load = [n_vars_load] #in case a string is passed
        if coincidence == 'similar':
            selection = [s for s in df.columns if any(xs in s for xs in n_vars_load)]
        elif coincidence == 'exact':
            selection = n_vars_load
        df = df.select(pl.col(selection))
    
        
    
    #----Transform polars to xarray
    if section in ['Trajectories', 'Model Outputs', 'Forces']:
        #Decompose on its axes
        x = df.select(pl.col('^*x$')).to_numpy() #los que acaban en la coordenada o si están repetidos 
        y = df.select(pl.col('^*y$')).to_numpy()
        z = df.select(pl.col('^*z$')).to_numpy()        
        data = np.stack([x,y,z])
        
        ending = -3 if section == 'Forces' else -2
        coords={'axis' : ['x', 'y', 'z'],
                'time' : np.arange(data.shape[1]) / freq,
                'n_var' : [x[:ending] for x in df.columns if 'x' in x[-1]],
                }
        da = (xr.DataArray(data=data,
                          dims=coords.keys(),
                          coords=coords,
                          )
             .astype(float)
             .transpose('n_var', 'axis', 'time')
             )
    
    elif section in ['EMG', 'Model Outputs EMG']:
        data = df.to_numpy().T
        coords={'n_var' : df.columns,#selection2,
                'time' : np.arange(df.shape[0]) / freq,                
                }
        da = xr.DataArray(data=data,
                          dims=coords.keys(),
                          coords=coords,
                         ).astype(float)
    
    
    da.name = section
    da.attrs['freq'] = freq
    da.time.attrs['units'] = 's'
    if section == 'Trajectories':
        da.attrs['units'] = 'mm'
    elif 'EMG' in section:
        da.attrs['units'] = 'V'
    elif 'Forces' in section:
        da.attrs['units'] = 'N'
    return da





def rename_duplicates(names):
    cols = pd.Series(names)
    if len(names) != len(cols.unique()):
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        names = list(cols)
    return names


def read_vicon_csv_pl_xr2(file, section='Model Outputs', n_vars_load=None, coincidence='similar', sep=','):
    """
    Parameters
    ----------
    file : string or path of the file
        DESCRIPTION.
    section : string, optional
        Kind of data variables to load.
        Options: 'Trajectories', 'Model Outputs', 'Forces', 'EMG'
        The default is 'Model Outputs'.
    n_vars_load : list, optional
        DESCRIPTION. The default is None.
    coincidence: string
        When selecting which variables to load, allows strings containing the
        indicated or forces to be exact.
        Options: 'similar', 'exact'. The default is 'similar'.
    sep : string, optional
        Separator used in the csv file. The default is ','.

    Returns
    -------
    Xarray DataArray.

    """
    if section in ['Forces', 'EMG']:
        n_block = 'Devices'
    elif section == 'Model Outputs EMG':
        n_block = 'Model Outputs'
    else:
        n_block = section
    
    
    #----Check for blank lines. In current Polars (0.17.2) line breaks do not match when starting with blank line
    with open(file, mode='rt') as f:
        offset_blank_ini = 0
        
        while f.readline() in [u'\r\n', u'\n', 'ï»¿\n']:
            offset_blank_ini += 1
    
    #----Search section position and length
    with open(file, mode='rt') as f:
        num_lin = 0
        ini_bloque = None
        fin_bloque = None        
                    
        #Scrolls through the entire file to find the start and end of the section and the number of lines
        for linea in f:
            #Search for section start label
            if ini_bloque is None and n_block in linea:
                ini_bloque = num_lin
                #After the section label is the frequency
                freq = int(f.readline().replace(sep,'')) #removes the separator for cases where the file has been saved with Excel (full line with separator)
                
                #Load columns names
                n_head = str(f.readline()[:-1]).split(sep) #variable names
                n_subhead = str(f.readline()[:-1]).lower().split(sep) #coordinate names (x,y,z)
                num_lin+=3
            
            #When start found, search the end
            if ini_bloque is not None and fin_bloque is None and linea=='\n':
                fin_bloque = num_lin-1                
            num_lin+=1            
        fin_archivo = num_lin
        
    if ini_bloque is None:
        raise Exception('Section header not found')
        return
        
    
    #----Assign header labels
    if n_block == 'Devices':
        if section == 'EMG':
            n_head = n_head[:len(n_subhead)] #ajusta el número de variables
            n_vars_merged = rename_duplicates(n_head)
            selection = [s for s in n_vars_merged[2:] if 'EMG' in s]
            selection2 = selection
                        
            
        elif section == 'Forces':
            n_vars_merged = [h+'_'+sh for h,sh in zip(n_head, n_subhead)]
            n_vars_merged = rename_duplicates(n_vars_merged)
            selection = [s for s in n_vars_merged[2:] if 'EMG' not in s]
            selection2 = []
            for i in range(2,len(n_subhead)):
                if 'x' in n_subhead[i] and 'EMG' not in n_subhead[i]:
                    #print(n_subhead[i], n_head[i])
                    selection2.append(n_head[i]+'_' + n_subhead[i][-1])#x
                    selection2.append(n_head[i]+'_' + n_subhead[i+1][-1])#y
                    selection2.append(n_head[i]+'_' + n_subhead[i+2][-1])#z
                    
            
    elif n_block in ['Trajectories', 'Model Outputs', 'Model Outputs EMG']:
        n_vars_merged = ['Frame', 'Sub Frame']            
        for i in range(2,len(n_subhead)):
            if n_subhead[i] in 'xX' and "'" not in n_subhead[i] and "''" not in n_subhead[i]:
                #print(n_subhead[i], n_head[i])
                n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i])#X
                n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i+1])#Y
                n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i+2])#Z
            elif 'emg' in n_subhead[i]:
                 #print(n_subhead[i], n_head[i])
                 n_vars_merged.append(n_head[i].split(':')[1]+'_' + n_subhead[i])
        selection = set(n_vars_merged[2:]) #remove duplicates
        if section == 'Model Outputs EMG':
            selection = sorted([s for s in selection if '_emg' in s])
            selection2 = [s[:-4] for s in selection]
        else:
            selection = sorted([s for s in selection if '_emg' not in s])
            selection2 = selection
        n_vars_merged = rename_duplicates(n_vars_merged)
    
    #----Load data from file            
    df = (pl.read_csv(file, has_header=False, skip_rows=ini_bloque+3-offset_blank_ini,
                 n_rows=fin_bloque-ini_bloque-2,
                 columns=range(len(n_vars_merged)),
                 new_columns=n_vars_merged, separator=sep)
         .select(pl.col(selection))
         .rename(dict(zip(selection, selection2)))
         .slice(2, None) #remove subhead and units rows
        )
    
    #----Filter variables
    if n_vars_load:
        if not isinstance(n_vars_load, list):
            n_vars_load = [n_vars_load] #in case a string is passed
        if coincidence == 'similar':
            selection2 = [s for s in selection2 if any(xs in s for xs in n_vars_load)]
        elif coincidence == 'exact':
            selection2 = n_vars_load
        df = df.select(pl.col(selection2))
        
    
    #----Transform polars to xarray
    if section in ['EMG', 'Model Outputs EMG']:
        data = df.to_numpy().T
        coords={'n_var' : selection2,
                'time' : np.arange(df.shape[0]) / freq,                
                }
        da = xr.DataArray(data=data,
                          dims=coords.keys(),
                          coords=coords,
                         ).astype(float)
    
    
    elif section in ['Trajectories', 'Model Outputs', 'Forces']:
        #Decompose on its axes
        x = df.select(pl.col('^*_x|_x.$')).to_numpy() #los que acaban en la coordenada o si están repetidos 
        y = df.select(pl.col('^*_y|_y.$')).to_numpy()
        z = df.select(pl.col('^*_z|_z.*$')).to_numpy()        
        data = np.stack([x,y,z])
                
        coords={'axis' : ['x', 'y', 'z'],
                'time' : np.arange(data.shape[1]) / freq,
                'n_var' : [x[:-2] for x in df.columns if '_x' in x or '_X' in x],            
                }
        da = (xr.DataArray(data=data,
                          dims=coords.keys(),
                          coords=coords,
                          )
             .astype(float)
             .transpose('n_var', 'axis', 'time')
             )
        
    da.name = section
    da.attrs['freq'] = freq
    da.time.attrs['units'] = 's'
    if section == 'Trajectories':
        da.attrs['units'] = 'mm'
    elif 'EMG' in section:
        da.attrs['units'] = 'V'
    elif 'Forces' in section:
        da.attrs['units'] = 'N'
    return da

   
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
        
        
        iniBloque = numLinea
               
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
                raise Exception('End of section not found')
                
            numLinea+=1
            #print('Linea '+ str(numLinea))
            linea = f.readline()
          
    finBloque = numLinea-1 #quita 1 para descontar la línea vacía
    
    #Cuenta el nº de líneas totales    
    with open(nombreArchivo, mode='rt') as f:
        finArchivo = len(f.readlines())
        
    #Etiquetas de columnas
    if nomBloque == 'Devices':
        nomVars = ['Frame', 'Sub Frame']+ nomColsVar[2:-1] #hay que quitar el último, que es ''
        #nomVars2=list(filter(lambda c: c!='', nomVars))        
    
    else: #para trajectories y Models
        #primero asigna los nombres según el propio archivo
        # nomVars=['Frame', 'Sub Frame']
        # for i in range(2,len(nomCols),3):
        #     if "'" not in nomCols[i] and "''" not in nomCols[i] and 'EMG' not in nomCols[i]: #elimina las posibles columnas de velocidad y aceleración
        #         print(nomCols[i])    
        #         nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i])#X
        #         nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+1])#Y
        #         nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+2])#Z
        
        nomVars=['Frame', 'Sub Frame']        
        for i in range(2,len(nomCols)):
            if nomCols[i] in 'xX' and "'" not in nomCols[i] and "''" not in nomCols[i]:
                #print(nomCols[i], nomColsVar[i])
                nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i])#X
                nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+1])#Y
                nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+2])#Z
            elif 'EMG' in nomCols[i]:
                 #print(nomCols[i], nomColsVar[i])
                 nomVars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i])
            #else:
                #print(nomCols[i])
                
                
    
    # [i for i in nomColsVar if "'" in i]
    # nomColsVar = [i for i in nomColsVar if "'" not in i]
        
   
    #carga todos los datos
    #CON GENFROMTXT FALLA SI NO EMPIEZA LA PRIMERA LÍNEA CON DATOS
    #provisional= np.genfromtxt(nombreArchivo, skip_header= iniBloque+5, max_rows=finBloque-iniBloque-1, delimiter=separador, missing_values='', filling_values=np.nan, invalid_raise=True)
    #provisional=provisional[:, :len(nomVars)] #recorta solo hasta las variables 
    
    #Convierte los datos en pandas dataframe. Pasa solo los que no son de velocidad o aceleración
    #dfReturn = pd.DataFrame(provisional[:, :len(nomVars)], columns=nomVars)
    #dfReturn = dfReturn.iloc[:, :len(nomVars)] #se queda solo con las columnas de las variables, quita las de velocidad si las hay
    
    
    
    #Con pandas directamente funciona (para evitar error si primera línea no son datos, lee la fila de las unidades y luego la quita)
    dfReturn = pd.read_csv(nombreArchivo, delimiter=separador, header=None, skiprows=iniBloque+4, skipfooter=finArchivo-finBloque-5, usecols=range(len(nomVars)), engine='python')
    dfReturn = dfReturn.drop(index=0).reset_index(drop=True).astype(float) #borra la primera fila, que contiene las unidades
    
    # x=pd.read_csv(nombreArchivo, delimiter=separador, header=iniBloque, skipfooter=finArchivo-finBloque-5, engine='python')
    # x.columns
    # sub_nom_cols = x.iloc[0,:]
    
    #Nombra encabezado
    if nomBloque == 'Devices':
        if 'Noraxon Ultium' in nomVars[3]:
            var = [s.split('- ')[-1] for s in nomVars]
            coord = nomCols[:len(nomVars)]
            dfReturn.columns = var
            dimensiones=['nom_var', 'time']
            var_name=['nom_var']
            
    else:
        var = ['_'.join(s.split('_')[:-1]) for s in nomVars[:len(nomVars)]] #gestiona si la variable tiene separador '_', lo mantiene
        coord = [s.split(':')[-1].lower() for s in nomCols[:len(nomVars)]] #pasa coordenadas a minúscula
        dfReturn.columns = pd.MultiIndex.from_tuples(list(zip(*[var,coord])), names=['nom_var', 'eje'])
        #Elimina columnas con variables modeladas EMG si las hay
        dfReturn = dfReturn.drop(columns=dfReturn.filter(regex='emg'))
        #dfReturn.duplicated().sum()
        dimensiones = ['nom_var', 'eje', 'time']
        var_name=['nom_var', 'eje']
        
    
    #dfReturn.columns=[var, coord]
    #dfReturn.columns.set_names(names=['Variable', 'Coord'], level=[0,1], inplace=True)
    
    #Incluye columna con tiempo
    dfReturn.insert(2,'time', np.arange(len(dfReturn))/frecuencia)
    
    if formatoxArray:
        #if nomBloque == 'Devices':
        #    dfReturn.columns = dfReturn.columns.droplevel(1)

        #dfReturn.iloc[:,2:].melt(id_vars='time', var_name=['nom_var', 'eje']).set_index(dimensiones).to_xarray().to_array()

#TODO: Arreglar esto que no funciona pasar a xarray
        daReturn = (dfReturn.iloc[:,2:]
                    #.assign(**{'time':np.arange(len(dfReturn))/frec})
                    .melt(id_vars='time', var_name=var_name).set_index(dimensiones)
                    .to_xarray().to_array()
                    .squeeze('variable').drop_vars('variable')  #la quita de dimensiones y coordenadas
                    )
        daReturn.name = nomBloque
        daReturn.attrs['frec'] = frecuencia
        daReturn.time.attrs['units'] = 's'
        
    
    if header_format=='flat' and nomBloque != 'Devices':
        dfReturn.columns = dfReturn.columns.map('_'.join).str.strip()
        
    # #Elimina las columnas de velocidad y aceleración, si las hay
    # borrarColsVA = dfReturn.filter(regex='|'.join(["'", "''"])).columns
    # dfReturn = dfReturn.drop(columns=borrarColsVA)
    
    #Si hace falta lo pasa a xArray
    if False:#formatoxArray:
        #prueba para hacerlo directamente desde dataframe
        #dfReturn.assign(**{'time':np.arange(len(dfReturn))/frec}).drop(columns='').melt(id_vars='time').set_index(['nom_var', 'eje', 'time']).to_xarray().to_array()
        
        if header_format!='flat':
            dfReturn.columns = dfReturn.columns.map('_'.join).str.strip()
        
        #transforma los datos en xarray
        x=dfReturn.filter(regex='|'.join(['_x','_X'])).to_numpy().T
        y=dfReturn.filter(regex='|'.join(['_y','_Y'])).to_numpy().T
        z=dfReturn.filter(regex='|'.join(['_z','_Z'])).to_numpy().T
        data=np.stack([x,y,z])
        
        #Quita el identificador de la coordenada del final
        nom_vars = dfReturn.filter(regex='|'.join(['_x','_X'])).columns.str.rstrip('|'.join(['_x','_X']))
        
        time = np.arange(x.shape[1]) / frecuencia
        coords = {}
        coords['eje'] = ['x', 'y', 'z']
        coords['nom_var'] = nom_vars
        coords['time'] = time
        
        daReturn=xr.DataArray(
                    data=data,
                    dims=('eje', 'nom_var', 'time'),
                    coords=coords,
                    name=nomBloque,
                    attrs={'frec':frecuencia}
                    #**kwargs,
                )
        if header_format!='flat': #si hace falta lo vuelve a poner como multiindex
            dfReturn.columns=pd.MultiIndex.from_tuples(list(zip(*[var,coord])), names=['nom_var', 'eje'])
    
            
    
    if formatoxArray and returnFrec:
        return dfReturn, daReturn, frecuencia
    elif formatoxArray:
        return dfReturn, daReturn
    elif returnFrec:
        return dfReturn, frecuencia
    else:   
        return dfReturn

# =============================================================================
# %% MAIN    
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
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs')   
    
    #Sin fila inicial en blanco
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN_SinFilaBlancoInicial.csv'
    nombreArchivo = Path(ruta_Archivo)
    
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs')   
    
    
    #Solo bloque modelos
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN_2.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs')
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Trajectories')
    
    
    #Con hueco muy grande al inicio
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconConHuecoInicio_S27_WHT_T2_L01.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    dfDatos['R5Meta_z'].plot()
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Trajectories')
    daDatos.sel(n_var='R5Meta', axis='z').plot.line(x='time')
    
    
    #Con formato dataarray de xArray    
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN.csv'
    nombreArchivo = Path(ruta_Archivo)    
    dfDatos, daDatos1 = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', formatoxArray=True)
    dfDatos['Right_Toe_z'].plot()
    daDatos1.sel(nom_var='Right_Toe', eje='z').plot.line()
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Trajectories')
    daDatos.sel(n_var='Right_Toe', axis='z').plot.line()
    
    
    dfDatos, daDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', formatoxArray=True)
    dfDatos['AngArtLKnee_x'].plot()
    daDatos.sel(nom_var='AngArtLKnee', eje='x').plot.line()

    dfDatos, daDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', formatoxArray=True, header_format='multi')
    dfDatos['AngArtLKnee'].plot()
    daDatos.sel(nom_var='AngArtLKnee').plot.line(x='time')
    
    
    
    #Archivo con huecos
    ruta_Archivo = r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconConHuecos_S01_WHF_T1_L04.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', returnFrec=True)
    dfDatos.plot()
    
    dfDatos, daDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', formatoxArray=True, returnFrec=True)
    dfDatos.plot()
    daDatos.sel(eje='x').plot.line(x='time', hue='nom_var')
    
    dfDatos, daDatos, frecuencia = read_vicon_csv(nombreArchivo, nomBloque='Trajectories', formatoxArray=True, returnFrec=True, header_format='multi')
    dfDatos.plot(x='time')
    daDatos.sel(eje='x').plot.line(x='time', hue='nom_var')
    
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Trajectories')
    daDatos.sel(axis='x').plot.line(x='time')
    
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

    dfDatosFlat[['Right_Toe_x','Right_Toe_y','Right_Toe_z']].plot()
    dfDatosMulti['Right_Toe'].plot()
    
    dfDatosMulti.loc[:, (slice(None), 'z')].plot() #todas las variables de una misma coordenada

    
    #Lectura device EMG
    ruta_Archivo =r'F:\Investigacion\Proyectos\BikeFitting\Bikefitting\EstudioEMG_MVC\Registros\01_SofiaSanchez\SofiaSanchez\Normal-00.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatosFlat = read_vicon_csv(nombreArchivo, nomBloque='Devices')
    dfDatosFlat = read_vicon_csv(nombreArchivo, nomBloque='Devices', header_format='multi')
    dfDatosMulti, daDatos, frec = read_vicon_csv(nombreArchivo, nomBloque='Devices', header_format='multi', returnFrec=True, formatoxArray=True)
    dfDatosFlat['EMG1'].plot()
    daDatos.sel(nom_var='EMG1').plot(x='time')
    
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='EMG')
    daDatos.sel(n_var=daDatos.n_var.str.endswith('EMG1')).plot(x='time')
    
    #Lectura Modelos con variables modeladas EMG por medio
    ruta_Archivo =r'F:\Investigacion\Proyectos\BikeFitting\Bikefitting\EstudioEMG_MVC\Registros\01_SofiaSanchez\SofiaSanchez\Normal-00.csv'
    nombreArchivo = Path(ruta_Archivo)
    dfDatosMulti, frec = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', header_format='multi', returnFrec=True)
    dfDatosMulti['AngBiela'].plot()    
    dfDatos, frec = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', returnFrec=True)
    #En este formato no funciona con xarray
    
    #Modelados no EMG
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs')
    daDatos.sel(n_var='AngBiela').plot.line(x='time')
    
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs', n_vars_load=['AngArtAnkle_L', 'AngArtAnkle_R'])
    daDatos.plot.line(x='time', col='n_var')
    
    #Modelados EMG
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs EMG')
    daDatos.plot.line(x='time')
    daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs EMG', n_vars_load=['BIC'])
    
    #Pruebas con Polars
    
    ruta_Archivo = Path(r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN.csv')
    file = ruta_Archivo
    daDatos = read_vicon_csv_pl_xr(file, section='Model Outputs')
    
    daDatos = read_vicon_csv_pl_xr(file, section='Model Outputs', n_vars_load=['AngArtCuello', 'AngArtL1', 'Right_Pedal', 'vAngBiela'])
    #daDatos.sel(n_var=['Right_Pedal', 'vAngBiela']).plot.line(x='time', col='n_var', col_wrap=4, hue='axis', aspect=1.2)
    
    
    
    #Pruebas cuando empieza con primera fila en blanco
    file = Path(r"F:\Programacion\Python\Mios\TratamientoDatos\pruebaLecturaFilaCero-Nexus-Polars.csv")
    file = Path(r"F:\Investigacion\Proyectos\Saltos\2023PreactivacionSJ2\DataCollection\S00\S00\New Session\S00_SJ_006.csv")
    daDatos = read_vicon_csv_pl_xr(file, section='Trajectories')
    daDatos = read_vicon_csv_pl_xr(file, section='EMG')
    daDatos = read_vicon_csv_pl_xr(file, section='Forces')
    
    
    #Carga con archivo con EMG y Fuerzas
    ruta_Archivo = Path(r'F:\Programacion\Python\Mios\SeminarioDoctorado-ProgramacionAnalisisDatos\2021\ArchivosEjemplos\Vicon\ViconTraj-Kistler-EMG.csv')
    file = ruta_Archivo
    daDatos = read_vicon_csv_pl_xr(file, section='Trajectories')
    daDatos = read_vicon_csv_pl_xr(file, section='EMG')
    daDatos = read_vicon_csv_pl_xr(file, section='Forces')
    
    
    
    #Compara rapidez con versión Pandas
    import time
    
    ruta_Archivo = Path(r'F:\Programacion\Python\Mios\TratamientoDatos\EjemploViconSinHuecos_01_Carrillo_FIN.csv')
    nombreArchivo = ruta_Archivo
   
    tme=time.time()
    for x in range(10):
        daDatos = read_vicon_csv_pl_xr(nombreArchivo, section='Model Outputs')
    print(time.time()-tme)
    
    tme=time.time()
    for x in range(10):
        dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs')
    print(time.time()-tme)
 
    tme=time.time()
    for x in range(10):
        dfDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', header_format='multi')
    print(time.time()-tme)
 
    tme=time.time()
    for x in range(10):
        dfDatos, daDatos = read_vicon_csv(nombreArchivo, nomBloque='Model Outputs', formatoxArray=True, header_format='multi')
    print(time.time()-tme)
 