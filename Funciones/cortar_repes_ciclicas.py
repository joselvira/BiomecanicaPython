# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:01:08 2021

@author: josel
"""
#%%
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
import xarray as xr 
import itertools

import matplotlib.pyplot as plt


__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.2.1.0'
__date__ = '28/01/2023'


"""
Modificaciones:
    28/01/2023, v2.1.0
        - Función común corta_repes que distribuye según los datos sean Pandas
        o xarray.
        - Función en xarray que devuelve sólo los nº de índice de los cuts.
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
# %%Funciones intermedias para detectar a partir de umbrales con detect_onset
# =============================================================================

"""
Necesaria para pasarla como argumento a la función corta_repes. Se queda con 
el primer dato de cada par (cuando supera el umbral). Si se quiere que se quede 
con el de bajada, cambiar por cuts = cuts[:,1]
"""
def detect_onset_detecta_aux(dfData, **args_func_cortes):
    #Si se pasa como argumento corte_ini=1, coge el corte del final de cada ventana
    try:
        from detecta import detect_onset
    except:		
        from detect_onset import detect_onset
    
    try: #si no se ha indicado el núm corte, coge el primero
        corte_ini=args_func_cortes['corte_ini']
        args_func_cortes.pop('corte_ini', None)
    except:
        corte_ini=0
        
    cortes = detect_onset(dfData, **args_func_cortes)
    
    if corte_ini==1:
        cortes = cortes[:, corte_ini] + 1 #si se elije el final de la ventana, se añade 1 para que empiece cuando ya ha superado el umbral
        cortes = cortes[:-1] #quita el último porque suele quedar cortado
    else:
        cortes = cortes[:, corte_ini] #se queda con el primer o segundo dato de cada par de datos.
        cortes = cortes[1:] #quita el primero porque suele quedar cortado
    return cortes


# =============================================================================
# Función intermedia para detectar a partir find_peaks
# =============================================================================
"""
Necesaria para pasarla como argumento a la función corta_repes. Se queda con 
el primer listado de la función detect_peaks. Lo segundo que pasa son las 
características de los cortes.
"""
def find_peaks_aux(data, **args_func_cortes):
    from scipy.signal import find_peaks
    data=data.copy() #esto para solucionar problema cuando se llama desde xr.apply_ufunc
    cortes, _ = find_peaks(data, **args_func_cortes)
    return cortes #se queda con el primer dato de cada par de datos.


# =============================================================================
# Función de Pyomeca
# =============================================================================
def detect_onset_pyomeca(
    x,
    threshold: Union[float, int],
    n_above: int = 1,
    n_below: int = 0,
    threshold2: int = None,
    n_above2: int = 1,
) -> np.array:
    if x.ndim != 1:
        raise ValueError(
            f"detect_onset works only for one-dimensional vector. You have {x.ndim} dimensions."
        )
    if isinstance(threshold, xr.DataArray):
        threshold = threshold.item()
    if isinstance(threshold2, xr.DataArray):
        threshold2 = threshold2.item()

    x = np.atleast_1d(x.copy())
    x[np.isnan(x)] = -np.inf
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack(
            (
                inds[np.diff(np.hstack((-np.inf, inds))) > n_below + 1],
                inds[np.diff(np.hstack((inds, np.inf))) > n_below + 1],
            )
        ).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1] - inds[:, 0] >= n_above - 1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if (
                    np.count_nonzero(x[inds[i, 0] : inds[i, 1] + 1] >= threshold2)
                    < n_above2
                ):
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])
    return inds



# =============================================================================
# Función de detecta
# =============================================================================

def detect_onset_detecta(x, threshold=0, n_above=1, n_below=0,
                 threshold2=None, n_above2=1, corte_ini=0, umbrales=None):

            
    x = np.atleast_1d(x).astype('float64')
    # deal with NaN's (by definition, NaN's are not greater than threshold)
    x[np.isnan(x)] = -np.inf
    # indices of data greater than or equal to threshold
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
                          inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if np.count_nonzero(x[inds[i, 0]: inds[i, 1]+1] >= threshold2) < n_above2:
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])  # standardize inds shape for output
    
    if corte_ini==1:
        inds = inds[:, corte_ini] + 1 #si se elije el final de la ventana, se añade 1 para que empiece cuando ya ha superado el umbral
        inds = inds[:-1] #quita el último porque suele quedar cortado
    else:
        inds = inds[:, corte_ini] #se queda con el primer o segundo dato de cada par de datos.
        inds = inds[1:] #quita el primero porque suele quedar cortado
    
    return inds


def detect_onset_detecta_mean_sd_aux(data, corte_ini=0, xSD=None, **args_func_cortes):
    #Si se pasa como argumento corte_ini=1, coge el corte del final de cada ventana
    try:
        from detecta import detect_onset
    except:		
        from detect_onset import detect_onset
    
    # try: #si no se ha indicado el núm corte, coge el primero
    #     corte_ini=args_func_cortes['corte_ini']
    #     args_func_cortes.pop('corte_ini', None)
    # except:
    #     corte_ini=0
    
    if xSD is not None:
        if 'threshold' in args_func_cortes:
            print('hay umbral', args_func_cortes['threshold'])
            args_func_cortes.pop('threshold', None)
        args_func_cortes['threshold'] = np.mean(data) + np.std(data)*xSD
        print(xSD, args_func_cortes['threshold'])
    
    cortes = detect_onset(data, **args_func_cortes)
    
    if corte_ini==1:
        cortes = cortes[:, corte_ini] + 1 #si se elije el final de la ventana, se añade 1 para que empiece cuando ya ha superado el umbral
        cortes = cortes[:-1] #quita el último porque suele quedar cortado
    else:
        cortes = cortes[:, corte_ini] #se queda con el primer o segundo dato de cada par de datos.
        cortes = cortes[1:] #quita el primero porque suele quedar cortado
    return cortes




# =============================================================================
# =============================================================================
# %%VERSIONES CON FUNCIONES SUELTAS

# =============================================================================
# Selecciona según sea Pandas o xarray y según lo que se le pida
# =============================================================================
def corta_repes(data, idx=False, precortado=None, frec=None, var_referencia=None, max_cortes=50, col_tiempo='time', col_factores=[], col_referencia='value', col_variables=[], descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, func_cortes=None, **args_func_cortes):
    if idx==True: #devuelve solo los índices de los cortes
        if isinstance(data, xr.DataArray):
            data_cortado = corta_repes_xr_idx(data, frec=frec, var_referencia=var_referencia, descarta_corte_ini=descarta_corte_ini, num_cortes=num_cortes, descarta_corte_fin=descarta_corte_fin, func_cortes=func_cortes, max_cortes=max_cortes, **args_func_cortes)
        
        elif isinstance(data, pd.DataFrame):
            print('Por completar')
    
    else: #devuelve señal cortada 
        if not (isinstance(precortado, xr.DataArray) or isinstance(precortado, pd.DataFrame)): #busca cortes y corta señal
            if isinstance(data, xr.DataArray):
                data_cortado = corta_repes_xr(data, frec=frec, var_referencia=var_referencia, descarta_corte_ini=descarta_corte_ini, num_cortes=num_cortes, descarta_corte_fin=descarta_corte_fin, incluye_primero_siguiente=incluye_primero_siguiente, func_cortes=func_cortes, max_cortes=max_cortes, **args_func_cortes)
            
            elif isinstance(data, pd.DataFrame):
                data_cortado = corta_repes_pd(data, frec=frec, col_tiempo=col_tiempo, col_factores=col_factores, col_referencia=col_referencia, col_variables=[], descarta_corte_ini=descarta_corte_ini, num_cortes=num_cortes, descarta_corte_fin=descarta_corte_fin, incluye_primero_siguiente=incluye_primero_siguiente, func_cortes=func_cortes, **args_func_cortes)
        
        else: #corta señal a partir de los cortes pasados
            if isinstance(data, xr.DataArray):
                data_cortado = corta_repes_xr_precortes(data, daPrecortes=precortado, frec=frec, descarta_corte_ini=descarta_corte_ini, num_cortes=num_cortes, descarta_corte_fin=descarta_corte_fin, incluye_primero_siguiente=incluye_primero_siguiente)
            
            elif isinstance(data, pd.DataFrame):
                print('Por completar')
        
    return data_cortado

# =============================================================================
# %% Función cortar directamente CON PANDAS
# =============================================================================
#from detecta import detect_peaks
def corta_repes_pd(dfData, frec=None, col_tiempo='time', col_factores=[], col_referencia='value', col_variables=[], descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, func_cortes=None, **args_func_cortes):
    """
    Función para hacer cortes en señales cíclicas
           
    Parameters
    ----------
    dfData : Pandas dataframe
        Contiene los datos continuos
        Puede estar en formato 'long' o 'tidy'
        Puede contener distinto número de factores.
        
    frec : float
        frecuencia de muestreo de la señal. Si no se indica la calcula a partir
        del primer dato de la columna tiempo.
        
    col_tiempo : string
        Nombre de la columna que contiene la variable temporal.
        
    col_factores : list
        Listado de nombres de las columnas que contienen los factores.
        
    col_referencia : string
        Nombre de la columna que contiene la variable a utilizar de referencia 
        para los cortes.
        
    col_variables : string o list
        Nombres de las columnas que contienen las variables a cortar. Puede ser
        una única columna o una lista con los combres de varias.
        
    descarta_corte_ini: int
        Número de repeticiones que descarta desde el inicio.
        
    descarta_corte_fin: int
        Número de repeticiones que descarta desde el final.
        No se tiene en cuenta si se especifica num_cortes.
        
    num_cortes: int
        Número de repeticiones a considerar desde las descartadas al inicio.
    
    incluye_primero_siguiente: bool
        Indica si se incluye el primer dato de la siguiente repetición como 
        último de la anterior (se duplican). Útil para representar gráficamente.
    
    func_cortes : nombre de función
        Nombre de la función a emplear para hacer los cortes. La función debe 
        admitir un array 1D y devolver una lista también 1D con los índices de 
        los cortes.
        
    args_func_cortes : dictionary
        Diccionario con los argumentos a pasar a la función que realiza los 
        cortes.
        
    
    Returns
    -------
    dfVar_cortes : Pandas dataframe
        dataframe con los datos originales con dos columnas añadidas:
            repe : contiene el número de la repetición
            time_repe : contiene el tiempo local de cada repetición
            
        
    Examples
    --------
    >>> df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  
                    col_factores=['ID', 'tiempo'], col_referencia='value', 
                    col_variables=['value'], descarta_corte_ini=11, num_cortes=4, 
                    descarta_corte_fin=2, **dict(mpd=100, show=False))
    
    >>> df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  
                    col_factores=['ID', 'tiempo'], col_referencia='value',
                    col_variables=['value'], descarta_corte_ini=10, 
                    descarta_corte_fin=2, **dict(mpd=100, show=False))
    """
        
        
    if not isinstance(col_variables, list):
        col_variables= [col_variables]
    
    if not isinstance(col_factores, list):
        col_factores = [col_factores]
        
    if func_cortes==None:
        raise Exception('Debes especificar una función para buscar cortes')
        #func=detect_peaks
    
    if col_factores==[]:
        col_factores = list(dfData.columns[:-2])
    
    if frec==None:
        frec = 1/dfData.iloc[1, dfData.columns.get_loc(col_tiempo)]
        
    var_bloque=[] #lista vacía donde irá guardando cada ensayo completo
    #PROBAR PASARLO A FORMATO TIDY Y CORTAR DE UNA VEZ
    for n, gb in dfData.groupby(col_factores):
      #print(n)
      
      #Busca los cortes
      cortes = func_cortes(gb[col_referencia], **args_func_cortes)
      
      #Ajusta el corte inicial y final si hace falta
      cortes = cortes[descarta_corte_ini:]
      if num_cortes==None:
          cortes = cortes[:len(cortes)-descarta_corte_fin]
      else: #si se pide un nº determinado de repeticiones desde la inicial
          if len(cortes) > num_cortes:
              cortes = cortes[:num_cortes+1]
          else:
              print('No hay suficiente número de repeticiones en el bloque {0}, se trunca hasta el final'.format(n))
      
        
      #Divide en los cortes encontrados
      var_cortes = [] #lista vacía donde iremos incluyendo cada repetición
      for n_corte in range(len(cortes)-1):
          #print(cortes[n_corte], cortes[n_corte+1])
          var_cortes.append(gb.iloc[cortes[n_corte]:cortes[n_corte+1]+incluye_primero_siguiente, :].assign(**{'n_event':n_corte, 'time_corte':np.arange(0, (cortes[n_corte+1]+incluye_primero_siguiente-cortes[n_corte])/frec, 1/frec)[:(cortes[n_corte+1]+incluye_primero_siguiente-cortes[n_corte])]})) #coge el trozo de la variable desde un corte hasta el siguiente
        
      if var_cortes!=[]:
          var_bloque.append(pd.concat(var_cortes))
      
    dfVar_cortes = pd.concat(var_bloque).reset_index(drop=True) #puede ser útil tener el índice original?
    
    #Reordena las columnas
    if [col_referencia] != col_variables:
        if col_referencia not in col_variables:
            columnas = col_factores+[col_tiempo, 'n_event', 'time_corte', col_referencia]+ col_variables #reordena factores
        else:
            columnas = col_factores+[col_tiempo, 'n_event', 'time_corte', col_referencia]+ [x for x in col_variables if x != col_referencia]#col_variables.pop(col_variables.index(col_referencia)) #quita la variable de referencia de la lista de variables
    else:
        columnas = col_factores+[col_tiempo, 'n_event', 'time_corte', col_referencia] #reordena factores
    dfVar_cortes = dfVar_cortes[columnas]
    
    return dfVar_cortes





# =============================================================================
# %% con xarray, igual pero mucho más rápido. Esta versión va pasando dimensiones de una en una PROVISIONAL (Supuestamente más lenta)
# =============================================================================

# data_var_referencia = daData.isel(Archivo=0).sel(var_referencia)
# data = daData.isel(Archivo=0).sel(nom_var='AngArtHip', lado='L', eje='x').data
# data = daData.isel(Archivo=0).sel(nom_var='AngArtHip', eje='x').data
# data = daData.isel(Archivo=0).sel(nom_var='REC').data
# **dict(threshold=0.0, n_above=2, corte_ini=1, show=True)
def corta_repes_xr(daData, frec=None, var_referencia=None, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, func_cortes=None, max_cortes=50, **args_func_cortes):
    """
    Función para hacer cortes en señales cíclicas
           
    Parameters
    ----------
    daData : xarray DataArray
        Puede contener distinto número de factores en las dimensiones
    
    frec : float
        Frecuencia de muestreo de la señal. Si no se indica, intenta tomarla
        de los atributos de propio DataArray, y si no la tiene la calcula a 
        partir del primer dato de la dimensión tiempo.
    
    var_referencia : dict
        Diccionario con las dimensiones y nombres de la columna que contiene 
        la variable a utilizar de referencia para los cortes.
        
    descarta_corte_ini: int
        Número de repeticiones que descarta desde el inicio.
        
    descarta_corte_fin: int
        Número de repeticiones que descarta desde el final.
        No se tiene en cuenta si se especifica num_cortes.
        
    num_cortes: int
        Número de repeticiones a considerar desde las descartadas al inicio.
    
    incluye_primero_siguiente: bool
        Indica si se incluye el primer dato de la siguiente repetición como 
        último de la anterior (se duplican). Útil para representar gráficamente.
    
    func_cortes : nombre de función
        Nombre de la función a emplear para hacer los cortes. La función debe 
        admitir un array 1D y devolver una lista también 1D con los índices de 
        los cortes.
        
    max_cortes : int
        Máximo número de repeticiones esperables. Actualmente no es capaz de 
        ajustarse automáticamente.
        
    args_func_cortes : dictionary
        Diccionario con los argumentos a pasar a la función que realiza los 
        cortes.
        
    
    Returns
    -------
    daCortes : xarray DataArray
        DataArray con los datos originales con dos dimensiones añadidas:
            repe : contiene el número de la repetición
            time : contiene el tiempo local de cada repetición
            
        
    Examples
    --------
    >>> daCortado = cortar_repes_ciclicas_xr(daTodos, func_cortes=detect_peaks,
                                var_referencia='a', descarta_corte_ini=0,
                                num_cortes=None, descarta_corte_fin=0,
                                incluye_primero_siguiente=True)
    
    >>> daCortado = cortar_repes_ciclicas_xr(daTodos, func_cortes=detect_onset_detecta_aux,
                                var_referencia='a', incluye_primero_siguiente=True,
                                **dict(threshold=60))
    """
    
    def corta_simple_aux_xr(data, data_var_referencia=None, func_cortes=None, max_cortes=50, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, **args_func_cortes):
        """
        Función intermedia para poder utilizar la función cortar_repes_ciclicas_xr        
        """
        if np.count_nonzero(~np.isnan(data))==0 or np.count_nonzero(~np.isnan(data_var_referencia))==0:
            return np.full((max_cortes, len(data)),np.nan)
        
        cortes = func_cortes(data_var_referencia.data, **args_func_cortes)
        
        #Ajusta el corte inicial y final si hace falta
        cortes = cortes[descarta_corte_ini:]
        
        if num_cortes==None:
            cortes = cortes[:len(cortes)-descarta_corte_fin]
        else: #si se pide un nº determinado de repeticiones desde la inicial
            if len(cortes) >= num_cortes:
                cortes = cortes[:num_cortes+1]
            else:
                print('No hay suficiente número de repeticiones en el bloque, se trunca hasta el final')  
        
        x1 = np.full((max_cortes, len(data)),np.nan)
        if len(cortes) > 0: #Si no ha encontrado cortes, devolverá el array vacío
            x2 = np.split(data, cortes)[1:-1]
            x2 = np.array(list(itertools.zip_longest(*x2, fillvalue=np.nan))).T
            #print(x1.shape, x2.shape)
            x1[:x2.shape[0], : x2.shape[1]] = x2
            
            #Para que incluya como último dato de cada repe el primero de la siguiente
            #Se puede mejorar vectorizando
            if incluye_primero_siguiente:      
              for r in range(len(cortes)-1):
                  x1[r, cortes[r+1]-cortes[r]] = data[cortes[r+1]]        
        return x1

    
    if func_cortes==None:
        raise Exception('Debes especificar una función para buscar cortes')
        
    if frec==None:
        try:
            frec = daData.attrs['frec'] #si tiene la frecuencia en los atributos, la coge
        except:
            frec = 1/(daData.time[1]-daData.time[0]).values.round(3) #si no está en atributos, la calcula a partir de la dimensión time
    
    """
    data = daData[0,0,0].data.copy()
    data_var_referencia = daData.sel(var_referencia)[0,0].data
    """
    da = xr.apply_ufunc(corta_simple_aux_xr,  #nombre de la función
                      daData,  #después los argumentos de la función en el mismo orden
                      daData.sel(var_referencia),
                      func_cortes,
                      max_cortes,
                      descarta_corte_ini,
                      num_cortes,
                      descarta_corte_fin,
                      incluye_primero_siguiente,
                      daData.dims,
                      input_core_dims=[['time'], ['time'], [], [], [], [], [], [], []],  #lista con una entrada por cada argumento
                      output_core_dims=[['n_event', 'time']],  #datos que devuelve
                      exclude_dims=set(('n_event', 'time' )),  #dimensiones que se permite que cambien (tiene que ser un set)
                      dataset_fill_value=np.nan,
                      vectorize=True,
                      dask='parallelized',
                      keep_attrs=True,
                      kwargs=args_func_cortes,
                      )
    da = (da.assign_coords(n_event=range(len(da.n_event)))
          .assign_coords(time=np.arange(0, len(da.time)/frec, 1/frec))
          .dropna(dim='n_event', how='all').dropna(dim='time', how='all')
          )
    
    return da




def corta_repes_xr_idx(daData, frec=None, var_referencia=None, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, func_cortes=None, max_cortes=50, dims=None, **args_func_cortes):
    def corta_simple_aux_xr_idx(data, data_var_referencia=None, func_cortes=None, max_cortes=50, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, **args_func_cortes):
        """
        Función intermedia
        """
        cortes = np.full(max_cortes, np.nan)
        if np.count_nonzero(~np.isnan(data))==0:
            return cortes
        cort = func_cortes(data_var_referencia, **args_func_cortes)
                
        #Ajusta el corte inicial y final si hace falta
        cort = cort[descarta_corte_ini:]
        
        if num_cortes==None:
            cort = cort[:len(cort)-descarta_corte_fin]
        else: #si se pide un nº determinado de repeticiones desde la inicial
            if len(cort) >= num_cortes:
                cort = cort[:num_cortes+1]
            else:
                print('No hay suficiente número de repeticiones en el bloque, se trunca hasta el final')  
        
        cortes[:len(cort)] = cort
        return cortes

    
    if func_cortes==None:
        raise Exception('Debes especificar una función para buscar cortes')
        
    if frec==None:
        try:
            frec = daData.attrs['frec'] #si tiene la frecuencia en los atributos, la coge
        except:
            frec = 1/(daData.time[1]-daData.time[0]).values.round(3) #si no está en atributos, la calcula a partir de la dimensión time
    
    """
    data = daData[0,0,0].data.copy()
    data_var_referencia = daData.sel(var_referencia)[0,0].data
    """    
    da = xr.apply_ufunc(corta_simple_aux_xr_idx,  #nombre de la función
                      daData,  #después los argumentos de la función en el mismo orden
                      daData.sel(var_referencia),
                      func_cortes,
                      max_cortes,
                      descarta_corte_ini,
                      num_cortes,
                      descarta_corte_fin,
                      input_core_dims=[['time'], ['time'], [], [], [], [], []],  #lista con una entrada por cada argumento
                      output_core_dims=[['n_event']],  #datos que devuelve
                      exclude_dims=set(('n_event', 'time' )),  #dimensiones que se permite que cambien (tiene que ser un set)
                      dataset_fill_value=np.nan,
                      vectorize=True,
                      dask='parallelized',
                      keep_attrs=True,
                      kwargs=args_func_cortes,
                      )
    da = (da.assign_coords(n_event=range(len(da.n_event)))
          .dropna(dim='n_event', how='all').dropna(dim='n_event', how='all')
          )
    
    return da



def corta_repes_xr_precortes(daData, daPrecortes=None, frec=None, var_referencia=None, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True):
    """
    Función para hacer cortes en señales cíclicas usando cortes prehechos
    """
    
    def corta_simple_aux_xr(data, cortes, max_cortes=50, incluye_primero_siguiente=True):
        if np.count_nonzero(~np.isnan(data))==0:
            return np.full((max_cortes, len(data)),np.nan)
        
        cortes = cortes[~np.isnan(cortes)].astype(int)
        
        x1 = np.full((max_cortes, len(data)),np.nan)
        #if len(cortes) > 0: #Si no ha encontrado cortes, devolverá el array vacío
        x2 = np.split(data, cortes)[1:-1]
        x2 = np.array(list(itertools.zip_longest(*x2, fillvalue=np.nan))).T
        #print(x1.shape, x2.shape)
        x1[:x2.shape[0], : x2.shape[1]] = x2
        
        #Para que incluya como último dato de cada repe el primero de la siguiente
        #Se puede mejorar vectorizando
        if incluye_primero_siguiente:
            for r in range(len(cortes)-1):
                x1[r, cortes[r+1]-cortes[r]] = data[cortes[r+1]]
            
        return x1
    
    if not isinstance(daPrecortes, xr.DataArray):
        raise Exception('Debes especificar los precortes')
        
    if frec==None:
        try:
            frec = daData.attrs['frec'] #si tiene la frecuencia en los atributos, la coge
        except:
            frec = 1/(daData.time[1]-daData.time[0]).values.round(3) #si no está en atributos, la calcula a partir de la dimensión time
    
    max_cortes = len(daPrecortes.n_event)
    
    """
    data=daData[0,0,0].data
    cortes=daPrecortes[0,0,0].data
    """    
    da = xr.apply_ufunc(corta_simple_aux_xr,  #nombre de la función
                      daData,  #después los argumentos de la función en el mismo orden
                      daPrecortes,
                      max_cortes,                      
                      incluye_primero_siguiente,
                      input_core_dims=[['time'], ['n_event'], [], []],  #lista con una entrada por cada argumento
                      output_core_dims=[['n_event', 'time']],  #datos que devuelve
                      exclude_dims=set(('n_event', 'time' )),  #dimensiones que se permite que cambien (tiene que ser un set)
                      dataset_fill_value=np.nan,
                      vectorize=True,
                      dask='parallelized',
                      keep_attrs=True,
                      #kwargs=args_func_cortes,
                      )
    da = (da.assign_coords(n_event=range(len(da.n_event)))
          .assign_coords(time=np.arange(0, len(da.time)/frec, 1/frec))
          .dropna(dim='n_event', how='all').dropna(dim='time', how='all')
          )
    
    return da




# =============================================================================


# =============================================================================
# %% Pasa dimensiones en bloque PRUEBAS!!
# =============================================================================
def corta_simple_aux_bloque_xr(data, archivo, data_var_referencia=None, func_cortes=None, max_cortes=40, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, **args_func_cortes):
    """
    Función intermedia para poder utilizar la función cortar_repes_ciclicas_xr        
    """
      
    cortes = func_cortes(data_var_referencia, **args_func_cortes)
    
    #Ajusta el corte inicial y final si hace falta
    cortes = cortes[descarta_corte_ini:]
    if num_cortes==None:
        cortes = cortes[:len(cortes)-descarta_corte_fin]
    else: #si se pide un nº determinado de repeticiones desde la inicial
        if len(cortes) > num_cortes:
            cortes = cortes[:num_cortes+1]
        else:
            print('No hay suficiente número de repeticiones en el bloque {0}, se trunca hasta el final'.format(n))  
    
    x1 = np.full((max_cortes, len(data)),np.nan)
    if len(cortes) > 0: #Si no ha encontrado cortes, devolverá el array vacío
        x2 = np.split(data, cortes, axis=-1)[1:-1]
        xx=np.array_split(data, cortes, axis=-1)[1:-1] #seguir probando con este, pero parece que hacen lo mismo
        plt.plot(xx[-1][8,2,:].T)
        """
        x2 = np.array(list(itertools.zip_longest(*x2, fillvalue=np.nan))).T
        
        x1[:x2.shape[0], : x2.shape[1]] = x2
        
        #Para que incluya como último dato de cada repe el primero de la siguiente
        #Se puede mejorar vectorizando
        if incluye_primero_siguiente:      
          for r in range(len(cortes)-1):
              x1[r, cortes[r+1]-cortes[r]] = data[cortes[r+1]]
              """
    print(archivo, x2[0].shape, x2[-1].shape)#data.shape)
    x1 = np.full((max_cortes, len(data)),np.nan)
    return x1

# data_var_referencia = daData.isel(Archivo=0).sel(nom_var='AngBiela', eje='z')
# data = daData.isel(Archivo=0).data#.sel(nom_var='AngArtKnee', lado='L', eje='x')
def corta_repes_bloque_xr(daData, list_dims_bloque=['time'], var_referencia=None, descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, func_cortes=None, max_cortes=50, frec=None, **args_func_cortes):
    """
    Función para hacer cortes en señales cíclicas
           
    Parameters
    ----------
    daData : xarray DataArray
        Puede contener distinto número de factores en las dimensiones
        
    frec : float
        Frecuencia de muestreo de la señal. Si no se indica, intenta tomarla
        de los atributos de propio DataArray, y si no la tiene la calcula a 
        partir del primer dato de la dimensión tiempo.
    
    var_referencia : dict
        Diccionario con las dimensiones y nombres de la columna que contiene 
        la variable a utilizar de referencia para los cortes.
        
    descarta_corte_ini: int
        Número de repeticiones que descarta desde el inicio.
        
    descarta_corte_fin: int
        Número de repeticiones que descarta desde el final.
        No se tiene en cuenta si se especifica num_cortes.
        
    num_cortes: int
        Número de repeticiones a considerar desde las descartadas al inicio.
    
    incluye_primero_siguiente: bool
        Indica si se incluye el primer dato de la siguiente repetición como 
        último de la anterior (se duplican). Útil para representar gráficamente.
    
    func_cortes : nombre de función
        Nombre de la función a emplear para hacer los cortes. La función debe 
        admitir un array 1D y devolver una lista también 1D con los índices de 
        los cortes.
        
    max_cortes : int
        Máximo número de repeticiones esperables. Actualmente no es capaz de 
        ajustarse automáticamente.
        
    args_func_cortes : dictionary
        Diccionario con los argumentos a pasar a la función que realiza los 
        cortes.
        
    
    Returns
    -------
    daCortes : xarray DataArray
        DataArray con los datos originales con dos dimensiones añadidas:
            repe : contiene el número de la repetición
            time : contiene el tiempo local de cada repetición
            
        
    Examples
    --------
    >>> daCortado = cortar_repes_ciclicas_xr(daTodos, func_cortes=detect_peaks,
                                var_referencia='a', descarta_corte_ini=0,
                                num_cortes=None, descarta_corte_fin=0,
                                incluye_primero_siguiente=True)
    
    >>> daCortado = cortar_repes_ciclicas_xr(daTodos, func_cortes=detect_onset_detecta_aux,
                                var_referencia='a', incluye_primero_siguiente=True,
                                **dict(threshold=60))
    """
    import xarray as xr 
    
    
    if func_cortes==None:
        raise Exception('Debes especificar una función para buscar cortes')
        
    if frec==None:
        try:
            frec = daData.attrs['frec'] #si tiene la frecuencia en los atributos, la coge
        except:
            frec = 1/daData.time[1].values #si no está en atributos, la calcula a partir de la dimensión time
            
    da = xr.apply_ufunc(corta_simple_aux_bloque_xr,  #nombre de la función
                      daData,  #después los argumentos de la función en el mismo orden
                      daData.Archivo,
                      daData.sel(var_referencia),
                      func_cortes,
                      max_cortes,
                      descarta_corte_ini,
                      num_cortes,
                      descarta_corte_fin,
                      incluye_primero_siguiente,
                      input_core_dims=[list_dims_bloque, [], ['time'], [], [], [], [], [], []],  #lista con una entrada por cada argumento
                      output_core_dims=[['repe', 'time']],  #datos que devuelve
                      exclude_dims=set(('repe', 'time' )),  #dimensiones que se permite que cambien (tiene que ser un set)
                      dataset_fill_value=np.nan,
                      vectorize=True,
                      #dask='parallelized',
                      keep_attrs=True,
                      kwargs=args_func_cortes,
                      )
    da = da.assign_coords(repe=range(len(da.repe)))
    da = da.assign_coords(time=np.arange(0, len(da.time)/frec, 1/frec))
    da = da.dropna(dim='repe', how='all').dropna(dim='time', how='all')    
    return da


#daL = corta_repes_xr(daDatos.sel(lado='L'), list_dims_bloque=['nom_var', 'eje', 'time'], func_events=detect_peaks, var_referencia=dict(nom_var='AngBiela', eje='z'), discard_phases_end=1, discard_phases_end=0, **dict(valley=True, show=graficas))
#daR = corta_repes_bloque_xr(daDatos.sel(lado='R'), list_dims_bloque=['nom_var', 'eje', 'time'], func_events=detect_onset_detecta_aux, var_referencia=dict(nom_var='AngBiela', eje='z'), discard_phases_end=1, discard_phases_end=0, **dict(threshold=0.0, corte_ini=0, n_above=2, show=graficas))
  
# =============================================================================
    






if __name__ == '__main__':
    # =============================================================================
    # %% Crea muestra
    # =============================================================================

    """
    import sys
    sys.path.append('F:\Programacion\Python\Mios\Functions')
    #sys.path.append('G:\Mi unidad\Programacion\Python\Mios\Functions')
    
    from cortar_repes_ciclicas import corta_repes
    """
    
    #Simula una muestra de datos con factores
    import numpy as np
    import pandas as pd
    import xarray as xr
    from scipy.signal import butter, filtfilt
    from pathlib import Path
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from detecta import detect_peaks
    
    
    #np.random.seed(123) #fija la aleatoriedad para asegurarse la reproducibilidad
    
    def crea_muestra_continua(num_suj=10, Fs=100.0, IDini=0, rango_offset = [-2.0, -0.5], rango_amp = [1.0, 2.2], rango_frec = [1.8, 2.4], rango_af=[0.0, 1.0], rango_duracion=[5.0, 5.1], amplific_ruido=[0.4, 0.7], fc_ruido=[7.0, 12.0]):
        sujeto=[]
        for suj in range(num_suj):            
            a = np.random.uniform(rango_amp[0], rango_amp[1])
            of = np.random.uniform(rango_offset[0], rango_offset[1])
            f =  np.random.uniform(rango_frec[0], rango_frec[1])
            af =  np.deg2rad(np.random.uniform(rango_af[0], rango_af[1])) #lo pasa a radianes
            err = a * np.random.uniform(amplific_ruido[0], amplific_ruido[1])
            fc_err = np.random.uniform(fc_ruido[0], fc_ruido[1])
            duracion = np.random.uniform(rango_duracion[0], rango_duracion[1])
            
            Ts = 1./Fs #intervalo de tiempo entre datos en segundos
            t = np.arange(0, duracion, Ts)
    
            senal = np.array(of + a*np.sin(2*np.pi*f*t + af))
            
            #Crea un ruido aleatorio controlado
            pasadas = 2.0 #nº de pasadas del filtro adelante y atrás
            orden = 2
            Cf = (2**(1/pasadas)-1)**(1/(2*orden)) #correction factor. Para 2nd order = 0.802 
            Wn = 2*fc_err/Fs/Cf
            b1, a1 = butter(orden, Wn, btype = 'low')
            ruido = filtfilt(b1, a1, np.random.uniform(a-err, a+err, len(t)))
            
            
            #################################
            sujeto.append(pd.DataFrame(senal + ruido, columns=['value']).assign(**{'ID':'{0:02d}'.format(suj+IDini), 'time':np.arange(0, len(senal)/Fs, 1/Fs)}))
        return pd.concat(sujeto)
    
    np.random.seed(12340) #fija la aleatoriedad para asegurarse la reproducibilidad
    n=10
    duracion=10
    frec=1000.0
    Pre_v1 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [25, 29], rango_amp = [40, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre', 'nom_var':'a'})
    Post_v1 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [22, 26], rango_amp = [36, 40], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post', 'nom_var':'a'})
    Pre_v2 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [35, 39], rango_amp = [50, 55], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre', 'nom_var':'b'})
    Post_v2 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [32, 36], rango_amp = [32, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post', 'nom_var':'b'})
    Pre_v3 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [35, 39], rango_amp = [10, 15], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre', 'nom_var':'c'})
    Post_v3 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [32, 36], rango_amp = [12, 16], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post', 'nom_var':'c'})
    
    dfTodosArchivos = pd.concat([Pre_v1, Post_v1, Pre_v2, Post_v2, Pre_v3, Post_v3]).reset_index()
    dfTodosArchivos = dfTodosArchivos[['ID', 'tiempo', 'nom_var', 'time', 'value']] #Reordena los factores
    
    #Lo pasa a DataArray
    daTodos = dfTodosArchivos.set_index(['ID', 'tiempo', 'nom_var', 'time']).to_xarray().to_array().squeeze('variable').drop_vars('variable')
    daTodos.attrs['frec'] = 1/(daTodos.time[1].values - daTodos.time[0].values)#incluimos la frecuencia como atributo
    daTodos.attrs['units'] = 'grados'
    daTodos.time.attrs['units'] = 's'
        
    #sns.relplot(data=dfTodosArchivos, x='time', y='value',  col='tiempo', row='nom_var', units='ID', estimator=None, hue='ID',  kind='line')
    
    # =============================================================================
    # %% PRUEBAS    
    # =============================================================================
    

    
    # =============================================================================
    # ----Pruebas con la función xarray
    # =============================================================================
    """
    #Ejemplo de importar
    sys.path.insert(1, r'F:\Programacion\Python\Mios\Functions')  # add to pythonpath
    from cortar_repes_ciclicas import corta_repes
    """
    #Prueba la función
    daCortado = corta_repes(daTodos, func_cortes=detect_peaks, var_referencia=dict(nom_var='a'), descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True)
    daCortado
    daCortado.sel(nom_var='b').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    #Prueba la función
    daCortado = corta_repes(daTodos, func_cortes=detect_onset_detecta_aux, var_referencia=dict(tiempo='pre', nom_var='b'), descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True, **dict(threshold=60))
    daCortado
    daCortado.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    
    
    import time
    tiempoProceso = time.time()
    for i in range(10):
        daCortado = corta_repes(daTodos, func_cortes=detect_peaks, var_referencia=dict(nom_var='a'), descarta_corte_ini=0, num_cortes=None, descarta_corte_fin=0, incluye_primero_siguiente=True)
    print('Tiempo {0:.3f} s \n'.format(time.time()-tiempoProceso))
    daCortado.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    """
    daL = corta_repes_xr(daDatos.isel(Archivo=0).sel(lado='L'), func_cortes=detect_peaks, var_referencia=dict(nom_var='AngBiela', eje='x'), descarta_corte_ini=1, descarta_corte_fin=0, **dict(valley=True, show=False))
    daR = corta_repes_xr(daDatos.sel(lado='R'), func_cortes=detect_onset_detecta_aux, var_referencia=dict(nom_var='AngBiela', eje='z'), descarta_corte_ini=1, descarta_corte_fin=0, **dict(threshold=0.0, corte_ini=0, n_above=2, show=False))
    
    daL.sel(nom_var='AngArtKnee', eje='x').plot.line(x='time', row='repe', hue='Archivo', sharey=False)
    daDatos.isel(Archivo=0).sel(nom_var='AngBiela', lado='L').plot.line(x='time', col='eje', sharey=False)
    """
    
    #Prueba la función con índices
    daCortado_idx = corta_repes(daTodos, idx=True, func_cortes=detect_peaks, var_referencia=dict(nom_var='a'), descarta_corte_ini=3, num_cortes=3, descarta_corte_fin=4)
    
    daCortadoIdx = corta_repes(daTodos, precortado=daCortado_idx, incluye_primero_siguiente=True)
    daCortadoIdx.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    daCortadoDirecto = corta_repes(daTodos, func_cortes=detect_peaks, var_referencia=dict(nom_var='a'), descarta_corte_ini=3, num_cortes=3, descarta_corte_fin=4, incluye_primero_siguiente=True)
    daCortadoDirecto.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    
    #Con función Pyomeca
    daCortado_idx = corta_repes(daTodos, idx=True, func_cortes=detect_onset_pyomeca, var_referencia=dict(nom_var='a'), descarta_corte_ini=0, num_cortes=5, descarta_corte_fin=0, **dict(threshold=60))
    
    
    #Con función Detecta
    daCortado_idx = corta_repes(daTodos, idx=True, func_cortes=detect_onset_detecta, var_referencia=dict(nom_var='a'), descarta_corte_ini=0, num_cortes=5, descarta_corte_fin=0, **dict(threshold=60, corte_ini=0, umbrales=np.std))
    daCortadoIdx = corta_repes(daTodos, precortado=daCortado_idx, incluye_primero_siguiente=True)
    daCortadoIdx.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    
    daCortado_idx = (corta_repes(daTodos, idx=True, func_cortes=detect_onset_detecta_mean_sd_aux, var_referencia=dict(nom_var='a'), descarta_corte_ini=0, num_cortes=5, descarta_corte_fin=0,
                                **dict(corte_ini=0, xSD=1.0))
                     )
    daCortadoIdx = corta_repes(daTodos, precortado=daCortado_idx, incluye_primero_siguiente=True)
    daCortadoIdx.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    
    
    
    
    #------------------
    xx=daTodos.copy()
    umbral=xr.full_like(xx, 100.0)
    var_referencia=dict(nom_var='a')
    xx=xx.where(~xx.isnull(), -np.inf)
    inds = xr.where(xx.sel(var_referencia) >= umbral, xx, np.nan)
    inds.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    xx_cort = corta_repes(inds, idx=True, func_cortes=detect_onset_detecta, **dict(threshold=60, corte_ini=0))
    xx_cort.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    
    inds = xr.where(xx.sel(var_referencia) >= umbral, xx.time*daTodos.frec, np.nan)
    inds = np.abs(inds.diff('time'))
    inds.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='n_event', row='ID')
    
    
    
    # =============================================================================
    # ----Pruebas con la función Pandas
    # =============================================================================
    """
    #Ejemplo de importar
    sys.path.insert(1, r'F:\Programacion\Python\Mios\Functions')  # add to pythonpath
    from cortar_repes_ciclicas import corta_repes
    """
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['ID', 'tiempo'], col_referencia='value', col_variables=['value'])#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_corte', y='value',  col='ID', row='tiempo', units='n_event', estimator=None, hue='n_event',  kind='line')

    #Prueba la función sin incluir el primer dato del siguiente como último del anterior
    df_cortes = corta_repes_pd(dfTodosArchivos, incluye_primero_siguiente=False, func_cortes=detect_peaks,  col_factores=['ID', 'tiempo'], col_referencia='value', col_variables=['value'])#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_corte', y='value',  col='ID', row='tiempo', units='n_event', estimator=None, hue='n_event',  kind='line')

    #Descartando repes iniciales y finales
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['ID', 'tiempo'], col_referencia='value', col_variables=['value'], descarta_corte_ini=10, descarta_corte_fin=2)#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_corte', y='value',  col='ID', row='tiempo', units='n_event', estimator=None, hue='n_event',  kind='line')

    #Descartando repes iniciales y con nº determinado de repes
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['ID', 'tiempo'], col_referencia='value', col_variables=['value'], descarta_corte_ini=11, num_cortes=4, descarta_corte_fin=2)#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_corte', y='value',  col='ID', row='tiempo', units='n_event', estimator=None, hue='n_event',  kind='line')


    #Prueba la función
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_onset_detecta_aux, col_factores=['ID', 'tiempo'], col_referencia='value', col_variables=['value'], **dict(threshold=80, corte_ini=1, show=True))
    df_cortes
    sns.relplot(data=df_cortes, x='time_corte', y='value',  col='ID', row='tiempo', units='n_event', estimator=None, hue='n_event',  kind='line')

    
    