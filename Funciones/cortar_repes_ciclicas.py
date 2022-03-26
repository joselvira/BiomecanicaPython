# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:01:08 2021

@author: josel
"""
import numpy as np
import pandas as pd

__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.2.0.0'
__date__ = '26/03/2022'

"""
Modificaciones:
    26/03/2022, v2.0.1
        - Como variable de referencia (var_referencia) ahora se pasa un dict
        con varias dimensiones y sus coordenadas.
        - Incluida una versión en pruebas para tratar en bloques de dataarray.

    11/12/2021, v2.0.0
        - Incluida una versión con xarray, mucho más rápida.

    24/11/2021, v1.2.0
        - Incluida opción de que incluya al final de cada repetición el primer dato de la siguiente. De esta forma tienen más continuidad.

    08/11/2021, v1.1.1
        - A la función auxiliar detect_onset_aux se le puede pasar como argumento corte_ini=1 para que coja el final de la ventana encontrada. Por defecto coge el inicio.
        - Además a la misma función cuando se pide que corte con el final de la ventana, le suma 1 para que coja cuando ha superado el umbral.
        - También si el corte_ini=0 quita el primer corte y si es =1 quita el último, porque suelen quedar cortados.
    
    13/10/2021, v1.1.0
        - Incluidos argumentos para eliminar repeticiones iniciales o finales.
        - Falta poder elegir eliminar repeticiones intermedias
    
    30/09/2021, v1.0.0
        - Versión inicial
"""


# =============================================================================
# %% Función cortar directamente CON PANDAS
# =============================================================================
#from detecta import detect_peaks
def corta_repes(dfData, frec=None, col_tiempo='time', col_factores=[], col_referencia='value', col_variables=[], descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True, func_cortes=None, **args_func_cortes):
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
        
    descarta_rep_ini: int
        Número de repeticiones que descarta desde el inicio.
        
    descarta_rep_fin: int
        Número de repeticiones que descarta desde el final.
        No se tiene en cuenta si se especifica num_repes..
        
    num_repes: int
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
                    col_factores=['partID', 'tiempo'], col_referencia='value', 
                    col_variables=['value'], descarta_rep_ini=11, num_repes=4, 
                    descarta_rep_fin=2, **dict(mpd=100, show=False))
    
    >>> df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  
                    col_factores=['partID', 'tiempo'], col_referencia='value',
                    col_variables=['value'], descarta_rep_ini=10, 
                    descarta_rep_fin=2, **dict(mpd=100, show=False))
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
    #----PROBAR PASARLO A FORMATO TIDY Y CORTAR DE UNA VEZ
    for n, gb in dfData.groupby(col_factores):
      #print(n)
      
      #Busca los cortes
      cortes = func_cortes(gb[col_referencia], **args_func_cortes)
      
      #Ajusta el corte inicial y final si hace falta
      cortes = cortes[descarta_rep_ini:]
      if num_repes==None:
          cortes = cortes[:len(cortes)-descarta_rep_fin]
      else: #si se pide un nº determinado de repeticiones desde la inicial
          if len(cortes) > num_repes:
              cortes = cortes[:num_repes+1]
          else:
              print('No hay suficiente número de repeticiones en el bloque {0}, se trunca hasta el final'.format(n))
      
        
      #Divide en los cortes encontrados
      var_cortes = [] #lista vacía donde iremos incluyendo cada repetición
      for n_corte in range(len(cortes)-1):
          #print(cortes[n_corte], cortes[n_corte+1])
          var_cortes.append(gb.iloc[cortes[n_corte]:cortes[n_corte+1]+incluye_primero_siguiente, :].assign(**{'repe':n_corte, 'time_repe':np.arange(0, (cortes[n_corte+1]+incluye_primero_siguiente-cortes[n_corte])/frec, 1/frec)[:(cortes[n_corte+1]+incluye_primero_siguiente-cortes[n_corte])]})) #coge el trozo de la variable desde un corte hasta el siguiente
        
      if var_cortes!=[]:
          var_bloque.append(pd.concat(var_cortes))
      
    dfVar_cortes = pd.concat(var_bloque).reset_index(drop=True) #puede ser útil tener el índice original?
    
    #Reordena las columnas
    if [col_referencia] != col_variables:
        if col_referencia not in col_variables:
            columnas = col_factores+[col_tiempo, 'repe', 'time_repe', col_referencia]+ col_variables #reordena factores
        else:
            columnas = col_factores+[col_tiempo, 'repe', 'time_repe', col_referencia]+ [x for x in col_variables if x != col_referencia]#col_variables.pop(col_variables.index(col_referencia)) #quita la variable de referencia de la lista de variables
    else:
        columnas = col_factores+[col_tiempo, 'repe', 'time_repe', col_referencia] #reordena factores
    dfVar_cortes = dfVar_cortes[columnas]
    
    return dfVar_cortes

# =============================================================================
# %% Función intermedia para detectar a partir de umbrales con detect_onset
# =============================================================================
"""
Necesaria para pasarla como argumento a la función corta_repes. Se queda con 
el primer dato de cada par (cuando supera el umbral). Si se quiere que se quede 
con el de bajada, cambiar por cortes = cortes[:,1]
"""
def detect_onset_aux(dfData, **args_func_cortes):
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
# %% Función intermedia para detectar a partir find_peak
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
# %% con xarray, igual pero mucho más rápido. Esta versión va pasando dimensiones de una en una PROVISIONAL (Supuestamente más lenta)
# =============================================================================
def corta_simple_aux_xr(data, data_var_referencia=None, func_cortes=None, max_repes=40, descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True, **args_func_cortes):
    """
    Función intermedia para poder utilizar la función cortar_repes_ciclicas_xr        
    """
    import itertools
    cortes = func_cortes(data_var_referencia.data, **args_func_cortes)
    
    #Ajusta el corte inicial y final si hace falta
    cortes = cortes[descarta_rep_ini:]
    if num_repes==None:
        cortes = cortes[:len(cortes)-descarta_rep_fin]
    else: #si se pide un nº determinado de repeticiones desde la inicial
        if len(cortes) > num_repes:
            cortes = cortes[:num_repes+1]
        else:
            print('No hay suficiente número de repeticiones en el bloque {0}, se trunca hasta el final'.format(n))  
    
    x1 = np.full((max_repes, len(data)),np.nan)
    if len(cortes) > 0: #Si no ha encontrado cortes, devolverá el array vacío
        x2 = np.split(data, cortes)[1:-1]
        x2 = np.array(list(itertools.zip_longest(*x2, fillvalue=np.nan))).T
        
        x1[:x2.shape[0], : x2.shape[1]] = x2
        
        #Para que incluya como último dato de cada repe el primero de la siguiente
        #Se puede mejorar vectorizando
        if incluye_primero_siguiente:      
          for r in range(len(cortes)-1):
              x1[r, cortes[r+1]-cortes[r]] = data[cortes[r+1]]        
    return x1


# data_var_referencia = daTodos.isel(Archivo=0).sel(nom_var='AngBiela', lado='L', eje='x')
# data = daData.isel(Archivo=0).sel(nom_var='AngArtKnee', lado='L', eje='x').data
def corta_repes_xr(daData, frec=None, var_referencia=None, descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True, func_cortes=None, max_repes=40, **args_func_cortes):
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
        
    descarta_rep_ini: int
        Número de repeticiones que descarta desde el inicio.
        
    descarta_rep_fin: int
        Número de repeticiones que descarta desde el final.
        No se tiene en cuenta si se especifica num_repes.
        
    num_repes: int
        Número de repeticiones a considerar desde las descartadas al inicio.
    
    incluye_primero_siguiente: bool
        Indica si se incluye el primer dato de la siguiente repetición como 
        último de la anterior (se duplican). Útil para representar gráficamente.
    
    func_cortes : nombre de función
        Nombre de la función a emplear para hacer los cortes. La función debe 
        admitir un array 1D y devolver una lista también 1D con los índices de 
        los cortes.
        
    max_repes : int
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
                                var_referencia='a', descarta_rep_ini=0,
                                num_repes=None, descarta_rep_fin=0,
                                incluye_primero_siguiente=True)
    
    >>> daCortado = cortar_repes_ciclicas_xr(daTodos, func_cortes=detect_onset_aux,
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
    
        
    da = xr.apply_ufunc(corta_simple_aux_xr,  #nombre de la función
                      daData,  #después los argumentos de la función en el mismo orden
                      daData.sel(var_referencia),
                      func_cortes,
                      max_repes,
                      descarta_rep_ini,
                      num_repes,
                      descarta_rep_fin,
                      incluye_primero_siguiente,
                      input_core_dims=[['time'], ['time'], [], [], [], [], [], []],  #lista con una entrada por cada argumento
                      output_core_dims=[['repe', 'time']],  #datos que devuelve
                      exclude_dims=set(('repe', 'time' )),  #dimensiones que se permite que cambien (tiene que ser un set)
                      dataset_fill_value=np.nan,
                      vectorize=True,
                      dask='parallelized',
                      keep_attrs=True,
                      kwargs=args_func_cortes,
                      )
    da = (da.assign_coords(repe=range(len(da.repe)))
          .assign_coords(time=np.arange(0, len(da.time)/frec, 1/frec))
          .dropna(dim='repe', how='all').dropna(dim='time', how='all')
          )
    return da
    
# =============================================================================


# =============================================================================
# %% Pasa dimensiones en bloque PRUEBAS!!
# =============================================================================
def corta_simple_aux_bloque_xr(data, archivo, data_var_referencia=None, func_cortes=None, max_repes=40, descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True, **args_func_cortes):
    """
    Función intermedia para poder utilizar la función cortar_repes_ciclicas_xr        
    """
      
    import itertools
    cortes = func_cortes(data_var_referencia, **args_func_cortes)
    
    #Ajusta el corte inicial y final si hace falta
    cortes = cortes[descarta_rep_ini:]
    if num_repes==None:
        cortes = cortes[:len(cortes)-descarta_rep_fin]
    else: #si se pide un nº determinado de repeticiones desde la inicial
        if len(cortes) > num_repes:
            cortes = cortes[:num_repes+1]
        else:
            print('No hay suficiente número de repeticiones en el bloque {0}, se trunca hasta el final'.format(n))  
    
    x1 = np.full((max_repes, len(data)),np.nan)
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
    print(cortes)
    x1 = np.full((max_repes, len(data)),np.nan)
    return x1

# data_var_referencia = daData.isel(Archivo=0).sel(nom_var='AngBiela', eje='z')
# data = daData.isel(Archivo=0).data#.sel(nom_var='AngArtKnee', lado='L', eje='x')
def corta_repes_bloque_xr(daData, list_dims_bloque=['time'], var_referencia=None, descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True, func_cortes=None, max_repes=40, frec=None, **args_func_cortes):
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
        
    descarta_rep_ini: int
        Número de repeticiones que descarta desde el inicio.
        
    descarta_rep_fin: int
        Número de repeticiones que descarta desde el final.
        No se tiene en cuenta si se especifica num_repes.
        
    num_repes: int
        Número de repeticiones a considerar desde las descartadas al inicio.
    
    incluye_primero_siguiente: bool
        Indica si se incluye el primer dato de la siguiente repetición como 
        último de la anterior (se duplican). Útil para representar gráficamente.
    
    func_cortes : nombre de función
        Nombre de la función a emplear para hacer los cortes. La función debe 
        admitir un array 1D y devolver una lista también 1D con los índices de 
        los cortes.
        
    max_repes : int
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
                                var_referencia='a', descarta_rep_ini=0,
                                num_repes=None, descarta_rep_fin=0,
                                incluye_primero_siguiente=True)
    
    >>> daCortado = cortar_repes_ciclicas_xr(daTodos, func_cortes=detect_onset_aux,
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
                      max_repes,
                      descarta_rep_ini,
                      num_repes,
                      descarta_rep_fin,
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

#daL = corta_repes_xr(daDatos.sel(lado='L'), list_dims_bloque=['nom_var', 'eje', 'time'], func_cortes=detect_peaks, var_referencia=dict(nom_var='AngBiela', eje='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=True, show=graficas))
#daR = corta_repes_bloque_xr(daDatos.sel(lado='R'), list_dims_bloque=['nom_var', 'eje', 'time'], func_cortes=detect_onset_aux, var_referencia=dict(nom_var='AngBiela', eje='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(threshold=0.0, corte_ini=0, n_above=2, show=graficas))
  
# =============================================================================
    






# =============================================================================
# %% Pruebas
# =============================================================================
if __name__ == '__main__':
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
            sujeto.append(pd.DataFrame(senal + ruido, columns=['value']).assign(**{'partID':'{0:02d}'.format(suj+IDini), 'time':np.arange(0, len(senal)/Fs, 1/Fs)}))
        return pd.concat(sujeto)
    
    np.random.seed(12340) #fija la aleatoriedad para asegurarse la reproducibilidad
    n=5
    duracion=10
    frec=200.0
    Pre_v1 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [25, 29], rango_amp = [40, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre', 'nom_var':'a'})
    Post_v1 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [22, 26], rango_amp = [36, 40], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post', 'nom_var':'a'})
    Pre_v2 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [35, 39], rango_amp = [50, 55], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre', 'nom_var':'b'})
    Post_v2 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [32, 36], rango_amp = [32, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post', 'nom_var':'b'})
    Pre_v3 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [35, 39], rango_amp = [10, 15], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre', 'nom_var':'c'})
    Post_v3 = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [32, 36], rango_amp = [12, 16], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post', 'nom_var':'c'})
    
    dfTodosArchivos = pd.concat([Pre_v1, Post_v1, Pre_v2, Post_v2, Pre_v3, Post_v3]).reset_index()
    dfTodosArchivos = dfTodosArchivos[['partID', 'tiempo', 'nom_var', 'time', 'value']] #Reordena los factores

    sns.relplot(data=dfTodosArchivos, x='time', y='value',  col='tiempo', row='nom_var', units='partID', estimator=None, hue='partID',  kind='line')
    
    
    #########################
    #Prueba la función Pandas
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['partID', 'tiempo'], col_referencia='value', col_variables=['value'])#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_repe', y='value',  col='partID', row='tiempo', units='repe', estimator=None, hue='repe',  kind='line')

    #Prueba la función sin incluir el primer dato del siguiente como último del anterior
    df_cortes = corta_repes(dfTodosArchivos, incluye_primero_siguiente=False, func_cortes=detect_peaks,  col_factores=['partID', 'tiempo'], col_referencia='value', col_variables=['value'])#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_repe', y='value',  col='partID', row='tiempo', units='repe', estimator=None, hue='repe',  kind='line')

    #Descartando repes iniciales y finales
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['partID', 'tiempo'], col_referencia='value', col_variables=['value'], descarta_rep_ini=10, descarta_rep_fin=2)#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_repe', y='value',  col='partID', row='tiempo', units='repe', estimator=None, hue='repe',  kind='line')

    #Descartando repes iniciales y con nº determinado de repes
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['partID', 'tiempo'], col_referencia='value', col_variables=['value'], descarta_rep_ini=11, num_repes=4, descarta_rep_fin=2)#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_repe', y='value',  col='partID', row='tiempo', units='repe', estimator=None, hue='repe',  kind='line')


    #Prueba la función
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_onset_aux, col_factores=['partID', 'tiempo'], col_referencia='value', col_variables=['value'], **dict(threshold=80, corte_ini=1, show=True))
    df_cortes
    sns.relplot(data=df_cortes, x='time_repe', y='value',  col='partID', row='tiempo', units='repe', estimator=None, hue='repe',  kind='line')

    #########################
    #prueba la función xarray
    daTodos = dfTodosArchivos.set_index(['partID', 'tiempo', 'nom_var', 'time']).to_xarray().to_array().squeeze('variable').drop_vars('variable')
    daTodos.attrs['frec'] = 1/daTodos.time[1].values #incluimos la frecuencia como atributo
    daTodos.attrs['units'] = 'grados'
    daTodos.time.attrs['units'] = 'segundos'
    daTodos
    
    #Prueba la función
    daCortado = corta_repes_xr(daTodos, func_cortes=detect_peaks, var_referencia=dict(nom_var='a'), descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True)
    daCortado
    daCortado.sel(nom_var='b').plot.line(x='time', col='tiempo', hue='repe', row='partID')
    
    #Prueba la función
    daCortado = corta_repes_xr(daTodos, func_cortes=detect_onset_aux, var_referencia=dict(tiempo='pre', nom_var='b'), descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True, **dict(threshold=60))
    daCortado
    daCortado.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='repe', row='partID')
    
    
    import time
    tiempoProceso = time.time()
    for i in range(10):
        daCortado = corta_repes_xr(daTodos, func_cortes=detect_peaks, var_referencia=dict(nom_var='a'), descarta_rep_ini=0, num_repes=None, descarta_rep_fin=0, incluye_primero_siguiente=True)
    print('Tiempo {0:.3f} s \n'.format(time.time()-tiempoProceso))
    daCortado.sel(nom_var='a').plot.line(x='time', col='tiempo', hue='repe', row='partID')
    
    """
    daL = corta_repes_xr(daDatos.isel(Archivo=0).sel(lado='L'), func_cortes=detect_peaks, var_referencia=dict(nom_var='AngBiela', eje='x'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(valley=True, show=False))
    daR = corta_repes_xr(daDatos.sel(lado='R'), func_cortes=detect_onset_aux, var_referencia=dict(nom_var='AngBiela', eje='z'), descarta_rep_ini=1, descarta_rep_fin=0, **dict(threshold=0.0, corte_ini=0, n_above=2, show=False))
    
    daL.sel(nom_var='AngArtKnee', eje='x').plot.line(x='time', row='repe', hue='Archivo', sharey=False)
    daDatos.isel(Archivo=0).sel(nom_var='AngBiela', lado='L').plot.line(x='time', col='eje', sharey=False)
    """