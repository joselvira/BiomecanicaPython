# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:01:08 2021

@author: josel
"""
import numpy as np
import pandas as pd

__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.1.0.0'
__date__ = '30/09/2021'

"""
Modificaciones:
    30/09/2021, v1.0.0
        - Versión inicial
"""


# =============================================================================
# %% Función cortar directamente CON PANDAS
# =============================================================================
#from detecta import detect_peaks
def corta_repes(dfData, frec=None, col_tiempo='time', col_factores=[], col_referencia='value', col_variables=[], func_cortes=None, **args_func_cortes):
    """
    Function for making cuts in continuous cyclic signals
           
    Parameters
    ----------
    dfData : Pandas dataframe
        Contiene los datos continuos
        Puede estar en formato 'long' o 'tidy'
        Puede contener distinto número de factores
        
    frec : float
        frecuencia de muestreo de la señal.
        
    col_tiempo : string
        Nombre de la columna que contiene la variable temporal
        
    col_factores : list
        Listado de nombres de las columnas que contienen los factores
        
    col_referencia : string
        Nombre de la columna que contiene la variable a utilizar de referencia 
        para los cortes
        
    col_variables : streng o list
        Nombres de las columnas que contienen las variables a cortar. Puede ser
        una única columna o una lista con los combres de varias.
        
    func_cortes : nombre de función
        Nombre de la función a emplear para hacer los cortes. La función debe 
        admitir un array 1D y devolver una lista también 1D con los índices de 
        los cortes
        
    args_func_cortes : dictionary
        Diccionario con los argumentos a pasar a la función que realiza los 
        cortes
        
    
    Returns
    -------
    dfVar_cortes : Pandas dataframe
        dataframe con los datos originales con dos columnsa añadidas:
            repe : contiene el número de la repetición
            time_repe : contiene el tiempo local de cada repetición
            
        
    Examples
    --------
    >>> import numpy as np
    >>> from filtrar_Butter import filtrar_Butter
    >>> y = np.cumsum(np.random.randn(1000))
    >>> fy = filtrar_Butter(y, fr=1000, fc=10, order=2, show=True)
    >>>
    >>> dfCaminos = pd.DataFrame((np.random.random([100, 4])-0.5).cumsum(axis=0), columns=['A','B','C','D'])
    >>> dfCaminosFilt, RMS = filtrar_Butter(dfCaminos, 1000, 50, 2, show=True, returnRMS=True)
    
    """
        
        
    if not isinstance(col_variables, list):
        col_variables= [col_variables]
        
    if func_cortes==None:
        raise Exception('Debes especificar una función para buscar cortes')
        #func=detect_peaks
    
    if col_factores==[]:
        col_factores = list(dfData.columns[:-2])
    
    if frec==None:
        frec = 1/dfData.iloc[1, dfData.columns.get_loc(col_tiempo)]
        
    var_bloque=[] #lista vacía donde irá guardando cada ensayo completo
    for n, gb in dfData.groupby(col_factores):
      #print(n)
      
      #Busca los cortes
      cortes = func_cortes(gb[col_referencia], **args_func_cortes)
      
      #Divide en los cortes encontrados
      var_cortes = [] #lista vacía donde iremos incluyendo cada repetición
      for n_corte in range(len(cortes)-1):
          #print(cortes[n_corte], cortes[n_corte+1])
          var_cortes.append(gb.iloc[cortes[n_corte]:cortes[n_corte+1], :].assign(**{'repe':n_corte, 'time_repe':np.arange(0, (cortes[n_corte+1]-cortes[n_corte])/frec, 1/frec)[:(cortes[n_corte+1]-cortes[n_corte])]})) #coge el trozo de la variable desde un corte hasta el siguiente
        
      if var_cortes!=[]:
          var_bloque.append(pd.concat(var_cortes))
      
    dfVar_cortes=pd.concat(var_bloque)
    
    #Reordena las columnas
    if [col_referencia] != col_variables:
        columnas = col_factores+[col_tiempo, 'repe', 'time_repe', col_referencia]+ col_variables #reordena factores
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
def detect_onset_arriba(dfData, **args_func_cortes):
    from detecta import detect_onset  
    cortes = detect_onset(dfData, **args_func_cortes)
    cortes = cortes[:,0] #se queda con el primer dato de cada par de datos.
    return cortes


# =============================================================================
# %% Función intermedia para detectar a partir find_peak
# =============================================================================
"""
Necesaria para pasarla como argumento a la función corta_repes. Se queda con 
el primer listado de la función detect_peaks. Lo segundo que pasa son las 
características de los cortes.
"""
def find_peaks_adaptado(dfData, **args_func_cortes):
    from scipy.signal import find_peaks
    cortes = find_peaks(dfData, **args_func_cortes)
    cortes = cortes[0] #se queda con el primer dato de cada par de datos.
    return cortes



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
    
            señal = np.array(of + a*np.sin(2*np.pi*f*t + af))
            
            #Crea un ruido aleatorio controlado
            pasadas = 2.0 #nº de pasadas del filtro adelante y atrás
            orden = 2
            Cf = (2**(1/pasadas)-1)**(1/(2*orden)) #correction factor. Para 2nd order = 0.802 
            Wn = 2*fc_err/Fs/Cf
            b1, a1 = butter(orden, Wn, btype = 'low')
            ruido = filtfilt(b1, a1, np.random.uniform(a-err, a+err, len(t)))
            
            
            #################################
            sujeto.append(pd.DataFrame(señal + ruido, columns=['value']).assign(**{'partID':'{0:02d}'.format(suj+IDini), 'time':np.arange(0, len(señal)/Fs, 1/Fs)}))
        return pd.concat(sujeto)
    
    np.random.seed(12340) #fija la aleatoriedad para asegurarse la reproducibilidad
    n=20
    duracion=10
    frec=200.0
    Pre = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [25, 29], rango_amp = [40, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'pre'})
    Post = crea_muestra_continua(n, Fs=frec, IDini=0, rango_offset = [22, 26], rango_amp = [36, 40], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion]).assign(**{'tiempo':'post'})
    dfTodosArchivos = pd.concat([Pre, Post]).reset_index()
    dfTodosArchivos = dfTodosArchivos[['partID', 'tiempo', 'time', 'value']] #Reordena los factores
    
    sns.relplot(data=dfTodosArchivos, x='time', y='value',  col='tiempo', units='partID', estimator=None, hue='partID',  kind='line')
    
    
    
    #Prueba la función
    df_cortes = corta_repes(dfTodosArchivos, func_cortes=detect_peaks,  col_factores=['partID', 'tiempo'], col_referencia='value', col_variables=['value'])#, **dict(mpd=100, show=False))
    df_cortes
    sns.relplot(data=df_cortes, x='time_repe', y='value',  col='partID', row='tiempo', units='repe', estimator=None, hue='repe',  kind='line')