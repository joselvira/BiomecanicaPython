# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Aplica filtro paso Butterworth. Se puede pasar array 1D o en pandas DataFrame 2D.
Función de paso bajo o alto y función de bandpass.
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal


__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.1.4'
__date__ = '26/02/2021'

# =============================================================================
# %% Función filtrar low o High pass
# =============================================================================
def filtrar_Butter(DatOrig, Fr, Fc, Orden=2.0, Tipo='low', ReturnRMS=False, show=False, ax=None):
    """    
    Parameters
    ----------
    DatOrig : array 1D o dataframe de pandas en 2D o xarray.
            Datos originales a filtrar.
    Fr : frecuencia de registro.
    Fc : frecuencia de corte del filtro.
    Orden : 'agudeza' del filtro.
            2 por defecto.
    Tipo : 'low' o 'high'
            low por defecto.
    ReturnRMS: True o False
                (False por defecto). Devuelve o no el RMS de la diferencia 
                entre filtrado y original.
    show : muestra o no el gráfico con datos originales y filtrados con el RMSE.
    ax : ejes de una figura creada previamente.
    
    Returns
    -------
    filtData : array de datos filtrados.
    RMS: root mean square de la diferencia entre los datos originales y los filtrados.
    
    Notes
    -----
    Describir filtro de 2º orden y 2 pasadas como "double 2nd order Butterworth filter"
    (van den Bogert) http://biomch-l.isbweb.org/threads/26625-Matlab-Code-for-EMG-processing-(negative-deflection-after-normalization!!!)?p=32073#post32073
    
    
    Examples
    --------
    >>> import numpy as np
    >>> from filtrar_Butter import filtrar_Butter
    >>> y = np.cumsum(np.random.randn(1000))
    >>> fy = filtrar_Butter(y, Fr=1000, Fc=10, Orden=2, show=True)
    >>>
    >>> dfCaminos = pd.DataFrame((np.random.random([100, 4])-0.5).cumsum(axis=0), columns=['A','B','C','D'])
    >>> dfCaminosFilt, RMS = filtrar_Butter(dfCaminos, 1000, 50, 2, show=True, ReturnRMS=True)
    
    """
    
                
    #orden = 2 #orden 2 para que al hacer el doble paso sea de 4th orden
    pasadas = 2.0 #nº de pasadas del filtro adelante y atrás
    
    #Fc = 15
    Cf = (2**(1/pasadas)-1)**(1/(2*Orden)) #correction factor. Para 2nd order = 0.802 
    Wn = 2*Fc/Fr/Cf    
           
    b, a = scipy.signal.butter(Orden, Wn, btype = Tipo)
    
    if isinstance(DatOrig, pd.DataFrame): #Si los datos son pandas dataframe
        DatFilt=pd.DataFrame()
        RMS=pd.DataFrame()
        for i in range(DatOrig.shape[1]):
            DatFilt[DatOrig.columns[i]] = scipy.signal.filtfilt(b, a, DatOrig.iloc[:, i])
            RMS.at[0, DatOrig.columns[i]] = np.linalg.norm(DatFilt.iloc[:,i].values-DatOrig.iloc[:,i].values) / np.sqrt(len(DatOrig.iloc[:,i]))
        DatFilt.index=DatOrig.index #esto es necesario por si se pasa un slice del dataframe
        
    elif isinstance(DatOrig, pd.Series):
        DatFilt = pd.Series(scipy.signal.filtfilt(b, a, DatOrig), index=DatOrig.index, name=DatOrig.name)
        RMS = np.linalg.norm(DatFilt-DatOrig) / np.sqrt(len(DatOrig))
    
    elif isinstance(DatOrig, xr.DataArray):
        DatFilt = xr.apply_ufunc(scipy.signal.filtfilt, b, a, DatOrig.dropna(dim='time')) #se asume que hay una dimensión tiempo
        RMS=pd.DataFrame()
        for i in range(DatOrig.shape[0]):
            RMS.at[0, i]=np.linalg.norm(DatFilt[i,:]-DatOrig[i,:]) / np.sqrt(len(DatOrig[i,:]))
            #xr.apply_ufunc(np.linalg.norm, DatFilt[0,:], DatOrig[0,:])
        #pip install xskillscore
        #import xskillscore as xs        
        #RMS = xs.rmse(DatFilt, DatOrig, dim='time')
        #Investigar para hacer el RMSE directamente sin necesitar la librería xskillscore
        #CON XARRAY NO FUNCIONAN LOS GRÁFICOS
        
    else: #si los datos no son pandas dataframe
        DatFilt = scipy.signal.filtfilt(b, a, DatOrig)
        RMS = np.linalg.norm(DatFilt-DatOrig) / np.sqrt(len(DatOrig))
    
    if show:
        _plot(DatOrig, DatFilt, RMS, Fc, ax)

    if ReturnRMS:
        return DatFilt, RMS
    else:
        return DatFilt
# =============================================================================


# =============================================================================
# Presenta la gráfica
# =============================================================================
def _plot(DatOrig, DatFilt, RMS, Fc, ax):
    import matplotlib.pyplot as plt
    
    bNecesarioCerrarFigura = False
    
    if ax is None:
        bNecesarioCerrarFigura = True
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    if isinstance(DatOrig, pd.DataFrame): #Si los datos son pandas dataframe
        import seaborn as sns
        cmap = sns.color_palette('bright', n_colors=DatOrig.shape[1])
        DatFilt.plot(color=cmap, legend=False, ax=ax)
        DatOrig.plot(color=cmap, alpha=0.6, linestyle=':', legend=False, ax=ax)
        labels=[DatOrig.columns[x]+', RMSE='+'{:.3f}'.format(RMS.iloc[0,x]) for x in range(DatOrig.shape[1])]            
        plt.legend(labels)

    else: #cuando no son dataframe, incluso si son pandas series
        ax.plot(DatOrig, 'b:', label='Original')            
        ax.plot(DatFilt, 'b-', label='Filt (RMSE={:.3f})'.format(RMS))
        plt.legend(loc='best')
        
    ax.set_xlabel('Num. datos')
    ax.set_ylabel('Variable')
    ax.set_title('Filtrado Butterworth {0:3g} Hz'.format(Fc))

    if bNecesarioCerrarFigura:
        plt.show()
    
# =============================================================================
        
# =============================================================================
# %% Función filtrar low o High pass
# =============================================================================
def filtrar_Butter_bandpass(DatOrig, Fr, Fclow, Fchigh, Orden=2.0, show=False, ax=None):
    """    
    Parameters
    ----------
    DatOrig : array 1D o dataframe de pandas en 2D.
            Datos originales a filtrar.
    Fr : frecuencia de registro.
    Fclow, Fchigh : frecuencias de corte del filtro.
    Orden : 'agudeza' del filtro.
            2 por defecto.
        
    show : muestra o no el gráfico con datos originales y filtrados con el RMSE.
    ax : ejes de una figura creada previamente.
    
    Returns
    -------
    filtData : array de datos filtrados.
        
    Notes
    -----
    Información sobre el filtro Butterworth de bandpass en 
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    
    Examples
    --------
    >>> import numpy as np
    >>> from filtrar_Butter import filtrar_Butter
    >>> y = np.cumsum(np.random.randn(1000))
    >>> fy = filtrar_Butter(y, Fr=1000, Fc=10, Orden=2, show=True)
    >>>
    >>> dfCaminos = pd.DataFrame((np.random.random([100, 4])-0.5).cumsum(axis=0), columns=['A','B','C','D'])
    >>> dfCaminosFilt, RMS = filtrar_Butter(dfCaminos, 1000, 50, 2, show=True, ReturnRMS=True)
    
    """
    
    nyq = 0.5 * Fr
    low = Fclow / nyq
    high = Fchigh / nyq
    b, a = scipy.signal.butter(Orden, [low, high], btype='band')
    
    if isinstance(DatOrig, pd.DataFrame): #Si los datos son pandas dataframe
        DatFilt=pd.DataFrame()
        RMS=pd.DataFrame()
        for i in range(DatOrig.shape[1]):
            DatFilt[DatOrig.columns[i]] = scipy.signal.lfilter(b, a, DatOrig.iloc[:, i])
        DatFilt.index=DatOrig.index #esto es necesario por si se pasa un slice del dataframe
        
    elif isinstance(DatOrig, pd.Series):
        DatFilt = pd.Series(scipy.signal.lfilter(b, a, DatOrig), index=DatOrig.index, name=DatOrig.name)
        
    else: #si los datos no son pandas dataframe
        DatFilt = scipy.signal.lfilter(b, a, DatOrig)
        
    if show:
        _plot(DatOrig, DatFilt, RMS, Fclow, ax)

    return DatFilt
# =============================================================================

        
        
        
        
        
        
# =============================================================================
# %% PRUEBAS
# =============================================================================
if __name__ == '__main__':
    
    np.random.seed(2)
    y = np.cumsum(np.random.randn(1000))
    fy, rms = filtrar_Butter(y, 1000, 10, 2, show=True, ReturnRMS=True)
    fy2, rms2 = filtrar_Butter(y[100:300], 1000, 10, 2, show=True, ReturnRMS=True)
    
     
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle('Título grande', y=1.03)
    
    fy = filtrar_Butter(y, 1000, 50, 2, show=True, ax=ax)
    ax.set_title('Superpone el Título pequeno', y=1.0)
    plt.show()

    #Con dataframe de varias columnas
    num=1000
    colNames=['A','B','C','D']
    dfCaminos = pd.DataFrame((np.random.random([num, 4])-0.5).cumsum(axis=0), columns=colNames)
    
    dfCaminosFilt = filtrar_Butter(dfCaminos, 1000, 5, 2, show=True) 
    dfCaminosFilt, RMS = filtrar_Butter(dfCaminos, 1000, 50, 2, show=True, ReturnRMS=True)
    
    #con pd series
    dfCaminosFilt, RMS = filtrar_Butter(dfCaminos.iloc[:,0], 1000, 5, 2, show=True, ReturnRMS=True) 
    dfCaminosFilt, RMS = filtrar_Butter(dfCaminos['A'], 1000, 50, 2, show=True, ReturnRMS=True)
    
    
    #%%Onda con ruido
    t = np.arange(0, 2, 1/1000)
    #offset vertical
    of=[0,0, 0,0]    
    #ampitudes
    a=[3,0.5, 5,0.3]    
    #frecuencias
    f=[1,60, 3,40]    
    #phase angle, ángulo al inicio del tiempo
    pa=[0,0, 0,0]    
    ondas = pd.DataFrame(np.array([of[i] + a[i]*np.sin(2*np.pi*f[i]*t + pa[i]) for i in range(len(a))]).T)
        
    Onda=pd.DataFrame({'Onda1':ondas[0]+ondas[1], 'Onda2':ondas[2]+ondas[3]})
    
    dfOndaFilt = filtrar_Butter(Onda, 1000, 10, 2, show=True)
    
    # con cambio de index
    dfOndaFiltCacho = filtrar_Butter(Onda[100:300], 1000, 20, 2, show=True)
    
    dfOndaFiltCacho, RMS = filtrar_Butter(Onda.iloc[100:300, 0], 1000, 20, 2, show=True, ReturnRMS=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fy = filtrar_Butter(Onda.iloc[400:600, 0], 1000, 50, 2, show=True, ax=ax)
    ax.set_title('(Superpone el Título pequeño)', y=1.0)
    plt.suptitle('Título grande', y=1.03)    
    plt.show()
    
    #%%prueba bandpass
    # Filter a noisy signal.
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0
    
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02 #amplitud de la señal
    f0 = 600.0 #frecuencia principal a extraer de la señal
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    
    xFiltBand= filtrar_Butter_bandpass(x, fs, lowcut, highcut, Orden=6, show=False, ax=None)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, x,'b--')
    ax.plot(t, xFiltBand, 'r')
    plt.hlines([-a, a], 0, T, 'r', linestyles='--')
    plt.title('Filtro bandpass')    
    plt.show()
    
    ###############################
    #%%prueba con xarray
    t = np.arange(0, 2, 1/1000)
    #offset vertical
    of=[0,0, 0,0]    
    #ampitudes
    a=[3,0.5, 5,0.3]    
    #frecuencias
    f=[1,60, 3,40]    
    #phase angle, ángulo al inicio del tiempo
    pa=[0,0, 0,0]    
    ondas = pd.DataFrame(np.array([of[i] + a[i]*np.sin(2*np.pi*f[i]*t + pa[i]) for i in range(len(a))]).T)
        
    Onda=pd.DataFrame({'Onda1':ondas[0]+ondas[1], 'Onda2':ondas[2]+ondas[3]})
    
    
    da = xr.DataArray(data=np.array(Onda).T,
        
        dims=['channel', 'time'],
        coords={'channel': Onda.columns,
                'time': np.arange(0, len(Onda)/1000, 1/1000),                
                },
    )
    o = da.isel(channel=-1)
    da.plot.line(x='time') #sin filtrar
    da.isel(channel=1).plot()
    plt.show()
    
    np.linalg.norm(da.isel(channel=1)-da.isel(channel=0)) / np.sqrt(len(da.isel(channel=0)))
    
    o_filt, RMSEda = filtrar_Butter(da, 1000, 10, 2, ReturnRMS=True, show=False)
    da.plot.line(x='time')#sin filtrar
    o_filt.plot.line(x='time') #filtrado
    plt.show()
    
    #Al compararlo con el pandas sale igual
    dfOndaFilt, RMSEdf = filtrar_Butter(Onda, 1000, 10, 2, ReturnRMS=True, show=True)
    
    #%% Con xarray con varias dimensiones
    
        
    df_multiindex = pd.DataFrame({'nombre': list('xxxxxxxxyyyyyyyyzzzz'),
                                  #'test' : ['AP', 'ML']
                                  'repe' : [1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1],
                                    'time': [0,1,2,3]*5,
                                    'F' : [10,9,12,11,15,12,19,17,  10,15,8,11,5,4,18,14,  2,5,4,3]})
    df_multiindex = df_multiindex.set_index(['nombre', 'repe', 'time'])

        
    da=df_multiindex.to_xarray().to_array()
    
    #cuando tienen duraciones distintas y terminana en nan, hay que rellenar con interpolado y después dejar nan donde los había en el original
    da_filt = filtrar_Butter(da.interpolate_na(dim='time', method='linear', fill_value='extrapolate'), Fr=100, Fc=2, Orden=2, Tipo='low')
    da_filt = da_filt.where(xr.where(np.isnan(da), False, True), np.nan)

    #otras pruebas
    da.drop('time').groupby('nombre').map(filtrar_Butter, Fr=1000, Fc=10, Orden=2, show=False) #no va
    da.pipe(filtrar_Butter, Fr=1000, Fc=10, Orden=2, show=False)
