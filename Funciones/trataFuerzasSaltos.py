# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:33:55 2022

@author: josel
"""
"""
Clase con funciones para tratar fuerzas de saltos desde archivos de plataforma
de fuerzas.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages #para guardar gráficas en pdf
import time #para cuantificar tiempos de procesado

from pathlib import Path

from detecta import detect_onset
import scipy.integrate as integrate

__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.1.0.2'
__date__ = '09/01/2023'

"""
Modificaciones:
    '09/01/2023', v.1.0.2
            - En la función calcula_variables devuelve la fuerza normalizada
            como otra variable del dataset y mantiene el dtype original.
    
    '22/12/2022', v.1.0.1
            - Corrección, devuelve float al buscar inicio movimiento con DJ.
    
    '10/09/2022', v.1.0.0
            - Versión inicial con varias funciones.
"""


def carga_preprocesados(ruta_trabajo, nomArchivoPreprocesado, tipo_test):
    if Path((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix('.nc')).is_file():
        tpo = time.time()
        daDatos = xr.load_dataarray((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix('.nc')).sel(tipo=tipo_test)
        print('\nCargado archivo preprocesado ', nomArchivoPreprocesado + '_Vicon.nc en {0:.3f} s.'.format(time.time()-tpo))
    else: 
        raise Exception('No se encuentra el archivo preprocesado')
    return daDatos


def corrige_mal_ajuste_cero_medicion(daDatos):
    def ajuste_cero(data):
        dat=data.copy()
        try:            
            ind = detect_onset(-dat, threshold=max(-data)*0.9, n_above=100, show=False)
            recorte_ventana = int((ind[0,1]-ind[0,0])*10/100)
            ind[0,0] += recorte_ventana
            ind[0,1] -= recorte_ventana
            peso = dat[ind[0,0]:ind[0,1]].mean()
            dat -= peso
        except:
            pass
            
        return dat
    
    """
    data = daDatos[0].data
    """
    
    daCortado = xr.apply_ufunc(ajuste_cero, daDatos,
                   input_core_dims=[['time'] ],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   )#.dropna(dim='time', how='all')
    # daCortado.attrs = daDatos.attrs
    # daCortado.name = daDatos.name
    #daCortado.sel(eje='z').plot.line(x='time', row='ID', col='eje')
    
    return daCortado


def calcula_peso(daDatos, ventana_peso=None, show=False):
    #Con ventana de peso única para todos
    #daPeso = daDatos.sel(eje='z').isel(time=slice(ventana[0], ventana[1])).mean(dim='time')
    
    #Con ventanas personalizadas
    
    if isinstance(ventana_peso, xr.DataArray):
        def peso_indiv_xSD(data, vent0, vent1):
            peso=[]
            peso.append(data[vent0:vent1].mean())
            peso.append(data[vent0:vent1].std())
            # plt.plot(data[vent0:vent1])
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)            
            return np.asarray(peso)
        
        daPeso = xr.apply_ufunc(peso_indiv_xSD, daDatos.sel(eje='z'), ventana_peso.sel(ventana='ini'), ventana_peso.sel(ventana='fin'),
                       input_core_dims=[['time'], [], []],
                       output_core_dims=[['stat']],
                       #exclude_dims=set(('time',)),
                       vectorize=True).assign_coords(stat=['media', 'sd'])
     
    if show:
        def dibuja_peso(x,y, **kwargs): #de momento no funciona
            print(x)#kwargs['data'])
            #plt.plot()
        g=daDatos.sel(eje='z').plot.line(col='ID', col_wrap=3, hue='repe', sharey=False)
        #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
        #g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')
        col=['C0', 'C1', 'C2']
        for h, ax in enumerate(g.axes): #extrae cada fila
            for i in range(len(ax)): #extrae cada axis (gráfica)     
                if g.name_dicts[h, i]==None: #en los cuadros finales que sobran de la cuadrícula se sale
                    break
                try:
                    idn = g.data.loc[g.name_dicts[h, i]].ID
                    #print('peso=', daPeso.sel(ID=idn).data)#idn)
                    #Rango medida peso
                    #ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.frec)], g.data.time[int(ventana[1]*self.datos.frec)], alpha=0.2, color='C1')
                    for j in daDatos.repe:
                        ax[i].axvspan(ventana_peso.sel(ID=idn, repe=j, ventana='ini')/daDatos.frec, ventana_peso.sel(ID=idn, repe=j, ventana='fin')/daDatos.frec, alpha=0.2, color=col[j.data-1])
                        ax[i].axhline(daPeso.sel(ID=idn, repe=j, stat='media').data, color=col[j.data-1], lw=1, ls='--', alpha=0.6)
                    #Líneas peso
                    #ax[i].hlines(daPeso.sel(ID=idn, stat='media').data, xmin=daDatos.time[0], xmax=daDatos.time[-1], colors=col, lw=1, ls='--', alpha=0.6)
                except:
                    print('Error al pintar peso en', g.name_dicts[h, i], h,i)
    return daPeso



def detecta_despegue_aterrizaje(daDatos, tipo_test, umbral=10.0, show=False):
    def detect_onset_aux(data, ID, repe, **args_func_cortes):
        if np.count_nonzero(~np.isnan(data))==0:
            return np.array([np.nan, np.nan])
        #plt.plot(data)
        # plt.show()
        # print(ID, repe)
        ind = detect_onset(-data, **args_func_cortes)
        if len(ind)<1:
            print(f'No se ha encontrado despegue/aterrizaje en archivo ID {ID}, repe {repe}')
            return np.array([np.nan, np.nan])
        # if tipo_test == 'CMJ':
        #     ind=ind[0] #coge el primer bloque que encuentra
        #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
        # elif tipo_test in ['DJ', 'DJ2P']:
        #     ind=ind[1] #coge el primer bloque que encuentra
        #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
        
        #Independientemente del tipo de salto que sea, se queda con el últimoque en cuentre
        ind=ind[-1]
        
        return ind.astype('float')#[1]
        
    #data = daDatos[0,0,-1].data
    #args_func_cortes = dict(threshold=-umbral, n_above=50, show=True)
    daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daDatos.ID, daDatos.repe,
                   input_core_dims=[['time'],[],[]],
                   output_core_dims=[['evento']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   kwargs=dict(threshold=-umbral, n_above=100, show=show)
                   ).assign_coords(evento=['despegue', 'aterrizaje'])
    #Comprobaciones
    #daDatos.sel(eje='z').isel(time=daCorte.sel(evento='despegue')-1) #despegue cuando ya ha pasado por debajo del umbral
    #daDatos.sel(eje='z').isel(time=daCorte.sel(evento='aterrizaje')-1) #aterrizaje cuando ya ha pasado por debajo del umbral
    return daCorte



def detecta_ini_mov(daDatos, tipo_test, daPeso=None, daDespegue=None, SDx=10, show=False):
    if tipo_test=='DJ':
        def detect_onset_aux(data, **args_func_cortes):
            #plt.plot(data)
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            ini = detect_onset(-data, **args_func_cortes)[0]
            return float(ini[1] + 1) #+1 para que se quede con el que ya ha pasado el umbral
    
        
        #data= daDatos[0,0,2].data
        #args_func_cortes = dict(threshold=-10.0, n_above=50, show=False)
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'),
                       input_core_dims=[['time']],
                       #output_core_dims=[['peso']],
                       #exclude_dims=set(('time',)),
                       vectorize=True,
                       kwargs=dict(threshold=-daPeso, n_above=50, show=False)
                       )
        #daDatos.sel(eje='z').isel(time=daCorte-1)
        
    elif tipo_test=='CMJ':
        def detect_iniMov_peso_pcto(data, peso, umbral, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-umbral), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:0:-1], threshold=peso, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return float(ini)
     
        #data = daDatos[0,1,-1].data
        #peso = daPeso[0,1].sel(stat='media').data
        #pcto = 10
        #sdpeso = (daPeso[0,1].sel(stat='sd')*SDx).data
        #umbral = (daPeso[0,1].sel(stat='media') - daPeso[0,1].sel(stat='sd')*SDx).data
        #idespegue = daDespegue[0,1].data
        daCorte = xr.apply_ufunc(detect_iniMov_peso_pcto, daDatos.sel(eje='z'), daPeso.sel(stat='media'), daPeso.sel(stat='media')*SDx/100, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        
    elif tipo_test=='DJ2P':
        def detect_iniMov_peso_pcto(data, peso, umbral, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            try:
                #Intenta detectar qué es antes: descenso por debajo del peso o ascenso por encima (dando saltito)
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-umbral), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:0:-1], threshold=peso, n_above=5, show=False)
                ini_abajo = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por debajo del peso
                
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(data[:int(idespegue)], threshold=(peso+umbral), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:0:-1], threshold=peso, n_above=5, show=False)
                ini_arriba = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por encima del peso
                
                ini = np.min([ini_arriba, ini_abajo])
                
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return float(ini)
        """
        data = daDatos[-1,0,-1].data        
        peso = daPeso[1,0].sel(stat='media').data
        pcto = 10
        sdpeso = (daPeso[1,0].sel(stat='sd')*SDx).data
        umbral = daPeso[1,0].sel(stat='media').data*SDx/100
        idespegue = daDespegue[1,0].data
        
        data = daDatos.sel(ID='2PLAT_01_40', eje='z', repe=3).data
        peso = daPeso.sel(ID='2PLAT_01_40', repe=3,stat='media')
        """
        daCorte = xr.apply_ufunc(detect_iniMov_peso_pcto, daDatos.sel(eje='z'), daPeso.sel(stat='media'), daPeso.sel(stat='media')*SDx/100, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        
        
        """
        def detect_iniMov_peso_XSD(data, peso, sdpeso, idespegue):
            #Parte del despegue hacia atrás buscando cuándo supera el umbral del peso - sd*SDx
            #ini = detect_onset(-data[:int(idespegue):-1], threshold=-umbral, n_above=50, show=True)[0]
            #ini = detect_onset(data[int(idespegue)::-1], threshold=umbral, n_above=5, show=True)
            #ini = idespegue - ini[1,0] + 1 #+1 para coger el que ya ha superado el umbral
            
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-sdpeso), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:ini1[0,0]-100:-1], threshold=peso, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha superado el umbral
                
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
        daCorte = xr.apply_ufunc(detect_iniMov_peso_XSD, daDatos.sel(eje='z'), daPeso.sel(stat='media'), daPeso.sel(stat='sd')*SDx, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """
        
        #Comprobaciones
        #daDatos.sel(eje='z').isel(time=daCorte-1) #        
                
    return daCorte


def detecta_fin_mov(daDatos, tipo_test, daPeso=None, daAterrizaje=None, SDx=2):
    def detect_onset_aux(data, umbral, iaterrizaje, ID):
        #print(ID)            
        if np.count_nonzero(~np.isnan(data))==0:
            return np.nan
        fin = detect_onset(data[int(iaterrizaje):], threshold=umbral, n_above=50, show=False)
        try:
            fin = iaterrizaje + fin[1,1] + 1 #+1 para coger el que ya ha superado el umbral            
        except:
            fin = len(data) #por si no encuentra el criterio
        return float(fin)
        
    #data = daDatos[0,1,-1].data
    #umbral = daPeso[0,1,0].data
    #iaterrizaje = daAterrizaje[0,1].data
    daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daPeso.sel(stat='media') , daAterrizaje, daDatos.ID,
                   input_core_dims=[['time'], [], [], []],
                   #output_core_dims=[['peso']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   #kwargs=dict(threshold=10, n_above=50, show=False)
                   )
    
    if False:#tipo_test=='CMJ':
        def detect_onset_aux(data, umbral, iaterrizaje):            
            fin = detect_onset(data[int(iaterrizaje):], threshold=umbral, n_above=50, show=False)[1]
            return iaterrizaje + fin[1] +1 #+1 para coger el que ya ha superado el umbral
        
        #data = daDatos[0,1,-1].data
        #umbral = daPeso[0,1,0].data
        #idespegue = daAterrizaje[0,1].data
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daPeso.sel(stat='media') , daAterrizaje,
                       input_core_dims=[['time'], [], []],
                       #output_core_dims=[['peso']],
                       #exclude_dims=set(('time',)),
                       vectorize=True,
                       #kwargs=dict(threshold=10, n_above=50, show=False)
                       )
                
    return daCorte



def detecta_maxFz(daDatos, tipo_test, daPeso=None, daEventos=None):
    #from detecta import detect_peaks
    if tipo_test in ['CMJ', 'DJ', 'DJ2P']:
        def detect_onset_aux(data, ini, fin):
            try:
                ini=int(ini)
                fin=int(fin)
                ind = float(np.argmax(data[ini:fin]) + ini)
                # plt.plot(data[ini:fin])
                # plt.show()
                #detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                #data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan #por si no encuentra el criterio
            return np.array([ind])
        """      
        data = daDatos[0,1,-1].data
        ini = daEventos[0,1].sel(evento='iniMov').data
        fin = daEventos[0,1].sel(evento='despegue').data
        """
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daEventos.sel(evento='iniMov').data, daEventos.sel(evento='despegue').data,
                                   input_core_dims=[['time'], [], []],
                                   #output_core_dims=[['evento']],
                                   #exclude_dims=set(('evento',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
                                   )#.assign_coords(evento=['minFz'])
        
        return daCorte




def detecta_minFz(daDatos, tipo_test, daPeso=None, daEventos=None):
    #from detecta import detect_peaks
    if tipo_test=='CMJ':
        def detect_onset_aux(data, ini, fin):
            try:
                ini=int(ini)
                fin=int(fin)
                ind = float(np.argmin(data[ini:fin]) + ini)
                # plt.plot(data[ini:fin])
                # plt.show()
                #detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                #data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan #por si no encuentra el criterio
            return np.array([ind])
                
        #data = daDatos[0,1,-1].data
        #ini = daEventos[0,1].sel(evento='iniMov').data
        #fin = daEventos[0,1].sel(evento='maxFlex').data
        
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daEventos.sel(evento='iniMov').data, daEventos.sel(evento='maxFlex').data,
                                   input_core_dims=[['time'], [], []],
                                   #output_core_dims=[['evento']],
                                   #exclude_dims=set(('evento',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
                                   )#.assign_coords(evento=['minFz'])
        
    elif tipo_test in ['DJ', 'DJ2P']:
        def detect_onset_aux(data, fin, **args_func_cortes):
            #plt.plot(data)
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan            
            fin=int(fin)
            data = data[fin::-1]
            ind = detect_onset(-data, **args_func_cortes)[0,0]
            return float(fin - ind) #+1 para que se quede con el que ya ha pasado el umbral
    
        """
        data= daDatos[0,0,2].data
        ini = daEventos[0,0].sel(evento='iniMov').data
        fin = daEventos[0,0].sel(evento='iniImpPositivo').data
        args_func_cortes = dict(threshold=-10.0, n_above=50, show=True)
        """
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daEventos.sel(evento='iniImpPositivo').data,
                       input_core_dims=[['time'],[]],
                       #output_core_dims=[['evento']],
                       #exclude_dims=set(('time',)),
                       vectorize=True,
                       kwargs=dict(threshold=-10, n_above=50, show=False)
                       )
    
    return daCorte
        

        
        
def detecta_ini_fin_impulso(daDatos, tipo_test, daPeso=None, daEventos=None, show=False):
    if tipo_test in ['CMJ', 'DJ']:
        def detect_onset_aux(data, peso, iinimov):
            try:
                ini1 = detect_onset(data[int(iinimov):], threshold=peso, n_above=100, show=False)                
                ind = int(iinimov) + ini1[0]
                ind[1] += 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ini[1]+1] #peso
            except:
                ind = np.array([np.nan, np.nan]) #por si no encuentra el criterio
            return ind.astype('float')            
        """
        data = daDatos[1,0,-1].data
        peso = daPeso[1,0].sel(stat='media').data
        iinimov = daEventos[1,0].data
        """
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daPeso.sel(stat='media').data, daEventos.data,
                                   input_core_dims=[['time'], [], []],
                                   output_core_dims=[['evento']],
                                   #exclude_dims=set(('evento',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
                                   ).assign_coords(evento=['iniImpPositivo', 'finImpPositivo'])
        
    elif tipo_test in ['DJ2P']:
        def detect_onset_aux(data, peso, iinimov):
            try:
                #busca cuándo inicia primer aterrizaje
                ini0 = detect_onset(data, threshold=30.0, n_above=100, show=False)[1,0]
                ini1 = detect_onset(data[ini0:], threshold=peso, n_above=100, show=False)
                ind = ini0 + ini1[0]
                ind[1] += 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ind[0]-1] #peso
            except:
                ind = np.array([np.nan, np.nan]) #por si no encuentra el criterio                
            return ind.astype('float')            
        """
        data = daDatos[1,0,-1].data
        peso = daPeso[1,0].sel(stat='media').data
        iinimov = daIniMov[1,0].data
        """
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daPeso.sel(stat='media').data, daEventos.data,
                                   input_core_dims=[['time'], [], []],
                                   output_core_dims=[['evento']],
                                   #exclude_dims=set(('evento',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
                                   ).assign_coords(evento=['iniImpPositivo', 'finImpPositivo'])
        
    return daCorte
        

def detecta_max_flex(daDatos, tipo_test, v=None, daPeso=None, daEventos=None):
    """
    Para calcular desde 'DJ' tiene que venir reversed y con la velocidad como
    parámetro.
    """
    if tipo_test in ['CMJ', 'DJ', 'DJ2P']:
        def detect_onset_aux(data, ini, fin):
            try:                
                ini=int(ini)
                fin=int(fin)
                ind = detect_onset(data[ini:fin], threshold=0, n_above=10, show=False) #los datos que llegan de velocidad están cortados desde el iniMov
                ind = ind[0,0] + ini
                #ind += ini
                ind=float(ind)
                #data[ind-1] #data[ind]
            except:
                ind = np.nan #por si no encuentra el criterio
            return np.array(ind)
        
        #Calcula la velocidad, OJO, sin haber hecho el ajuste de offsetFz
        if not isinstance(v, xr.DataArray):
            v = calcula_variables(daDatos, daPeso=daPeso, daEventos=daEventos.sel(evento=['iniMov', 'finMov']))['v']
        v = v.sel(eje='z') #se queda solo con eje z la haya calculado aquí o venga calculada del reversed
        #v.plot.line(x='time', col='ID', col_wrap=5, sharey=False)
        """
        data = v[1,0].data
        peso = daPeso[1,0].data
        ini = daEventos[1,0].sel(evento='iniMov').data
        fin = daEventos[1,0].sel(evento='despegue').data
        """
        daCorte = xr.apply_ufunc(detect_onset_aux, v, daEventos.sel(evento='iniImpPositivo'), daEventos.sel(evento='despegue'),
                                   input_core_dims=[['time'], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
                
    return daCorte



def detecta_max_flex_desdeV(daDatos, tipo_test, daInstantes=None):
    from detecta import detect_peaks
    if tipo_test=='DJ':
        return
        
    elif tipo_test=='CMJ':
        def detect_onset_aux(data, peso, ini, fin):
            try:
                ini=int(ini)
                fin=int(fin)
                ind = detect_onset(data[int(iinimov):], threshold=peso, n_above=10, show=False)
                #ind += ini
                #data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan #por si no encuentra el criterio
            return ind
            
        
        #data = daDatos[0,1,-1].data
        #ini = daInstantes[0,1].sel(evento='iniMov').data
        #fin = daInstantes[0,1].sel(evento='despegue').data
        
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daPeso.sel(stat='media').data, daInstantes.sel(evento='iniMov'), daInstantes.sel(evento='despegue'),
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """
        def detect_iniMov_peso_XSD(data, peso, sdpeso, idespegue):
            #Parte del despegue hacia atrás buscando cuándo supera el umbral del peso - sd*SDx
            #ini = detect_onset(-data[:int(idespegue):-1], threshold=-umbral, n_above=50, show=True)[0]
            #ini = detect_onset(data[int(idespegue)::-1], threshold=umbral, n_above=5, show=True)
            #ini = idespegue - ini[1,0] + 1 #+1 para coger el que ya ha superado el umbral
            
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-sdpeso), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:ini1[0,0]-100:-1], threshold=peso, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha superado el umbral
                
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
        daCorte = xr.apply_ufunc(detect_iniMov_peso_XSD, daDatos.sel(eje='z'), daPeso.sel(stat='media'), daPeso.sel(stat='sd')*SDx, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['peso']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """
        
        #Comprobaciones
        #daDatos.sel(eje='z').isel(time=daCorte-1) #
        
                
    return daCorte


def graficas_eventos(daDatos, daEventos, daPeso=None, ruta_trabajo=None, nom_archivo_graf_global=None):
    import seaborn as sns
    
    #Si no se incluye nombre archivo no guarda el pdf
    if nom_archivo_graf_global!=None:
        if not isinstance(ruta_trabajo, Path):
            ruta_trabajo = Path(ruta_trabajo)
        nompdf = (ruta_trabajo / nom_archivo_graf_global).with_suffix('.pdf')
        pdf_pages = PdfPages(nompdf)
    if 'eje' in daDatos.dims: #por si se envía un da filtrado por eje
        daDatos=daDatos.sel(eje='z')

    def dibuja_X(x,y, color, **kwargs):      
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        #print(y, ID, repe, color, kwargs.keys())
        #plt.vlines(daEventos.sel(ID=ID, repe=repe)/daDatos.frec, ymin=kwargs['data'].loc[:,'Fuerza'].min(), ymax=kwargs['data'].loc[:,'Fuerza'].max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
        #Líneas del peso
        if isinstance(daPeso, xr.DataArray):
            plt.axhline(daPeso.sel(ID=ID, repe=repe, stat='media').data, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.6)
       
        for ev in daEventos.sel(ID=ID, repe=repe).evento:
            if str(ev.data) not in ['iniAnalisis', 'finAnalisis']: #se salta estos dos porque el array viene cortado por sus valores y tienen escala distinta
                #print(str(ev.data))
                #print(daEventos.sel(ID=ID, repe=repe,evento=ev))
            # for num, ev in daEventos.sel(ID=ID, repe=repe).groupby('evento'):
            #     print('\n',num)
                if not np.isnan(daEventos.sel(ID=ID, repe=repe, evento=ev)): #si existe el evento
                    plt.axvline(x=daEventos.sel(ID=ID, repe=repe, evento=ev)/daDatos.frec, c=col[str(ev.data)], lw=0.5, ls='--', dashes=(5, 5), dash_capstyle='round', alpha=0.5)
                    y_texto = plt.gca().get_ylim()[1] if str(ev.data) not in ['minFz', 'despegue', 'maxFz'] else plt.gca().get_ylim()[1]*0.8
                    plt.text(daEventos.sel(ID=ID, repe=repe, evento=ev).data/daDatos.frec, y_texto, ev.data,
                             ha='right', va='top', rotation='vertical', c='k', alpha=0.6, fontsize=8, 
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), transform=plt.gca().transData)
                
    dfDatos = daDatos.to_dataframe().reset_index().drop(columns='eje')
    
    col={'iniMov':'b', 'finMov':'b', 'iniImpPositivo':'orange', 'maxFz':'brown', 'minFz':'cyan', 'maxFlex':'g', 'finImpPositivo':'orange', 'despegue':'r', 'aterrizaje':'r', 'iniAnalisis':'k', 'finAnalisis':'k'}#['C0', 'C1', 'C2']
    
    """
    def fun(x,y): #prueba para dibujar con xarray directamente
        print(x,y)
    g=daDatos.sel(eje='z').plot.line(x='time', row='ID', col='repe')
    g.map_dataarray_line(fun, x='ID', y=daDatos, hue='repe')
    """
    # g = sns.relplot(data=dfDatos, x='time', y='Fuerza', col='ID', col_wrap=4, hue='repe',
    #                 estimator=None, ci=95, units='repe',
    #                 facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
    #                 palette=sns.color_palette(col), alpha=0.7)
    
    g = sns.relplot(data=dfDatos[dfDatos['Fuerza'].notnull()], x='time', y='Fuerza', row='ID', col='repe', #ATENCIÓN en esta versión de seaborn (12.1) falla con datos nan, por eso se seleccionan los notnull()
                    #estimator=None, ci=95, units='repe',
                    facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
                    alpha=0.7, aspect=1.5) #palette=sns.color_palette(col), 
    
    g.map_dataframe(dibuja_X, x='time', y='Fuerza', lw=0.25, alpha=0.3)
      
    """
    def dibuja_xr(x,y, **kwargs):
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        print(y, ID, repe, color, kwargs.keys())
    
    g=daDatos.sel(eje='z').plot.line(x='time', col='ID', col_wrap=4, hue='repe', sharey=False)
    #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
    g.map_dataarray_line(dibuja_xr, x='time', y=None, hue='repe')#, y='trial')
    col=['C0', 'C1', 'C2']
    for h, ax in enumerate(g.axs): #extrae cada fila
        for i in range(len(ax)): #extrae cada axis (gráfica)     
            try:
                idn = g.data.loc[g.name_dicts[h, i]].ID
                #print('peso=', daPeso.sel(ID=idn).data)#idn)
                #Rango medida peso
                #ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.frec)], g.data.time[int(ventana[1]*self.datos.frec)], alpha=0.2, color='C1')
                #for j in daDatos.repe:
                for e in daEventos.sel(ID=idn):
                    #print(e)
                    ax[i].vlines(e/daDatos.frec, ymin=g.data.sel(ID=idn).min(), ymax=g.data.sel(ID=idn).max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
            except:
                print("No va el", h,i)
    """
    
    if nom_archivo_graf_global!=None:
        pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print('Guardada la gráfica', nompdf)
        

def recorta_ventana_analisis(daDatos, ventana_analisis):
    def corta_ventana(datos, ini, fin):
        #print(datos.shape, ini,fin)        
        d2 = np.full(datos.shape, np.nan) #rellena con nan al final para que tengan mismo tamaño
        try:
            d2[:int(fin)-int(ini)] = datos[int(ini):int(fin)]
        except:
            pass
        return d2 #datos[int(ini):int(fin)]
    
    daCortado = xr.apply_ufunc(corta_ventana, daDatos, ventana_analisis.isel(evento=0).sel(ID=daDatos.ID, repe=daDatos.repe), ventana_analisis.isel(evento=1).sel(ID=daDatos.ID, repe=daDatos.repe),
                   input_core_dims=[['time'], [], []],
                   output_core_dims=[['time']],
                   exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   ).assign_coords({'time':daDatos.time}).dropna(dim='time', how='all')

    daCortado = daCortado.astype(daDatos.dtype)
    daCortado.attrs = daDatos.attrs
    daCortado.name = daDatos.name
    #daCortado.plot.line(x='time', row='ID', col='eje')
    return daCortado


def ajusta_offsetFz_vuelo(daDatos, tipo_test, umbral=20.0, pcto_ventana=5, show=False):
    #busca despegue y aterrizaje provisionales
    vuelo = detecta_despegue_aterrizaje(daDatos, tipo_test, umbral=umbral)
    #reduce la ventana un poco para evitar los rebotes posibles del filtrado
    recorte_ventana = ((vuelo.loc[dict(evento='aterrizaje')]-vuelo.loc[dict(evento='despegue')])*pcto_ventana/100).astype('int32')
    vuelo.loc[dict(evento='despegue')] += recorte_ventana
    vuelo.loc[dict(evento='aterrizaje')] -= recorte_ventana
    
    offset_vuelo = recorta_ventana_analisis(daDatos, vuelo).mean(dim='time')
    #recorta_ventana_analisis(daDatos, vuelo).sel(eje='x').plot.line(x='time', col='ID', col_wrap=4)
    #offset_vuelo.sel(eje='z').plot.line(col='ID', col_wrap=4, hue='repe')
    #daDatos -= offset_vuelo
    with xr.set_options(keep_attrs=True):
        #datos = daDatos - offset_vuelo
        daDatos = daDatos - offset_vuelo
        
    if show:
        recorta_ventana_analisis(daDatos, vuelo.sel(evento=['despegue', 'aterrizaje'])).plot.line(row='ID', col='repe', hue='eje', sharey=False)
      
        
    return daDatos#datos

def reset_Fz_vuelo(daDatos, tipo_test, umbral=20.0, pcto_ventana=5, show=False): #, ventana_vuelo=None):
    vuelo = detecta_despegue_aterrizaje(daDatos, tipo_test, umbral=umbral)
    #reduce la ventana un poco para evitar los rebotes posibles del filtrado
    recorte_ventana = ((vuelo.loc[dict(evento='aterrizaje')]-vuelo.loc[dict(evento='despegue')])*pcto_ventana/100).astype('int32')
    vuelo.loc[dict(evento='despegue')] += recorte_ventana
    vuelo.loc[dict(evento='aterrizaje')] -= recorte_ventana
           
    with xr.set_options(keep_attrs=True):
        daDatos = xr.where(~daDatos.isnull(), daDatos.where(daDatos.sel(eje='z')>umbral, 0.0), daDatos)
        daDatos.time.attrs['units'] ='s' #por alguna razón lo cambiaba a newtons
        #daDatos.plot.line(row='ID', col='repe', hue='eje', sharey=False)
    
    if show:
        recorta_ventana_analisis(daDatos, vuelo+[-50, 50]).plot.line(row='ID', col='repe', hue='eje', sharey=False)
    
    return daDatos
    
    """
    #Con ufunc necesario tener despegue y aterrizaje
    def reset_ventana(data, ini, fin):
        dat=data.copy()
        #print(datos.shape, ini,fin)  
        ini=int(ini)
        fin=int(fin)
        dat[ini:fin] = np.full(fin-ini, 0.0)
        return dat
    
    
    # data = daDatos[0,1].sel(eje='z').data
    # ini = daEventos[0,1].sel(evento='iniMov')
    # fin = daEventos[0,1].sel(evento='finMov')
    
    
    daCortado = xr.apply_ufunc(reset_ventana, daDatos, ventana_vuelo.isel(evento=0).sel(ID=daDatos.ID, repe=daDatos.repe), ventana_vuelo.isel(evento=1).sel(ID=daDatos.ID, repe=daDatos.repe),
                   input_core_dims=[['time'], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   ).dropna(dim='time', how='all')
    daCortado.attrs = daDatos.attrs
    daCortado.name = daDatos.name
    #daCortado.sel(eje='z').plot.line(x='time', row='ID', col='eje')
    return daCortado
    """


def calcula_variables(daDatos, daPeso=None, daEventos=None):
    #se puede integrar directamente con ufunc, pero no deja meter parámetro initial=0 y devuelve con un instante menos
    def integra(data,time,peso,ini,fin):
        # if np.count_nonzero(~np.isnan(data))==0:
        #     return np.nan
        dat = np.full(len(data), np.nan)        
        try:
            ini=int(ini)
            fin=int(fin)
            #plt.plot(data[ini:fin])
            dat[ini:fin] = integrate.cumtrapz(data[ini:fin]-peso, time[ini:fin], initial=0)
            #plt.plot(dat)
        except:
            pass #dat = np.full(len(data), np.nan)            
        return dat
    
    """
    data = daDatos[0,0].data #.sel(eje='z').data
    time = daDatos.time.data
    peso=daPeso[0,0].sel(stat='media').data
    ini = daEventos[0,0].sel(evento='iniMov').data
    fin = daEventos[0,0].sel(evento='finMov').data
    plt.plot(data[int(ini):int(fin)])
    """
    daV = (xr.apply_ufunc(integra, daDatos, daDatos.time, daPeso.sel(stat='media'), daEventos.isel(evento=0), daEventos.isel(evento=1), #eventos o y 1 para que sirva con reversed, se pasa iniMov y finMov en el orden adecuado
                   input_core_dims=[['time'], ['time'], [], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   join='exact',
                   ) / (daPeso.sel(stat='media')/9.8)).drop_vars('stat')
    #daV.plot.line(x='time', col='ID', col_wrap=4)
    
    
    daS = xr.apply_ufunc(integra, daV, daDatos.time, 0, daEventos.isel(evento=0), daEventos.isel(evento=1),
                   input_core_dims=[['time'], ['time'], [], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   )
    #daS.plot.line(x='time', col='ID', col_wrap=4)
    
    daP = daDatos * daV
    daRFD = daDatos.differentiate(coord='time')
    
    daV.attrs['units']='m/s'
    daS.attrs['units']='m'
    daP.attrs['units']='W'
    daRFD.attrs['units']='N/s'
     
    return xr.Dataset({'F':daDatos / daPeso.sel(stat='media').drop_vars('stat'), #F normalizada
                       'v':daV, 's':daS, 'P':daP, 'RFD':daRFD}
                      ).astype(daDatos.dtype).assign_attrs({'frec':daDatos.frec})
    

def calcula_results(daCinet=None, dsCinem=None, daPeso=None, daResults=None, daEventos=None):
    if not isinstance(daResults, xr.DataArray):
        daResults = (xr.DataArray(coords=[['tVuelo', 'hTVuelo', 'FzMax', 'FzMin', 'FzTransicion',
                                              'vDespegue', 'vAterrizaje', 'hVDespegue', 'vMax', 'vMin',
                                              'sDespegue', 'sAterrizaje', 'sIniMov', 'sFinMov', 'sMax', 'sMin', 'sDifDespAter', 'hS',
                                              'PMax', 'PMin',
                                              'RFDMax', 'RFDMed',
                                              'impNegDescenso', 'ImpPositDescenso', 'ImpPositAscenso', 'ImpNegAscenso',
                                              ]], dims=('var'))
                          .expand_dims({'ID':daCinet.coords['ID'], 'repe':daCinet.coords['repe']}) 
                         ).copy()
    try:    
        daCinet = daCinet.sel(eje='z') #en principio solo interesa el eje z
    except:
        pass
    
    
    daResults.loc[dict(var='tVuelo')] = (daEventos.sel(evento='aterrizaje') - daEventos.sel(evento='despegue')) / dsCinem.frec
    daResults.loc[dict(var='hTVuelo')] = 9.81 * daResults.loc[dict(var='tVuelo')]**2 / 8
    
    daResults.loc[dict(var='FzMax')] = recorta_ventana_analisis(daCinet, daEventos.sel(evento=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(var='FzMin')] = recorta_ventana_analisis(daCinet, daEventos.sel(evento=['iniMov', 'despegue'])).min(dim='time')
    daResults.loc[dict(var='FzTransicion')] = daCinet.sel(time=daEventos.sel(evento='maxFlex')/dsCinem.frec, method='nearest')
    #Velocidades
    daResults.loc[dict(var='vDespegue')] = dsCinem['v'].sel(time=daEventos.sel(evento='despegue')/dsCinem.frec, method='nearest')
    daResults.loc[dict(var='vAterrizaje')] = dsCinem['v'].sel(time=daEventos.sel(evento='aterrizaje')/dsCinem.frec, method='nearest')
    daResults.loc[dict(var='hVDespegue')] = daResults.loc[dict(var='vDespegue')]**2/(2*9.81)
    daResults.loc[dict(var='vMax')] = recorta_ventana_analisis(dsCinem['v'], daEventos.sel(evento=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(var='vMin')] = recorta_ventana_analisis(dsCinem['v'], daEventos.sel(evento=['iniMov', 'despegue'])).min(dim='time')
    
    #Posiciones
    daResults.loc[dict(var='sDespegue')] = dsCinem['s'].sel(time=daEventos.sel(evento='despegue')/dsCinem.frec, method='nearest')
    daResults.loc[dict(var='sAterrizaje')] = dsCinem['s'].sel(time=daEventos.sel(evento='aterrizaje')/dsCinem.frec, method='nearest')
    daResults.loc[dict(var='sIniMov')] = dsCinem['s'].sel(time=daEventos.sel(evento='iniMov')/dsCinem.frec, method='nearest')
    daResults.loc[dict(var='sFinMov')] = dsCinem['s'].sel(time=(daEventos.sel(evento='finMov')-1)/dsCinem.frec, method='nearest')
    daResults.loc[dict(var='sMax')] = recorta_ventana_analisis(dsCinem['s'], daEventos.sel(evento=['despegue', 'aterrizaje'])).max(dim='time')
    daResults.loc[dict(var='sMin')] = recorta_ventana_analisis(dsCinem['s'], daEventos.sel(evento=['iniMov', 'despegue'])).min(dim='time')
    daResults.loc[dict(var='sDifDespAter')] = daResults.loc[dict(var='sDespegue')] - daResults.loc[dict(var='sAterrizaje')]
    
    daResults.loc[dict(var='hS')] = daResults.loc[dict(var='sMax')] - daResults.loc[dict(var='sDespegue')]
    
    #Potencias
    daResults.loc[dict(var='PMax')] = recorta_ventana_analisis(dsCinem['P'], daEventos.sel(evento=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(var='PMin')] = recorta_ventana_analisis(dsCinem['P'], daEventos.sel(evento=['iniMov', 'despegue'])).min(dim='time')
    
    #RFD
    daResults.loc[dict(var='RFDMax')] = recorta_ventana_analisis(dsCinem['RFD'], daEventos.sel(evento=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(var='RFDMed')] = (daCinet.sel(time=daEventos.sel(evento='maxFlex')/dsCinem.frec, method='nearest') - daCinet.sel(time=daEventos.sel(evento='minFz')/dsCinem.frec, method='nearest')) / ((daEventos.sel(evento='maxFlex') - daEventos.sel(evento='minFz'))/dsCinem.frec)
    
    #Impulsos
    def integra_completo(daDatos, daEventos):
        def integra(data,time,ini,fin, ID,rep):
            if np.isnan(ini) or np.isnan(fin):
                return np.nan
            ini=int(ini)
            fin=int(fin)
            dat = integrate.cumtrapz(data[ini:fin], time[ini:fin], initial=0)[-1]                
            return dat
        """
        data = daDatos[0,1].data
        time = daDatos.time.data
        ini = daEventos[0,1].isel(evento=0).data
        fin = daEventos[0,1].isel(evento=1).data
        """
        daInt = xr.apply_ufunc(integra, daDatos, daDatos.time, daEventos.isel(evento=0), daEventos.isel(evento=1), daDatos.ID, daDatos.repe,
                       input_core_dims=[['time'], ['time'], [], [], [], []],
                       #output_core_dims=[['time']],
                       #exclude_dims=set(('time',)),
                       vectorize=True,
                       join='exact',
                       )
        return daInt
        
    # ImpNegDescenso
    daResults.loc[dict(var='impNegDescenso')] = integra_completo(daCinet - daPeso.sel(stat='media').drop_vars('stat'), daEventos=daEventos.sel(evento=['iniMov', 'iniImpPositivo']))
    daResults.loc[dict(var='ImpPositDescenso')] = integra_completo(daCinet - daPeso.sel(stat='media').drop_vars('stat'), daEventos=daEventos.sel(evento=['iniImpPositivo', 'maxFlex']))
    daResults.loc[dict(var='ImpPositAscenso')] = integra_completo(daCinet - daPeso.sel(stat='media').drop_vars('stat'), daEventos=daEventos.sel(evento=['maxFlex', 'finImpPositivo']))
    daResults.loc[dict(var='ImpNegAscenso')] = integra_completo(daCinet - daPeso.sel(stat='media').drop_vars('stat'), daEventos=daEventos.sel(evento=['finImpPositivo', 'despegue']))
    
    
    return daResults
    



# =============================================================================
# PRUEBA METIDO EN UNA CLASE
class trata_fuerzas_saltos:
    def __init__(self, datos=None, tipo_test='CMJ'):
        self.datos = datos
        self.tipo_test = tipo_test
        self.peso=None
        
    
    def carga_preprocesados(self, ruta_trabajo, nomArchivoPreprocesado):
        if Path((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix('.nc')).is_file():
            tpo = time.time()
            self.datos = xr.load_dataarray((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix('.nc')).sel(tipo=self.tipo_test)
            print('\nCargado archivo preprocesado ', nomArchivoPreprocesado + '_Vicon.nc en {0:.3f} s.'.format(time.time()-tpo))
        else: 
            raise Exception('No se encuentra el archivo Vicon preprocesado')
            
    def calcula_peso(self, ventana=[100, 600], show=False):
        self.peso = self.datos.sel(eje='z').isel(time=slice(ventana[0], ventana[1])).mean(dim='time')
        
        if show:
            def dibuja_peso(x,y, **kwargs): #de momento no funciona
                print(x)#kwargs['data'])
                #plt.plot()
            g=self.datos.sel(eje='z').plot.line(col='ID', col_wrap=4, hue='trial', sharey=False)
            #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
            #g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')
            
            for h, ax in enumerate(g.axes): #extrae cada fila
                for i in range(len(ax)): #extrae cada axis (gráfica)     
                    try:
                        idn = g.data.loc[g.name_dicts[h, i]].ID
                        #print('peso=', self.peso.sel(ID=idn).data)#idn)
                        #Rango medida peso
                        #ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.frec)], g.data.time[int(ventana[1]*self.datos.frec)], alpha=0.2, color='C1')
                        ax[i].axvspan((len(self.datos.time) + ventana[0])/self.datos.frec, (len(self.datos.time) + ventana[1])/self.datos.frec, alpha=0.2, color='C1')
                        #Líneas peso
                        ax[i].hlines(self.peso.sel(ID=idn).data, xmin=self.datos.time[0], xmax=self.datos.time[-1], colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6)
                    except:
                        print("No va el", h,i)
# =============================================================================


# =============================================================================
# %%
# =============================================================================
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import xarray as xr
    
    ruta_trabajo = Path('F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros')
    nom_archivo_preprocesado = 'PotenciaDJ_Preprocesado'
    
    
    daCMJ = carga_preprocesados(ruta_trabajo, nom_archivo_preprocesado, tipo_test='CMJ')
    
    
    
    
    ##################
    #DJ
    ##################
    daDJ = carga_preprocesados(ruta_trabajo, nom_archivo_preprocesado, tipo_test='DJ')
    
    daVentanasPeso = (xr.DataArray(data=[6500, 7000], coords=[['ini', 'fin']], dims=('ventana'))
                      .expand_dims({'ID':daDJ.coords['ID'], 'repe':daDJ.coords['repe']}) 
                     ).copy()
    daPeso = calcula_peso(daDJ, ventana_peso=daVentanasPeso, show=True)
    daVentanasPeso.loc[dict(ID='13', repe=2)] = [6000, 6500]
    
    #Ajustes de peso puntuales
    
    
    
    daDJ_norm = daDJ/daPeso
    daDJ_norm.sel(eje='z').plot.line(x='time', col='ID', col_wrap=4)    







    
    """
    # =============================================================================
    # PRUEBAS COMO CLASE
    # =============================================================================
    ruta_trabajo = Path('F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros')
    nomArchivoPreprocesado = 'PotenciaDJ_Preprocesado'
    
    dj = trata_fuerzas_saltos(tipo_test='DJ')
    
    dj.carga_preprocesados(ruta_trabajo, nomArchivoPreprocesado)
    dj.datos
    dj.tipo_test
    
    dj.calcula_peso(ventana=[-1500, -1000], show=True)
    dj.peso.sel(ID='01', trial='1')
    """
    
    
    
    