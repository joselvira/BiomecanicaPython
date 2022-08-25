# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:33:55 2022

@author: josel
"""
"""
Clase con funciones para tratar fuerzas de saltos
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time #para cuantificar tiempos de procesado

from pathlib import Path

from detecta import detect_onset


def carga_preprocesados(ruta_trabajo, nomArchivoPreprocesado, tipo_test):
    if Path((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix('.nc')).is_file():
        tpo = time.time()
        daDatos = xr.load_dataarray((ruta_trabajo / (nomArchivoPreprocesado)).with_suffix('.nc')).sel(tipo=tipo_test)
        print('\nCargado archivo preprocesado ', nomArchivoPreprocesado + '_Vicon.nc en {0:.3f} s.'.format(time.time()-tpo))
    else: 
        raise Exception('No se encuentra el archivo preprocesado')
    return daDatos


def calcula_peso(daDatos, ventana_peso=None, show=False):
    #Con ventana de peso única para todos
    #daPeso = daDatos.sel(eje='z').isel(time=slice(ventana[0], ventana[1])).mean(dim='time')
    
    #Con ventanas personalizadas
    
    if isinstance(ventana_peso, xr.DataArray):
        def peso_indiv_xSD(data, vent0, vent1):
            peso=[]
            peso.append(data[vent0:vent1].mean())
            peso.append(data[vent0:vent1].std())
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
        g=daDatos.sel(eje='z').plot.line(col='ID', col_wrap=4, hue='repe', sharey=False)
        #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
        #g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')
        col=['C0', 'C1', 'C2']
        for h, ax in enumerate(g.axes): #extrae cada fila
            for i in range(len(ax)): #extrae cada axis (gráfica)     
                try:
                    idn = g.data.loc[g.name_dicts[h, i]].ID
                    #print('peso=', daPeso.sel(ID=idn).data)#idn)
                    #Rango medida peso
                    #ax[i].axvspan(g.data.time[int(ventana[0]*self.datos.frec)], g.data.time[int(ventana[1]*self.datos.frec)], alpha=0.2, color='C1')
                    for j in daDatos.repe:
                        ax[i].axvspan(ventana_peso.sel(ID=idn, repe=j, ventana='ini')/daDatos.frec, ventana_peso.sel(ID=idn, repe=j, ventana='fin')/daDatos.frec, alpha=0.2, color=col[j.data-1])
                    #Líneas peso
                    ax[i].hlines(daPeso.sel(ID=idn).data, xmin=daDatos.time[0], xmax=daDatos.time[-1], colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6)
                except:
                    print("No va el", h,i)
    return daPeso


def detecta_ini_mov(daDatos, tipo_test, daPeso=None, daDespegue=None, SDx=10):
    if tipo_test=='DJ':
        def detect_onset_aux(data, **args_func_cortes):
            ini = detect_onset(-data, **args_func_cortes)[0]
            return ini[1] + 1 #+1 para que se quede con el que ya ha pasado el umbral
    
        
        #data= daDatos[0,0,0].data
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
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(peso-umbral), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el peso
                ini2 = detect_onset(data[ini1[0,0]:0:-1], threshold=peso, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
            
        
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
    def detect_onset_aux(data, umbral, iaterrizaje):            
        fin = detect_onset(data[int(iaterrizaje):], threshold=umbral, n_above=50, show=False)
        try:
            fin = iaterrizaje + fin[1,1] + 1 #+1 para coger el que ya ha superado el umbral            
        except:
            fin = len(data) #por si no encuentra el criterio
        return fin
        
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



def detecta_despegue_aterrizaje(daDatos, tipo_test, eventos=None, umbral=10.0, show=False):
    def detect_onset_aux(data, **args_func_cortes):
        ini = detect_onset(-data, **args_func_cortes)
        if tipo_test == 'CMJ':
            ini=ini[0] #coge el primer bloque que encuentra
            ini[1]+=1 #para que el aterrizaje coincida con pasado umbral
        elif tipo_test == 'DJ':
            ini=ini[1] #coge el primer bloque que encuentra
            ini[1]+=1 #para que el aterrizaje coincida con pasado umbral
        return ini#[1]
        
    #data = daDatos[0,1,-1].data
    #args_func_cortes = dict(threshold=-umbral, n_above=50, show=True)
    daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'),
                   input_core_dims=[['time']],
                   output_core_dims=[['evento']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   kwargs=dict(threshold=-umbral, n_above=50, show=False)
                   ).assign_coords(evento=['despegue', 'aterrizaje'])
    #Comprobaciones
    #daDatos.sel(eje='z').isel(time=daCorte.sel(evento='despegue')-1) #despegue cuando ya ha pasado por debajo del umbral
    #daDatos.sel(eje='z').isel(time=daCorte.sel(evento='aterrizaje')-1) #aterrizaje cuando ya ha pasado por debajo del umbral
    return daCorte


def detecta_ini_fin_impulso(daDatos, tipo_test, daPeso=None, daIniMov=None, show=False):
    if tipo_test=='DJ':
        return
    
    elif tipo_test=='CMJ':
        def detect_onset_aux(data, peso, iinimov):
            try:
                ini1 = detect_onset(data[int(iinimov):], threshold=peso, n_above=10, show=False)                
                ini = int(iinimov) + ini1[0]
                ini[1] += 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ini[1]+1] #peso
            except:
                ini = np.array([np.nan, np.nan]) #por si no encuentra el criterio
            return ini
            
        
        #data = daDatos[0,1,-1].data
        #peso = daPeso[0,1].sel(stat='media').data
        #iinimov = daIniMov[0,1].data
        daCorte = xr.apply_ufunc(detect_onset_aux, daDatos.sel(eje='z'), daPeso.sel(stat='media').data, daIniMov.data,
                                   input_core_dims=[['time'], [], []],
                                   output_core_dims=[['evento']],
                                   #exclude_dims=set(('evento',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=daPeso.sel(stat='media'), n_above=50, show=False)
                                   ).assign_coords(evento=['iniImpPositivo', 'finImpPositivo'])
    return daCorte
        

def detecta_max_flex(daDatos, tipo_test, daPeso=None, daEventos=None):
    if tipo_test=='DJ':
        return
        
    elif tipo_test=='CMJ':
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
        v = calcula_variables(daDatos, daPeso=daPeso, daEventos=daEventos)['v']
        
        #data = v[0,1].data
        #peso = daPeso[0,1].data
        #ini = daEventos[0,1].sel(evento='iniMov').data
        #fin = daEventos[0,1].sel(evento='despegue').data
                
        daCorte = xr.apply_ufunc(detect_onset_aux, v, daEventos.sel(evento='iniMov'), daEventos.sel(evento='despegue'),
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


def graficas_eventos(daDatos, daEventos):
    import seaborn as sns
    
    def dibuja_X(x,y, color, **kwargs):      
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        #print(y, ID, repe, color, kwargs.keys())
        #plt.vlines(daEventos.sel(ID=ID, repe=repe)/daDatos.frec, ymin=kwargs['data'].loc[:,'Fuerza'].min(), ymax=kwargs['data'].loc[:,'Fuerza'].max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
        for ev in daEventos.sel(ID=ID, repe=repe).evento:
            if str(ev.data) not in ['iniAnalisis', 'finAnalisis']: #se salta estos dos porque el array viene cortado pr sus valores y tienen escala distinta
                #print(str(ev.data))
                #print(daEventos.sel(ID=ID, repe=repe,evento=ev))
            # for num, ev in daEventos.sel(ID=ID, repe=repe).groupby('evento'):
            #     print('\n',num)
                if not np.isnan(daEventos.sel(ID=ID, repe=repe, evento=ev)): #si existe el evento
                    plt.axvline(x=daEventos.sel(ID=ID, repe=repe, evento=ev)/daDatos.frec, c=col[str(ev.data)], ls='--', dashes=(5, 5), dash_capstyle='round', alpha=0.5)
                    plt.text(daEventos.sel(ID=ID, repe=repe, evento=ev).data/daDatos.frec, plt.gca().get_ylim()[1], ev.data,
                             ha='right', va='top', rotation='vertical', c='k', alpha=0.6, fontsize=10, 
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), transform=plt.gca().transData)
                
    dfDatos = daDatos.sel(eje='z').to_dataframe().drop(columns='eje').reset_index()
    
    col={'iniMov':'b', 'finMov':'b', 'iniImpPositivo':'orange', 'maxFlex':'g', 'finImpPositivo':'orange', 'despegue':'r', 'aterrizaje':'r', 'iniAnalisis':'k', 'finAnalisis':'k'}#['C0', 'C1', 'C2']
    
    # g = sns.relplot(data=dfDatos, x='time', y='Fuerza', col='ID', col_wrap=4, hue='repe',
    #                 estimator=None, ci=95, units='repe',
    #                 facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
    #                 palette=sns.color_palette(col), alpha=0.7)
    g = sns.relplot(data=dfDatos, x='time', y='Fuerza', row='ID', col='repe',
                    estimator=None, ci=95, units='repe',
                    facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
                    alpha=0.7, aspect=1.3) #palette=sns.color_palette(col), 
    g.map_dataframe(dibuja_X, x='time', y='Fuerza', lw=0.25, alpha=0.3)
      
    """
    def dibuja_xr(x,y, **kwargs):
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        print(y, ID, repe, color, kwargs.keys())
    
    g=daDatos.sel(eje='z').plot.line(x='time', col='ID', col_wrap=4, hue='repe', sharey=False)
    #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
    #g.map_dataarray_line(dibuja_xr, x='time', y=None, hue='repe')#, y='trial')
    col=['C0', 'C1', 'C2']
    for h, ax in enumerate(g.axes): #extrae cada fila
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
        

def recorta_ventana_analisis(daDatos, ventana_analisis):
    def corta_ventana(datos, ini, fin):
        #print(datos.shape, ini,fin)        
        d2 = np.full(datos.shape, np.nan) #rellena con nan al final para que tengan mismo tamaño
        d2[:int(fin)-int(ini)] = datos[int(ini):int(fin)]
        return d2 #datos[int(ini):int(fin)]
    
    daCortado = xr.apply_ufunc(corta_ventana, daDatos, ventana_analisis.isel(evento=0).sel(ID=daDatos.ID, repe=daDatos.repe), ventana_analisis.isel(evento=1).sel(ID=daDatos.ID, repe=daDatos.repe),
                   input_core_dims=[['time'], [], []],
                   output_core_dims=[['time']],
                   exclude_dims=set(('time',)),
                   vectorize=True,
                   join='outer'
                   ).assign_coords({'time':daDatos.time}).dropna(dim='time', how='all')
    daCortado.attrs = daDatos.attrs
    daCortado.name = daDatos.name
    #daCortado.plot.line(x='time', row='ID', col='eje')
    return daCortado



def calcula_variables(daDatos, daPeso, daEventos):
    import scipy.integrate as integrate
    #se puede integrar directamente con ufunc, pero no deja meter parámetro initial=0 y devuelve con un instante menos
    def integra_v(data,time,peso,ini,fin):
        dat = np.full(len(data), np.nan)        
        #try:
        ini=int(ini)
        fin=int(fin)
        dat[ini:fin] = integrate.cumtrapz(data[ini:fin]-peso, time[ini:fin], initial=0)            
        # except:
        #     dat = np.full(len(data), np.nan)
            
        return dat
    
    #data = daDatos[0,1].sel(eje='z').data
    #time = daDatos.time.data
    #peso=daPeso[0,1].data
    #ini = daEventos[0,1].sel(evento='iniMov')
    #fin = daEventos[0,1].sel(evento='finMov')
    daV = (xr.apply_ufunc(integra_v, daDatos.sel(eje='z'), daDatos.time, daPeso, daEventos.sel(evento='iniMov'), daEventos.sel(evento='finMov'),
                   input_core_dims=[['time'], ['time'], [], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   join='exact',
                   ) / (daPeso/9.8)).drop_vars('stat')
    #daV.plot.line(x='time', col='ID', col_wrap=4)
    
    
    daS = xr.apply_ufunc(integra_v, daV, daDatos.time, 0, daEventos.sel(evento='iniMov'), daEventos.sel(evento='finMov'),
                   input_core_dims=[['time'], ['time'], [], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   )
    #daS.plot.line(x='time', col='ID', col_wrap=4)
    
    daP = daDatos.sel(eje='z') * daV
    daRFD = daDatos.sel(eje='z').differentiate(coord='time')
    
    daV.attrs['units']='m/s'
    daS.attrs['units']='m'
    daP.attrs['units']='W'
    daRFD.attrs['units']='N/s'
    
    return xr.Dataset({'v':daV, 's':daS, 'P':daP, 'RFD':daRFD})




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
    
    
    
    