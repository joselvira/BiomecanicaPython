#%% -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:01:08 2021

@author: josel
"""

from typing import Optional, Union, Any

import numpy as np
import pandas as pd
import xarray as xr 
import itertools

import matplotlib.pyplot as plt


__author__ = 'Jose Luis Lopez Elvira'
__version__ = 'v.3.1.2'
__date__ = '10/11/2023'


"""
Modificaciones:
    10/11/2023, v3.1.2
    - Ahora mantiene las unidades en el dataframe cortado.
    
    11/03/2023, v3.1.1
    - Solucionado error con función find_peaks_aux. Hace copia antes de buscar
      cortes.
    - Incluido parámetro show en función detect_onset_detecta_aux.
    
    13/02/2023, v3.1.0
    - Las funciones find_peaks_aux y detect_onset_detecta_aux admiten un
      argumento para buscar cortes a partir de la media + x veces SD.
    
    09/02/2023, v3.0.0
    - Metido todo dentro de la clase SliceTimeSeriesPhases.
    - Cambiada nomenclatura num_repes, max_repes, descarta_rep_ini y
      descarta_rep_fin a num_cuts, max_cuts, discard_cuts_end y
      discard_cuts_end.
    
    28/01/2023, v2.1.0
        - Función común corta_repes que distribuye según los datos sean Pandas
        o xarray.
        - Función en xarray que devuelve sólo los nº de índice de los events.
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

class SliceTimeSeriesPhases():
    def __init__(self, data: Optional[xr.DataArray]=xr.DataArray(),
                 freq: Optional[float]=None,
                 n_dim_time: Optional[str]='time',
                 reference_var: Optional[Union[str, dict]]=None,
                 discard_phases_ini: int=0,
                 n_phases: Optional[int]=None,
                 discard_phases_end: int=0,
                 include_first_next_last: Optional[bool]=False,
                 max_phases: int = 100,
                 func_events: Optional[Any]=None,
                 **kwargs_func_events):
        self.data = data
        self.events = None
        self.n_dim_time = n_dim_time
        self.reference_var = reference_var
        self.discard_phases_ini = discard_phases_ini
        self.n_phases = n_phases
        self.discard_phases_end = discard_phases_end
        self.include_first_next_last = include_first_next_last
        self.func_events = func_events
        self.max_phases = max_phases
        self.kwargs_func_events = kwargs_func_events
        
        if freq==None and not data.isnull().all():
            self.freq = (np.round(1/(self.data[self.n_dim_time][1] - self.data[self.n_dim_time][0]),1)).data
        else:
            self.freq = freq
        
    
    def detect_events(self) -> xr.DataArray:
        #TODO: AJUSTAR LA FUNCIÓN PARA QUE ADMITA UMBRALES ESPECÍFICOS DE CADA ENSAYO
        def detect_aux_idx(data, data_reference_var=None, func_events=None, max_phases=50, discard_phases_ini=0, n_phases=None, discard_phases_end=0, **kwargs_func_events):           
            events = np.full(max_phases, np.nan)
            if np.count_nonzero(~np.isnan(data))==0 or np.count_nonzero(~np.isnan(data_reference_var))==0:
                return events
            try:
                evts = func_events(data_reference_var, **kwargs_func_events)
            except:
                return events
            
            #If necessary, adjust initial an final events
            evts = evts[discard_phases_ini:]
            
            if n_phases==None:
                evts = evts[:len(evts)-discard_phases_end]
            else: #if a specific number of phases from the first event is required
                if len(evts) >= n_phases:
                    evts = evts[:n_phases+1]
                else: #not enought number of events in the block, trunkated to the end
                    pass
            events[:len(evts)] = evts
            return events
        
        if self.func_events==None:
            raise Exception('A function to detect the events must be specified')
        
        da = xr.apply_ufunc(detect_aux_idx,
                          self.data,
                          self.data.sel(self.reference_var),
                          self.func_events,
                          self.max_phases,
                          self.discard_phases_ini,
                          self.n_phases,
                          self.discard_phases_end,
                          input_core_dims=[[self.n_dim_time], [self.n_dim_time], [], [], [], [], []],  #lista con una entrada por cada argumento
                          output_core_dims=[['n_event']],
                          exclude_dims=set(('n_event', self.n_dim_time)), 
                          dataset_fill_value=np.nan,
                          vectorize=True,
                          dask='parallelized',
                          #keep_attrs=True,
                          kwargs=self.kwargs_func_events,
                          )
        da = (da.assign_coords(n_event=range(len(da.n_event)))
              .dropna(dim='n_event', how='all').dropna(dim='n_event', how='all')
              )
        self.events = da        
        return da


    def slice_time_series(self, events: Optional[xr.DataArray]=None) -> xr.DataArray:
        if events is not None: #the events are passed manually
            self.events = events
        elif self.events is None: #if the events are not detected yet, detect them
            self.detect_events()    
                
        def slice_aux(data, events, max_phases=50, include_first_next_last=True):
            if np.count_nonzero(~np.isnan(data))==0 or np.count_nonzero(~np.isnan(events))==0:
                return np.full((max_phases, len(data)),np.nan)
            
            events = events[~np.isnan(events)].astype(int)
            phases = np.full((max_phases, len(data)),np.nan)            
            t = np.split(data, events)[1:-1]
            try:
                t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
                phases[:t.shape[0], : t.shape[1]] = t
            except:
                pass
                
            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            #TODO: improve vectorizing
            if include_first_next_last:
                for sl in range(len(events)-1):
                    phases[sl, events[sl+1]-events[sl]] = data[events[sl+1]]
            return phases
  
        da = xr.apply_ufunc(slice_aux,
                          self.data,
                          self.events,
                          self.max_phases,                      
                          self.include_first_next_last,
                          input_core_dims=[[self.n_dim_time], ['n_event'], [], []],
                          output_core_dims=[['n_event', self.n_dim_time]],
                          exclude_dims=set(('n_event', self.n_dim_time )),
                          dataset_fill_value=np.nan,
                          vectorize=True,
                          dask='parallelized',
                          keep_attrs=True,
                          #kwargs=args_func_events,
                          )
        da = (da.assign_coords(n_event=range(len(da.n_event)))
              .assign_coords(time=np.arange(0, len(da.time)) / self.freq)
              .dropna(dim='n_event', how='all').dropna(dim=self.n_dim_time, how='all')
              .rename({'n_event':'phase'})
              )
        da.attrs= self.data.attrs
        da.time.attrs['units'] = self.data.time.attrs['units']
        
        return da
    
    
    # =============================================================================
    # Custom function to adapt from Detecta detect_onset
    # =============================================================================
    def detect_onset_detecta_aux(data, event_ini=0, xSD=None, show=False, **args_func_events):
        #Si se pasa como argumento corte_ini=1, coge el corte del final de cada ventana
        try:
            from detecta import detect_onset
        except ImportError:		
            raise Exception('This function needs Detecta to be installed (https://pypi.org/project/detecta/)')
        
        # try: #detect_onset returns 2 indexes. If not specified, select the first
        #     event_ini=args_func_events['event_ini']
        #     args_func_events.pop('event_ini', None)
        # except:
        #     event_ini=0
        if xSD is not None: #the threshold is defined by the mean + x times the standard deviation
            if 'threshold' in args_func_events:
                args_func_events.pop('threshold', None)
            args_func_events['threshold'] = np.mean(data, where=~np.isnan(data)) + np.std(data, where=~np.isnan(data))*xSD
            #print(args_func_events, np.mean(data, where=~np.isnan(data)), np.std(data, where=~np.isnan(data)), xSD)
        
        events = detect_onset(data, **args_func_events)
        
        if event_ini==1:
            events = events[:, event_ini] + 1 #if the end of the window is chosen, 1 is added to start when the threshold has already been exceeded
            events = events[:-1] #removes the last one because it is usually incomplete
        else:
            events = events[:, event_ini] #keeps the first or second value of each data pair
            events = events[1:] #removes the last one because it is usually incomplete
        
        if show:
            SliceTimeSeriesPhases.show_events(data, events, threshold=args_func_events['threshold'])
            
        return events
    
    
    # =============================================================================
    # Custom function to adapt from scipy.signal find_peaks
    # =============================================================================    
    def find_peaks_aux(data, xSD=None, show=False, **args_func_events):
        try:
            from scipy.signal import find_peaks
        except ImportError:		
            raise Exception('This function needs scipy.signal to be installed')
        if xSD is not None: #the threshold is defined by the mean + x times the standar deviation
            if isinstance(xSD, list):
                args_func_events['height'] = [np.mean(data[~np.isnan(data)]) + xSD[0] * np.std(data[~np.isnan(data)]), 
                                              np.mean(data[~np.isnan(data)]) + xSD[1] * np.std(data[~np.isnan(data)])]
            else:
                args_func_events['height'] = np.mean(data[~np.isnan(data)]) + xSD * np.std(data[~np.isnan(data)])#, where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))
        
        data = data.copy()
        
        #Deal with nans
        data[np.isnan(data)] = -np.inf
        
        events, _ = find_peaks(data, **args_func_events)
        
        if show:
            SliceTimeSeriesPhases.show_events(data, events, threshold=args_func_events['height'])
            
        return events #keeps the first value of each data pair


    def show_events(data, events, threshold=None):
        plt.plot(data, c='b')
        plt.plot(events, data[events], 'ro')
        if threshold is not None:
            plt.hlines(y=threshold,xmin=0, xmax=len(data), color = "C1", ls='--', lw=1)
        plt.show()
    
# =============================================================================


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == '__main__':
    # =============================================================================
    # ---- Create a sample
    # =============================================================================

           
    import numpy as np
    #import pandas as pd
    import xarray as xr
    from scipy.signal import butter, filtfilt
    from pathlib import Path
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    

    def create_time_series_xr(rnd_seed=None, num_subj=10, Fs=100.0, IDini=0, rango_offset = [-2.0, -0.5], rango_amp = [1.0, 2.2], rango_frec = [1.8, 2.4], rango_af=[0.0, 1.0], rango_duracion=[5.0, 5.1], amplific_ruido=[0.4, 0.7], fc_ruido=[7.0, 12.0]):
        if rnd_seed is not None:
            np.random.seed(rnd_seed) # para mantener la consistencia al crear los datos aleatorios
        subjects = []
        for subj in range(num_subj):
            #print(subj)
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
            subjects.append(senal + ruido)
            #subjects.append(np.expand_dims(senal + ruido, axis=0))
            #sujeto.append(pd.DataFrame(senal + ruido, columns=['value']).assign(**{'ID':'{0:02d}'.format(subj+IDini), 'time':np.arange(0, len(senal)/Fs, 1/Fs)}))
        
        #Pad data to last the same
        import itertools
        data = np.array(list(itertools.zip_longest(*subjects, fillvalue=np.nan)))

        data = xr.DataArray(data=data,
                    coords={'time':np.arange(data.shape[0])/Fs,
                            'ID':[f'{i:0>2}' for i in range(num_subj)], #rellena ceros a la izq. f'{i:0>2}' vale para int y str, f'{i:02}' vale solo para int                            
                            }
                    )
        return data

    rnd_seed = np.random.seed(12340) #fija la aleatoriedad para asegurarse la reproducibilidad
    n=10
    duracion=15
    freq=200.0
    Pre_a = (create_time_series_xr(rnd_seed=rnd_seed, num_subj=n, Fs=freq, IDini=0, rango_offset = [25, 29], rango_amp = [40, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion])
                .expand_dims({'n_var':['a'], 'momento':['pre']})
                .transpose('ID', 'momento', 'n_var', 'time')              
                )
    Post_a = (create_time_series_xr(rnd_seed=rnd_seed, num_subj=n, Fs=freq, IDini=0, rango_offset = [22, 26], rango_amp = [36, 40], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion])
                .expand_dims({'n_var':['a'], 'momento':['post']}) 
                .transpose('ID', 'momento', 'n_var', 'time')
                )
    var_a= xr.concat([Pre_a, Post_a], dim='momento')

    Pre_b = (create_time_series_xr(rnd_seed=rnd_seed, num_subj=n, Fs=freq, IDini=0, rango_offset = [35, 39], rango_amp = [50, 55], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion])
                .expand_dims({'n_var':['b'], 'momento':['pre']})
                .transpose('ID', 'momento', 'n_var', 'time')
                )
    Post_b = (create_time_series_xr(rnd_seed=rnd_seed, num_subj=n, Fs=freq, IDini=0, rango_offset = [32, 36], rango_amp = [32, 45], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion])
                .expand_dims({'n_var':['b'], 'momento':['post']})
                .transpose('ID', 'momento', 'n_var', 'time')
                )
    var_b= xr.concat([Pre_b, Post_b], dim='momento')

    Pre_c = (create_time_series_xr(rnd_seed=rnd_seed, num_subj=n, Fs=freq, IDini=0, rango_offset = [35, 39], rango_amp = [10, 15], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion])
                .expand_dims({'n_var':['c'], 'momento':['pre']})
                .transpose('ID', 'momento', 'n_var', 'time')
            )
    Post_c = (create_time_series_xr(rnd_seed=rnd_seed, num_subj=n, Fs=freq, IDini=0, rango_offset = [32, 36], rango_amp = [12, 16], rango_frec = [1.48, 1.52], rango_af=[0, 30], amplific_ruido=[0.4, 0.7], fc_ruido=[3.0, 3.5], rango_duracion=[duracion, duracion])
                .expand_dims({'n_var':['c'],'momento':['post']})
                .transpose('ID', 'momento', 'n_var', 'time')
                )
    var_c= xr.concat([Pre_c, Post_c], dim='momento')    
        
    #concatena todos los sujetos
    daTodos = xr.concat([var_a, var_b, var_c], dim='n_var')
    daTodos.name = 'Angle'
    daTodos.attrs['freq'] = 1/(daTodos.time[1].values - daTodos.time[0].values)#incluimos la frecuencia como atributo
    daTodos.attrs['units'] = 'deg'
    daTodos.time.attrs['units'] = 's'

    #Gráficas
    daTodos.plot.line(x='time', col='momento', hue='ID', row='n_var')


    # =============================================================================
    # %% Test the functions 
    # =============================================================================
   
    """
    #Example importing
    sys.path.insert(1, r'F:\Programacion\Python\Mios\Functions')  # add to pythonpath
    from slice_time_series_phases import SliceTimeSeriesPhases as stsp
    """
    from detecta import detect_peaks
    #Pasando por la clase
    da = SliceTimeSeriesPhases(data=daTodos, func_events=detect_peaks)
    da.detect_events() #devuelve los índices de los cuts
    da.events
    dacuts = da.slice_time_series() #corta con los índices buscados anteriormente
    dacuts.sel(n_var='a').plot.line(x='time', col='momento', hue='phase', row='ID')
    
   
    #Corta directamente
    dacuts = (SliceTimeSeriesPhases(data=daTodos, func_events=detect_peaks, max_phases=100)
                .slice_time_series()
                )
    dacuts.sel(n_var='a').plot.line(x='time', col='momento', hue='phase', row='ID')
    
    
    #Especificando una de las variables para hacer todos los cortes
    dacuts = (SliceTimeSeriesPhases(data=daTodos, func_events=detect_peaks, reference_var=dict(n_var='b'))
                .slice_time_series()
                )
    dacuts.stack(var_momento=('n_var', 'momento')).plot.line(x='time', col='var_momento', hue='phase', row='ID')
    


    #Cortar aportando cuts ya buscados o ajustados previamente
    cortes_idx = SliceTimeSeriesPhases(data=daTodos, func_events=detect_peaks, reference_var=dict(n_var='a'), max_phases=100).detect_events()
    cortes_retocados = cortes_idx.isel(n_event=slice(3,20,2))

    dacor = SliceTimeSeriesPhases(data=daTodos).slice_time_series(cortes_retocados)
    dacor.isel(ID=slice(None,6)).sel(n_var='a').plot.line(x='time', col='momento', hue='phase', row='ID')


    cortes_idx = SliceTimeSeriesPhases(data=daTodos, func_events=detect_peaks, reference_var=dict(n_var='a'), max_phases=100).detect_events()
    cortes_retocados = cortes_idx.isel(n_event=slice(5,20))

    dacor = SliceTimeSeriesPhases(data=daTodos).slice_time_series(events=cortes_retocados)
    dacor.isel(ID=slice(None,6)).sel(n_var='c').plot.line(x='time', col='momento', hue='phase', row='ID')


    daCortado = (SliceTimeSeriesPhases(daTodos, func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux, 
                                 reference_var=dict(momento='pre', n_var='b'),
                                 discard_phases_ini=0, n_phases=None, discard_phases_end=0,
                                 include_first_next_last=True, **dict(threshold=60)
                                 )
                 .slice_time_series()
                 )
    daCortado.sel(n_var='b').plot.line(x='time', col='momento', hue='phase', row='ID')
    
      
    daCortado = (SliceTimeSeriesPhases(data=daTodos, func_events=SliceTimeSeriesPhases.find_peaks_aux, 
                             reference_var=dict(momento='pre', n_var='b'),
                             discard_phases_ini=0, n_phases=None, discard_phases_end=0,
                             include_first_next_last=True, **dict(height=60, distance=10)
                             )
                     .slice_time_series()
                     )
    
    daCortado = (SliceTimeSeriesPhases(data=daTodos, func_events=SliceTimeSeriesPhases.find_peaks_aux, 
                             reference_var=dict(momento='pre', n_var='b'),
                             discard_phases_ini=0, n_phases=None, discard_phases_end=0,
                             include_first_next_last=True, **dict(height=140, distance=1))
                 .slice_time_series()
                 )
    daCortado.sel(n_var='b').plot.line(x='time', col='momento', hue='phase', row='ID')
    
    
    daCortado = (SliceTimeSeriesPhases(data=daTodos, func_events=detect_peaks, 
                                       reference_var=dict(momento='pre', n_var='b'), max_phases=100,
                                       **dict(mph=140))
                .slice_time_series()
                )
    daCortado.sel(n_var='b').plot.line(x='time', col='momento', hue='phase', row='ID')
    
    
    #find_peaks with xSD
    daCortado = (SliceTimeSeriesPhases(data=daTodos, func_events=SliceTimeSeriesPhases.find_peaks_aux, 
                             reference_var=dict(momento='pre', n_var='b'),
                             discard_phases_ini=0, n_phases=None, discard_phases_end=0,
                             include_first_next_last=True, **dict(xSD=1.4, distance=1))
                 .slice_time_series()
                 )
    daCortado.sel(n_var='b').plot.line(x='time', col='momento', hue='phase', row='ID')
    
    
    #onset by xSD
    daCortado = (SliceTimeSeriesPhases(daTodos, func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux, 
                                 #reference_var=dict(momento='pre', n_var='b'),
                                 discard_phases_ini=0, n_phases=None, discard_phases_end=0,
                                 include_first_next_last=True, **dict(xSD=-1.2)
                                 )
                 .slice_time_series()
                 )
    daCortado.sel(n_var='b').plot.line(x='time', col='momento', hue='phase', row='ID')
    
        
    
    #Trim data, slice ini and end events
    daEvents = (SliceTimeSeriesPhases(data=daTodos, func_events=SliceTimeSeriesPhases.find_peaks_aux, 
                             reference_var=dict(momento='pre', n_var='b'),
                             discard_phases_ini=0, n_phases=None, discard_phases_end=0,
                             include_first_next_last=True, **dict(height=0, distance=10))
                 .detect_events()
                 )
    
    ventana = xr.concat([daEvents.min('n_event'), daEvents.max('n_event')], dim='n_event')
    #ventana = daEvents.isel(n_event=[5,7]) #xr.concat([daEvents.isel(n_event=5), daEvents.isel(n_event=7)], dim='n_event')
     
    daTrimed = SliceTimeSeriesPhases(daTodos).slice_time_series(events=ventana)
    daTrimed.sel(n_var='b').plot.line(x='time', col='momento', hue='phase', row='ID')
    
   
    
    