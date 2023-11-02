# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 16:36:37 2023

@author: Jose L. L. Elvira

Los .c3d del Bioware exportan solo los datos de los sensores por separado,
8 por plataforma.
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
import xarray as xr

import c3d
import time

__author__ = 'Jose Luis Lopez Elvira'
__version__ = '0.0.1'
__date__ = '02/10/2023'

"""
Modificaciones:
    02/10/2023, v0.0.1
            - Empezado a partir de read_vicon.c3d.
            
"""     


# =============================================================================
# %% Carga trayectorias desde c3d
# =============================================================================
def read_kistler_c3d_xr(file, section='Forces', n_vars_load=None, coincidence='similar'):
    #if section not in ['Trajectories', 'Model Outputs', 'Forces', 'EMG']:
    #MEJORAR
    if not ('Trajectories' in section or 'Model Outputs'in section or 'Forces' in section or 'EMG'in section):
        raise Exception('Section header not found, try "Trajectories", "Model outputs", "Forces" or "EMG"')
        return

    timer = time.time() #inicia el contador de tiempo
    
    #se asegura de que la extensión es c3d
    file = file.with_suffix('.c3d')

    try:
        timerSub = time.time() #inicia el contador de tiempo
        print(f'Loading file: {file.name}')
        
        with open(file, 'rb') as handle:
            reader = c3d.Reader(handle)
                
            #freq = reader.point_rate
            freq_analog = reader.analog_rate
                            
            points = []
            analog = []
            for i, (_, p, a) in enumerate(reader.read_frames()):
                #points.append(p)
                analog.append(a)
                if not i % 10000 and i:
                    print('Extracted %d point frames', len(points))

        
    
        labels_analog = [s.split('.')[0].replace(' ', '') for s in reader.analog_labels]
        data_analog = np.concatenate(analog, axis=1)

        #data_analog.shape
        coords={
                'n_var' : labels_analog,
                'time' : np.arange(data_analog.shape[1]) / freq_analog,                
                }  
        da_analog = xr.DataArray(data=data_analog,
                                dims=coords.keys(),
                                coords=coords,
                                attrs={'freq': freq_analog}
                                )
        
        
        
        
        da_analog.attrs['units'] = 'N'
        da_analog.time.attrs['units']='s'
                
        
        print('Tiempo {0:.3f} s \n'.format(time.time()-timerSub))
        
    except Exception as err:
        print('\nATENCIÓN. No se ha podido procesar '+ file.name, err, '\n')          
           
    
    if n_vars_load:
        da_analog = da_analog.sel(n_var=n_vars_load)
        
    return da_analog


def separa_plataformas(da):
    plat1 = da.sel(n_var=da.n_var.str.startswith('F1'))
    plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip('F1'))
    
    plat2 = da.sel(n_var=da.n_var.str.startswith('F2'))
    plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip('F2'))
    
    da = (xr.concat([plat1, plat2], dim='plat')
            .assign_coords(plat=['plat1', 'plat2'])
         )

    return da

def separa_ejes(da):
    #TODO: Falta quitarles la letra del eje en el nombre
    x = da.sel(n_var=da.n_var.str.contains('x'))
    y = da.sel(n_var=da.n_var.str.contains('y'))
    z = da.sel(n_var=da.n_var.str.contains('z'))
    da = (xr.concat([x,y,z], dim='axis')
                #.assign_coords(n_var='plat1')
                .assign_coords(axis=['x', 'y', 'z'])
                #.expand_dims({'n_var':1})
                )
    return da




# =============================================================================
# %% MAIN    
# =============================================================================
if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.append('F:\Programacion\Python\Mios\Functions')
    from read_kistler_c3d import read_kistler_c3d_xr
    
    ruta_archivo =  Path(r'F:\Investigacion\Proyectos\Saltos\2023PreactivacionSJ\DataCollection\S01\FeedbackFuerza')
    file = ruta_archivo / 'S01_CMJ_000.c3d'
    daForce = read_kistler_c3d_xr(file)
    daForce = separa_plataformas(daForce)
    daForce.plot.line(x='time', col='plat')

