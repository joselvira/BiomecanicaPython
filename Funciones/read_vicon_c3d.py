# -*- coding: utf-8 -*-
"""
Created on Fry Sep 15 16:36:37 2023

@author: josel
"""
from __future__ import division, print_function

import warnings #para quitar warnings de no encontrar points

import numpy as np
import pandas as pd
import xarray as xr

import c3d
import time

__author__ = 'Jose Luis Lopez Elvira'
__version__ = '0.0.2'
__date__ = '23/12/2023'

"""
Modificaciones:
    23/12/2023, v0.0.2
            - Perfeccionada carga de fuerzas.

    15/09/2023, v0.0.1
            - Empezado tomando trozos sueltos.
            
"""     


# =============================================================================
# %% Carga trayectorias desde c3d
# =============================================================================
def read_vicon_c3d_xr(file, section=None, n_vars_load=None, coincidence='similar'):
    if section not in ['Trajectories', 'Model Outputs', 'Forces', 'EMG']: #not ('Trajectories' in section or 'Model Outputs'in section or 'Forces' in section or 'EMG'in section):
        raise Exception('Section header not found, try "Trajectories", "Model outputs", "Forces" or "EMG"')
        return

    timer = time.perf_counter() #inicia el contador de tiempo
    
    #se asegura de que la extensión es c3d
    file = file.with_suffix('.c3d')

    try:
        timerSub = time.perf_counter() #inicia el contador de tiempo
        #print(f'Loading section {section}, file: {file.name}')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, 'rb') as handle:
                reader = c3d.Reader(handle)
                    
                freq = reader.point_rate
                freq_analog = reader.analog_rate
                                
                points = []
                analog = []
                for i, (_, p, a) in enumerate(reader.read_frames()):
                    points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print('Extracted %d point frames', len(points))

        #Trajectiories and Modeled outputs
        if ('Trajectories' in section or 'Model Outputs'in section):
            labels = [s.replace(' ', '') for s in reader.point_labels]
            data = np.asarray(points)[:,:,:3]

            coords={
                'time' : np.arange(data.shape[0]) / freq,
                'n_var' : labels,
                'axis' : ['x', 'y', 'z'],
                }     
            da = xr.DataArray(data, #=np.expand_dims(data, axis=0),
                                dims = coords.keys(),
                                coords = coords,
                                name = 'Trajectories',
                                attrs = {'freq': freq,
                                        'units': 'mm',}
                                ).transpose('n_var', 'axis', 'time')
            if 'Trajectories' in section:
                #Delete unnamed trajectories and modeled outputs
                da = da.sel(n_var=(~da.n_var.str.startswith('*') & ~da.n_var.str.contains('USERMO')))
            if 'Model Outputs' in section:
                da = da.sel(n_var=da.n_var.str.contains('USERMO'))
            
            
        #Analogs
        elif section in ['Forces', 'EMG']: #('Forces' in section or 'EMG' in section):
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
            
            #Forces
            #Sometimes contains 'Force' and others 'Fx', Fy', 'Fz'
            #Get only force variables
            if section=='Forces':
                if 'Force' in da_analog.n_var: #new versions of Nexus?
                    da = da_analog.sel(n_var=da_analog.n_var.str.contains('Force'))
                elif da_analog.n_var.str.contains('Fz').any(): #old versions of Nexus?
                    da = da_analog.sel(n_var=da_analog.n_var.str.startswith('F'))
                else:
                    da = xr.DataArray()
                    raise Exception('Apparently no force data in file')
                #if da_analog.n_var.str.startswith('F').any():#da_analog.n_var.str.contains('Force').any(): #'Force' in da_analog.n_var:
                try:
                    #da = da_analog.sel(n_var=da_analog.n_var.str.startswith('F')) #'Force') #.sel(n_var=da_analog.n_var.str.contains('Force'))
                    if len(da.n_var) == 3: #1 platform
                        x = da.isel(n_var=0)
                        y = da.isel(n_var=1)
                        z = da.isel(n_var=2)
                        da = (xr.concat([x,y,z], dim='axis')
                                    .assign_coords(n_var='plat1')
                                    .assign_coords(axis=['x', 'y', 'z'])
                                    .expand_dims({'n_var':1})
                                    )
                    elif len(da.n_var) == 6: #2 platforms
                        x = da.isel(n_var=[0, 3]).assign_coords(n_var=['plat1', 'plat2'])
                        y = da.isel(n_var=[1, 4]).assign_coords(n_var=['plat1', 'plat2'])
                        z = da.isel(n_var=[2, 5]).assign_coords(n_var=['plat1', 'plat2'])
                        da = (xr.concat([x,y,z], dim='axis')
                                    #.assign_coords(n_var=['plat1', 'plat2'])
                                    .assign_coords(axis=['x', 'y', 'z'])
                                    )
                    else:
                        raise Exception('The number of Force variables is not 3 or 6')
                    da.attrs['units']='N'
                    #da.time.attrs['units']='s'
                    #da.plot.line(x='time', col='axis', hue='n_var')
                except:
                    da = xr.DataArray()
                    raise Exception('Not available force data in file')

            #EMG
            elif section=='EMG':
                if da_analog.n_var.str.contains('EMG').any():
                    da = da_analog.sel(n_var=da_analog.n_var.str.contains('EMG'))
                    da.attrs['units']='mV'
                    #da.n_var.sortby('n_var')
                    #da.plot.line(x='time', col='n_var', col_wrap=3)
                else:
                    da = xr.DataArray()
                    raise Exception('No EMG data in file')
                    
            
        da.time.attrs['units']='s'
        da.name = section                
        
        #print('Tiempo {0:.3f} s \n'.format(time.perf_counter()-timerSub))
        
    except Exception as err:
        da = xr.DataArray()
        print(f'\nATENCIÓN. No se ha podido procesar {file.name}, {err}\n')          
           
    
    if n_vars_load is not None and 'n_var' in da.coords:        
        da = da.sel(n_var=n_vars_load)

        
    return da #daTrajec, daModels, daForce, daEMG


def read_vicon_c3d_xr_global(file, section=None, n_vars_load=None, coincidence='similar'):
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
        print(f'Loading section {section}, file: {file.name}')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, 'rb') as handle:
                reader = c3d.Reader(handle)
                    
                freq = reader.point_rate
                freq_analog = reader.analog_rate
                                
                points = []
                analog = []
                for i, (_, p, a) in enumerate(reader.read_frames()):
                    points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print('Extracted %d point frames', len(points))

        #Trajectiories and Modeled outputs
        if ('Trajectories' in section or 'Model Outputs'in section):
            labels = [s.replace(' ', '') for s in reader.point_labels]
            data = np.asarray(points)[:,:,:3]

            coords={
                'time' : np.arange(data.shape[0]) / freq,
                'n_var' : labels,
                'axis' : ['x', 'y', 'z'],
                }     
            da = xr.DataArray(data, #=np.expand_dims(data, axis=0),
                                dims = coords.keys(),
                                coords = coords,
                                name = 'Trajectories',
                                attrs = {'freq': freq,
                                        'units': 'mm',}
                                ).transpose('n_var', 'axis', 'time')
            if 'Trajectories' in section:
                #Delete unnamed trajectories and modeled outputs
                daTraj = da.sel(n_var=(~da.n_var.str.startswith('*') & ~da.n_var.str.contains('USERMO')))
            if 'Model Outputs' in section:
                daMod = da.sel(n_var=da.n_var.str.contains('USERMO'))
            
            
        #Analogs
        elif ('Forces' in section or 'EMG' in section):
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
            
            #Forces
            if da_analog.n_var.str.contains('Force').any(): #'Force' in da_analog.n_var:
                daForces = da_analog.sel(n_var='Force') #.sel(n_var=da_analog.n_var.str.contains('Force'))
                if len(daForces.n_var) == 3: #1 platform
                    x = daForces.isel(n_var=0)
                    y = daForces.isel(n_var=1)
                    z = daForces.isel(n_var=2)
                    daForces = (xr.concat([x,y,z], dim='axis')
                                .assign_coords(n_var='plat1')
                                .assign_coords(axis=['x', 'y', 'z'])
                                .expand_dims({'n_var':1})
                                )
                elif len(daForces.n_var) == 6: #2 platforms
                    x = daForces.isel(n_var=[0, 3])
                    y = daForces.isel(n_var=[1, 4])
                    z = daForces.isel(n_var=[2, 5])
                    daForces = (xr.concat([x,y,z], dim='axis')
                                .assign_coords(n_var=['plat1', 'plat2'])
                                .assign_coords(axis=['x', 'y', 'z'])
                                )
                    daForces.time.attrs['units']='s'
                #da.plot.line(x='time', col='axis', hue='n_var')
            else:
                daFor = xr.DataArray()

            #EMG
            if da_analog.n_var.str.contains('EMG').any():
                daEMG = da_analog.sel(n_var=da_analog.n_var.str.contains('EMG'))
                daEMG.time.attrs['units']='s'
                #daEMG.n_var.sortby('n_var')
                #daEMG.plot.line(x='time', col='n_var', col_wrap=3)
            else:
                daEMG = xr.DataArray()
        
        
        #da.time.attrs['units']='s'
                
        
        print('Tiempo {0:.3f} s \n'.format(time.time()-timerSub))
        
    except Exception as err:
        print('\nATENCIÓN. No se ha podido procesar '+ file.name, err, '\n')          
           
    
    if n_vars_load:
        da = da.sel(n_var=n_vars_load)

    daRet = []
    if 'Trajectories' in section:
        daRet.append(daTraj)
    if 'Model Outputs' in section:
        daRet.append(daMod,)
    if 'Forces' in section:
        daRet.append(daForces)
    if 'EMG' in section:
        daRet.append(daEMG)
    
    if len(daRet) == 1:
        daRet = daRet[0]
        
    return daRet #daTrajec, daModels, daForce, daEMG

def read_vicon_c3d_xr_global_ds(file, section='Trajectories', n_vars_load=None, coincidence='similar'):
        
    timer = time.time() #inicia el contador de tiempo
    
    #se asegura de que la extensión es c3d
    file = file.with_suffix('.c3d')

    try:
        timerSub = time.time() #inicia el contador de tiempo
        print('Cargando archivo: {0:s}'.format(file.name))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, 'rb') as handle:
                reader = c3d.Reader(handle)
                    
                freq = reader.point_rate
                freq_analog = reader.analog_rate
                                
                points = []
                analog = []
                for i, (_, p, a) in enumerate(reader.read_frames()):
                    points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print('Extracted %d point frames', len(points))
                    
                labels = [s.replace(' ', '') for s in reader.point_labels]
                labels_analog = [s.split('.')[0].replace(' ', '') for s in reader.analog_labels]
        data = np.asarray(points)[:,:,:3]
        data_analog = np.concatenate(analog, axis=1)
        
        #Trajectiories and Modeled outputs
        coords={
            'time' : np.arange(data.shape[0]) / freq,
            'n_var' : labels,
            'axis' : ['x', 'y', 'z'],
            }     
        da = xr.DataArray(data, #=np.expand_dims(data, axis=0),
                            dims = coords.keys(),
                            coords = coords,
                            name = 'Trajectories',
                            attrs = {'freq': freq,
                                    'units': 'mm',}
                            ).transpose('n_var', 'axis', 'time')
        da.time.attrs['units']='s'
        
        # if section=='Trajectories':
        #     #Delete unnamed trajectories and modeled outputs
        #     da = da.sel(n_var=(~da.n_var.str.startswith('*') & ~da.n_var.str.contains('USERMO')))
        # elif section=='Model Outputs':
        #     da = da.sel(n_var=da.n_var.str.contains('USERMO'))
        
        daTrajec = da.sel(n_var=(~da.n_var.str.startswith('*') & ~da.n_var.str.contains('USERMO')))

        daModels = da.sel(n_var=da.n_var.str.contains('USERMO'))
        #da.isel(axis=0).plot.line(x='time', hue='n_var')

        #Analogs
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
        
        #Forces
        if da_analog.n_var.str.contains('Force').any(): #'Force' in da_analog.n_var:
            daForce = da_analog.sel(n_var='Force') #.sel(n_var=da_analog.n_var.str.contains('Force'))
            if len(daForce.n_var) == 3: #1 platform
                x = daForce.isel(n_var=0)
                y = daForce.isel(n_var=1)
                z = daForce.isel(n_var=2)
                daForce = (xr.concat([x,y,z], dim='axis')
                           .assign_coords(n_var='plat1')
                           .assign_coords(axis=['x', 'y', 'z'])
                           .expand_dims({'n_var':1})
                           )
            elif len(daForce.n_var) == 6: #2 platforms
                x = daForce.isel(n_var=[0, 3])
                y = daForce.isel(n_var=[1, 4])
                z = daForce.isel(n_var=[2, 5])
                daForce = (xr.concat([x,y,z], dim='axis')
                           .assign_coords(n_var=['plat1', 'plat2'])
                           .assign_coords(axis=['x', 'y', 'z'])
                           )
            #daForce.plot.line(x='time', col='axis', hue='n_var')
        else:
            daForce = xr.DataArray()

        #EMG
        if da_analog.n_var.str.contains('EMG').any():
            daEMG = da_analog.sel(n_var=da_analog.n_var.str.contains('EMG'))
            daEMG.n_var.sortby('n_var')
            #daEMG.plot.line(x='time', col='n_var', col_wrap=3)
        else:
            daEMG= xr.DataArray()
        
        
        print('Tiempo {0:.3f} s \n'.format(time.time()-timerSub))
        
    except Exception as err:
        print('\nATENCIÓN. No se ha podido procesar '+ file.name, err, '\n')          
           
    
    if n_vars_load:
        da = da.sel(n_var=n_vars_load)
    
    daTodo= xr.Dataset({'Trajectories':daTrajec,
                         'Modeled':daModels,
                         'Forces':daForce, 
                         'EMG':daEMG}
                      )
        
    return daTodo #daTrajec, daModels, daForce, daEMG



# =============================================================================
# %% MAIN    
# =============================================================================
if __name__ == '__main__':
    from pathlib import Path
    
    ruta_archivo =  Path(r'F:\Programacion\Python\Mios\ViconNexus\C3D\ArchivosC3D')
    file = ruta_archivo / 'SaltosS13_SJ_100S_03.c3d'
    daTrajec = read_vicon_c3d_xr(file, section='Trajectories')
    daTrajec.isel(n_var=slice(6)).plot.line(x='time', col='n_var', col_wrap=3)

    daTrajec = read_vicon_c3d_xr(file, section='Trajectories', n_vars_load=['LPSI', 'LASIMID', 'RASIMID', 'RPSI'])
    daTrajec.plot.line(x='time', col='n_var')

    daModels = read_vicon_c3d_xr(file, section='Model Outputs')
    #Mezcla variables modeladas de ángulo con EMG (3 canales por variable)
    daModels.isel(n_var=slice(None)).plot.line(x='time', col='n_var', col_wrap=3, sharey=False)
    #modelos sacados directamente de exportar a csv
    nom_vars =',,S13:AngArtAnkle_R,,,S13:AngArtHip_R,,,S13:AngArtKnee_R,,,S13:AngSegMUSLO_R,,,S13:AngSegPELVIS_LR,,,S13:AngSegPIERNA_R,,,S13:AngSegPIE_R,,,S13:EMG1,S13:EMG2,S13:EMG3,S13:EMG4,S13:EMG5,S13:EMG6,S13:Forces,,,S13:LASI,,,S13:LHJC,,,S13:RAJC,,,S13:RASI,,,S13:RHJC,,,S13:RKJC,,,S13:Right_AnkleExt,,,S13:Right_AnkleInt,,,S13:Right_KneeExt,,,S13:Right_KneeInt,,,S13:LASI,,,S13:LHJC,,,S13:RAJC,,,S13:RASI,,,S13:RHJC,,,S13:RKJC,,,S13:Right_AnkleExt,,,S13:Right_AnkleInt,,,S13:Right_KneeExt,,,S13:Right_KneeInt,,,'.split(',')[2::3]
    nom_vars = [s.split(':')[-1] for s in nom_vars]
    #Hay que hacer el ajuste de los nombres de las variables a mano para que coincidan
    replace_vars = dict(zip(daModels.n_var.data[:7], nom_vars[:7]))
    names = pd.DataFrame(daModels.n_var.data).replace(replace_vars)[0].tolist()
    daModels = daModels.assign_coords(n_var=names)
    
    daModel_forces = daModels.isel(n_var=-1)
    daModel_forces.plot.line(x='time')
    daModel_EMG = (daModels.sel(n_var=daModels.n_var.str.contains('USER'))
                    .isel(n_var=slice(None,-1))
                    )
    daModel_EMG.plot.line(x='time', col='n_var', col_wrap=3)


    daForce = read_vicon_c3d_xr(file, section='Forces')
    daForce.plot.line(x='time', col='n_var', col_wrap=3)

    
    daEMG = read_vicon_c3d_xr(file, section='EMG', n_vars_load=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6'])
    daEMG.plot.line(x='time', col='n_var', col_wrap=3)




    [daTrajec, daModels] = read_vicon_c3d_xr_global(file, section=['Trajectories','Model Outputs'])
    daTrajec.isel(n_var=slice(6)).plot.line(x='time', col='n_var', col_wrap=3)
    daModels.isel(n_var=slice(6)).plot.line(x='time', col='n_var', col_wrap=3)


    n_vars_load ={'Trajectories':['LPSI', 'LASIMID', 'RASIMID', 'RPSI'],
                  'Model Outputs':['USERMO', 'USERM1', 'USERM2', ]}
    [daTrajec, daModels] = read_vicon_c3d_xr_global(file, section=['Trajectories','Model Outputs'])
    daTrajec.isel(n_var=slice(6)).plot.line(x='time', col='n_var', col_wrap=3)
    daModels.isel(n_var=slice(6)).plot.line(x='time', col='n_var', col_wrap=3)


    ruta_archivo =  Path(r'F:\Programacion\Python\Mios\ViconNexus\C3D\ArchivosC3D')
    file = ruta_archivo / 'SillaRuedas-s1905.c3d'
    daForce = read_vicon_c3d_xr(file, section='Forces')
    daForce.plot.line(x='time', col='n_var', col_wrap=3)

    daEMG = read_vicon_c3d_xr(file, section='EMG', n_vars_load=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6'])
    

    ruta_archivo =  Path(r'F:\Programacion\Python\Mios\ViconNexus\C3D\ArchivosC3D')
    file = ruta_archivo / 'Pablo_FIN.c3d'
    daModels = read_vicon_c3d_xr(file, section='Model Outputs')
    daModels.isel(n_var=slice(6)).plot.line(x='time', col='n_var', col_wrap=3)
    daForce = read_vicon_c3d_xr(file, section='Forces')
    daForce.plot.line(x='time', col='n_var', col_wrap=3)

    daEMG = read_vicon_c3d_xr(file, section='EMG', n_vars_load=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6'])
    

