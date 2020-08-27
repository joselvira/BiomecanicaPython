# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 17:02:27 2016
Última revisión: 16/06/2019
    
@author: YoMismo
"""

from __future__ import division, print_function #division #Ensure division returns float

# =============================================================================
# TAMAÑO EFECTO
# =============================================================================
import numpy as np
from scipy import stats
import pandas as pd

# =============================================================================
# #para quitar el cero inicial
# =============================================================================
def _remove_leading_zero(value, string):
    if 1 > value > -1:
        string = string.replace('0', '', 1)
    return string

class MyFloat(float):
    def __format__(self, format_string):
        if format_string.endswith('z'):  # 'fz' is format sting for floats without leading the zero
            format_string = format_string[:-1]
            remove_leading_zero = True
        else:
            remove_leading_zero = False

        string = super(MyFloat, self).__format__(format_string)
        return _remove_leading_zero(self, string) if remove_leading_zero else string
        # `_remove_leading_zero` function is same as in the first example
        #Ejemplos
        #print('some text {:.3f} some more text'.format(MyFloat(.4444)))
        #print('some text {:.3fz} some more text'.format(MyFloat(.4444)))
# =============================================================================

#Comprobada con el archivo Excel F:\Biomec\HerramientasCalculo\LimitesConfianza.xlsx
def Hedges_g(group1, group2, grouplabels=['Grupo1', 'Grupo2'], varlabels=[], muestras_dep=False, pctcl = 95, sdpooled= True, tipose='Nakagawa', leadingZero=False, decVariable=2, decES=3, decP=3, borders=False, grid=False, numbers=False, show=False, ax=None):
    """Calcula la g de Hedges.
    
    Parameters
    ----------
    group1 : 1D array_like data or 2D pandas DataFrame. Este es el de referencia, si es mayor, la diferencia es positiva.
    group2 : 1D array_like data or 2D pandas DataFrame.
    grouplabels : etiquetas de los dos grupos
    varlabels : etiquetas de las variables o columnas. Si no se ponen las etiquetas son numeros correlativos
    muestras_dep : si las muestras son dependientes (medidas repetidas) = True.
                   si las muestras son independientes = False (por defecto).
    pctcl : porcentaje para los límites de confianza (por defecto 95%).
    sdpooled : True calcula g con la desviación típica agrupada; con False utiliza la del primer grupo. Con comparación para muestras independientes mejor usar la pooled (por defecto).
    tipose : tipo de Standard Error para calcular los CL: 'Nakagawa', según Nakagawa y Cuthill (2007) (por defecto)
                                                          'Otro' según curso metaanalisis (sin referencia).
    leadingZero: False quita el cero inicial en p y ES
    decVariable : número de decimales para la variable. Por defecto 2.
    decP : número de decimales para la P. Por defecto 3.
    decES : número de decimales para el ES. Por defecto 3.
    borders: True muestra los bordes del grafico y los ticks. Por defecto False.
    grid: True muestra la línea horizontal de cada variable (útil cualdo hay muchas variables). Por defecto False.
    numbers: True incluye como etiqueta encima de cada barra su valor g y limites. Por defecto False.
    show: True muestra la grafica o no. Por defecto False.
    ax : axis para incluir el gráficos en una figura hecha de antemano    
    
    Returns
    -------    
    DataFrame con los datos formateados para imprimir en Excel.
    tabla con los datos x+-SD, p, g, g_inf y g_sup.
    g : tamaño del efecto (Hedge's g).
    g_inf : X% límite inferior.
    g_sup : X% límite superior.
    p : probabilidad del factor p del t test
    
    Example
    -------
    Hedges_g(dfx, dfy)
    ESCI, tabla=Hedges_g(dfx, dfy, grouplabels=['G1', 'G2'], varlabels=dfx.columns, muestras_dep=0, pctcl = 95, sdpooled= True, tipose='Nakagawa', decVariable=3, decP=3, decES=3, numbers=True, show=True, ax=ax)
    """
          
    if len(group1.shape) == 1:
        numCols=1
    else:
        numCols=group1.shape[1]
    ESCL = np.full((numCols, 3), np.nan)
    
    #Para tabla pandas
    var1=[]
    var2=[]
    p=[]
    es = []#np.full(len(ESCL[:,0]), np.nan)
    dfTablaResumen=pd.DataFrame({'':varlabels, })
    
    for col in range(numCols):
#        x=np.array(var1[var1.iloc[:,col].name])
#        y=np.array(var2[var2.iloc[:,col].name])

        """
        Probar sustituir las siguientes lineas con esto:
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)
        """
        if len(group1.shape) == 1:
            x=group1
            y=group2
        else:
            x=group1[group1.iloc[:,col].name]
            y=group2[group2.iloc[:,col].name]
        
        #determina la n de cada variable, no vale len() por si hay valores NaN
        nx= int(x.notnull().sum())#pd.notnull(x).count()
        ny= int(y.notnull().sum())#pd.notnull(y).count()
        pct = (100-float(100-pctcl)/2)/100
        
        if muestras_dep == 1: #muestras dependientes
            if sdpooled == True:
                S=np.sqrt((((nx-1) * x.std()**2) + ((ny-1) * y.std()**2)) / (nx+ny-2.0))
            else:
                S= x.std() #np.std(x, ddof=1)
            c_n_1 = 1-(3/float(4*nx-5)) #factor de corrección
            g = c_n_1*((y.mean() - x.mean()) / S)
                        
            t_crit = stats.t.ppf(q=pct, df=nx-1)
            if tipose=='Nakagawa':
                SE = np.sqrt((2*(1-stats.pearsonr(x,y)[0])/nx) + g**2/(2*(nx-1)))
            else:
                SE = np.sqrt(((nx-1) / float(nx*(nx-3))) * (1+nx*g**2)-(g**2/c_n_1**2))
            g_inf = g - t_crit * SE
            g_sup = g + t_crit * SE
            
#            ttestrel=[x,y]
#            dfTtestrel=pd.concat(ttestrel, axis=1).dropna()
#            t, p_val = stats.ttest_rel(dfTtestrel.iloc[:,0], dfTtestrel.iloc[:,1], axis=0, nan_policy='omit')
            t, p_val = stats.ttest_rel(x, y, axis=0, nan_policy='omit')
        else: #muestras independientes
            if sdpooled == True:
                S=np.sqrt((((nx-1) * x.std()**2) + ((ny-1) * y.std()**2)) / float(nx+ny-2))
            else:
                S= x.std()  #np.std(x, ddof=1)
            c_m = 1-(3/float(4*(nx + ny)-9)) #factor de corrección
            g = c_m*((x.mean() - y.mean()) / S)
            
            t_crit=stats.t.ppf(q=pct, df=nx+ny-2)
            #t_crit=stats.t.interval(alpha=0.95, df=100-1, loc=0, scale=1) * np.sqrt(1+1/float(len(x)+len(y)))
#            stats.t.interval(alpha=0.95, df=100-1, loc=0, scale=1) * np.sqrt(1+1/100)
#            stats.t.ppf(q=0.975, df=100-1)            
            if tipose=='Nakagawa':
                SE = np.sqrt(((nx+ny) / float(nx*ny)) + (g**2/float(2*(nx+ny-2))))
            else:
                SE = np.sqrt(((nx+ny) / float(nx*ny)) + (g**2/float(2*(nx+ny))))
            #intervalo = t_crit*(np.sqrt(((len(x)+len(y)) / float(len(x)*len(y))) + (((np.mean(x)-np.mean(y)) / float(np.sqrt((((len(x)-1)*np.std(x, ddof=1)**2) + ((len(y)-1)*np.std(y, ddof=1)**2)) / float(len(x)+len(y)-2))))**2 / float(2*(len(x)+len(y))))))            
            g_inf = g - t_crit * SE
            g_sup = g + t_crit * SE
            
            ########
            #comprueba igualdad varianzas
            w, p_lev=stats.levene(x.dropna(), y.dropna()) #CUANDO HAY ALGÚN NAN SALE NAN REVISAR
            igualVarianzas=True #por defecto
            if p_lev<0.05:
                igualVarianzas=False
            ########
            t, p_val = stats.ttest_ind(x, y, equal_var=igualVarianzas, nan_policy='omit')#CUANDO HAY ALGÚN NAN SALE NAN REVISAR
                    
            
        ESCL[col] = [g, g_inf, g_sup]
        var1.append(('{0:.{dec}f} {1} {2:.{dec}f}').format(x.mean(), r'±', x.std(), dec=decVariable))#±
        var2.append('{0:.{dec}f} {1} {2:.{dec}f}'.format(y.mean(), r'±', y.std(), dec=decVariable))
        
        if(leadingZero==False):
            p.append('{:.{dec}fz}'.format(MyFloat(p_val), dec=decP))
            es.append('{:.{dec}fz} [{:.{dec}fz}, {:.{dec}fz}]'.format(MyFloat(ESCL[col,0]), MyFloat(ESCL[col,1]), MyFloat(ESCL[col,2]), dec=decES))
        else:
            p.append('{:.{dec}f}'.format(p_val, dec=decP))
            es.append('{:.{dec}f} [{:.{dec}f}, {:.{dec}f}]'.format(ESCL[col,0], ESCL[col,1], ESCL[col,2], dec=decES))
    ###########################################
    #Crea la tabla con los resultados en Pandas
    ###########################################
    
    """
    #SEGUIR AQUÍIIIIIIIIIII
    for i in range(len(ESCL[:,0])):        
        if numCols>1:#PROBAR METER TODO ESTO EN LA PARTE ANTERIOR 
            var1.append(('{0:.2f} {1} {2:.2f}').format(group1.iloc[:,i].mean(), '±', group1.iloc[:,i].std()))
            var2.append('%.2f ± %.2f'%(group2.iloc[:,i].mean(), group2.iloc[:,i].std()))
        else: #cuando solo hay una variable
            var1.append(('{0:.2f} {1} {2:.2f}').format(x.mean(), '±', x.std()))
            #var1.append(('%.2f '+u'±'+' %.2f').format(x.mean(), x.std()))
            var2.append('%.2f ± %.2f'%(group2.mean(), y.std()))
       
        #prueba t muestras relacionadas
        #para muestras independientes ttest_ind(a, b[, axis, equal_var])
        if muestras_dep==1:
            t, p_val = stats.ttest_rel(x, y, axis=0)
            
        else: #para muestras indep
            ########
            #comprueba igualdad varianzas
            w, p_lev=stats.levene(group1.iloc[:,i], group2.iloc[:,i])
            igualVarianzas=True #por defecto
            if p_lev<0.05:
                igualVarianzas=False
            ########
            t, p_val = stats.ttest_ind(group1.iloc[:,i], group2.iloc[:,i], equal_var=igualVarianzas)
        

        p.append('%.3f'%p_val)
        es.append('%.2f [%.2f, %.2f]'%(ESCL[i,0], ESCL[i,1], ESCL[i,2]))
    """
    dfTablaResumen[grouplabels[0]] = var1
    dfTablaResumen[grouplabels[1]] = var2
    dfTablaResumen['p'] = p
    dfTablaResumen['ES [95% CI]'] = es
    dfTablaResumen.reindex(columns = ['', 'Var1', 'Var2', 'p' 'ES [95% CI]'])
    #transforma el mas menos para que se pueda excribir en Excel
    dfTablaResumen=dfTablaResumen.replace({'±': u'±'}, regex=True)
    ###########################################
    
    if show:
        _plot(ESCL, varlabels, decES, borders, numbers, grid, ax)
        
    return ESCL, dfTablaResumen

def cohen_d(x,y):
    from numpy import mean, std # version >= 1.7.1 && <= 1.9.1
    from math import sqrt
    
    return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
    
    
def _plot(ESCL, varlabels, decES, borders, numbers, grid, axx):
    """Grafica de tamaños efecto."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
       
        
        if axx is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, len(ESCL)*0.5))
        else: ax=axx
        
        #plt.rcParams.update(plt.rcParamsDefault)
        #fig.subplots_adjust(top=.9, bottom=0.1, left=0.52, right=0.99)
        out = ESCL[:, 1]*ESCL[:, 2] > 0       # CIs that don't contain the true mean
        
        ind = np.arange(0, len(ESCL))
        ind = ind[::-1]#les da la vuelta para que los ponga del primero al último
        ax.axvline(x=0, color=[0, 0, 0], zorder=0) #linea vertical en el cero
#       ax.spines['left'].set_position(('data', 1))
        #ax.plot([ESCL[np.logical_not(out), 1], ESCL[np.logical_not(out), 2]], [ind[np.logical_not(out)], ind[np.logical_not(out)]], color=[0.4, 0.4, 0.4, 1], marker='', ms=10, linewidth=2) #barra solo para los no significativos
#        ax.plot([ESCL[out, 1], ESCL[out, 2]], [ind[out], ind[out]], color=[1, 0, 0, 1], marker='', ms=10, linewidth=2) #barra solo para los significativos
#        ax.plot(ESCL[:,0], ind, color=[0, 1, 0, 1], marker='.', ms=10, linestyle='')#marcador para el valor g

#        ax.errorbar(ESCL[:,0], ind, xerr= [ESCL[:,0]-ESCL[:, 1], ESCL[:, 2]-ESCL[:,0]],
#            fmt='o', ms=7, color='b', ecolor='r', capthick=2)
        
        #barra solo para los no significativos        
        plotline, caps, barlinecols = ax.errorbar(ESCL[np.logical_not(out),0], ind[np.logical_not(out)], xerr= [ESCL[np.logical_not(out),0]-ESCL[np.logical_not(out), 1], ESCL[np.logical_not(out), 2]-ESCL[np.logical_not(out),0]],
            fmt='o', ms=5, color='0.6', elinewidth=1.5, capsize=3, capthick=2, solid_capstyle='round', zorder=2)
        for cap in caps: #como no funciona el solid_capstyle='round', hace los caps redondeados uno a uno
            cap._marker._capstyle = 'round'
        
        #barra solo para los significativos
        plotline, caps, barlinecols = ax.errorbar(ESCL[out,0], ind[out], xerr= [ESCL[out,0]-ESCL[out, 1], ESCL[out, 2]-ESCL[out,0]],
            fmt='o', ms=6, color='0', elinewidth=1.5, capsize=3, capthick=2, solid_capstyle='round', zorder=2)
        for cap in caps:
            cap._marker._capstyle = 'round'
        #ax.set_xlim(-2, 2)
        
        #Ajusta el eje vertical para dejar un poco de espacio por encima y por debajo
        ax.set_ylim(-0.5, len(ESCL)-0.5)
        
        if numbers==True:
            for i in range(len(ESCL)):
                #ax.annotate(str(ESCL[:,0]), xy=([1,1],[1.3,2.5]), zorder=10, ha='center', va='bottom')
                if np.isnan(ESCL[i,0]):
                    plt.text(0.0, ind[i]+.25, 'nan',
                         ha = 'center',va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), size = 10, zorder=1)
                else:
                    plt.text(ESCL[i,0], ind[i]+.25, '{0:.{dec}f} [{1:.{dec}f}, {2:.{dec}f}]'.format(ESCL[i,0],ESCL[i,1],ESCL[i,2], dec=decES),
                             ha = 'center',va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), size = 10, zorder=1)
            
        
        if borders==False:
            #quita los ticks de la derecha e izquierda
            ax.tick_params(left=False)#quita los ticks del lado izquierdo #labelbottom='off', bottom=False, labelleft='off', 
            
            #quita las líneas de los bordes excepto la de abajo
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
                
        #pinta líneas horizontales              
#        if grid:            
#            ax.grid(b=None, axis='y') #el vertical siempre desactivado
        ax.yaxis.grid(grid)
        
        ax.set_xlabel('Effect size', fontsize=11)
        if(len(varlabels)>0): #escribe las etiquetas en el eje vertical
            #plt.yticks(ind, varlabels, fontsize=12, ha = 'right') #ha es akineación horizontal; rotation='horizontal',
            ax.set_yticks(ind)
            ax.set_yticklabels(varlabels, fontsize=12, ha = 'right')
                    
        plt.xticks(fontsize=12)
        
        #plt.margins(.1)
        plt.tight_layout(rect=[0,0,1,0.95])
        
        if axx is None:
            plt.show()

#%%        
if __name__ == '__main__':
    
    help(Hedges_g)
    
    """    
    x = np.vstack((x, x+np.random.rand(1)/5, x+np.random.rand(1)/2))  # simulate two instants (two rows)
    y = np.vstack((y, y+np.random.rand(1)/5, y+np.random.rand(1)/2))"""
    
    """
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 500)
    mu2, sigma2 = 0.7, 0.3 # mean and standard deviation
    s2 = np.random.normal(mu2, sigma2, 500)
    mu3, sigma3 = 1.2, 0.7 # mean and standard deviation
    s3 = np.random.normal(mu3, sigma3, 500)
    
    x = np.vstack((s1, s2, s3))
    y = x+np.random.rand(1)/5
    
    dfx=pd.DataFrame(x, columns=['var1', 'var2', 'var3'])
    dfy=pd.DataFrame(y, columns=['var1', 'var2', 'var3'])
    """
    
    

    import pandas as pd
    #nomVars=[]
    
    numVars = 5
    numDat = 50           
    
#    ###############################################
#    #crea medias y sd aleatorias
    for i in range(numVars):
        mux=2 + np.random.rand(numVars)*10
        muy=mux + np.random.rand(numVars)*1.1
        sdx=np.random.rand(numVars)
        sdy=sdx + np.random.rand(numVars)/4.2        
                
    x = np.random.normal(mux, sdx, (numDat, numVars))
    y = np.random.normal(muy, sdy, (numDat, numVars))
    
#    x = sdx * np.random.randn(numDat, numVars) + mux
#    y = sdy * np.random.randn(numDat, numVars) + muy
    
    
    nomVars= ["Var{0}".format(nom) for nom in range(numVars)]
    dfx=pd.DataFrame(x, columns=nomVars)
    dfy=pd.DataFrame(y, columns=nomVars)
    dfx.hist()
    dfy.hist()
 #    ###############################################

           
    import matplotlib.pyplot as plt
    plt.rcParams.update(plt.rcParamsDefault) #para el fondo blanco, etc
    
    plt.figure()
    plt.subplot()
    #plt.plot(xx[:,0], 'bo')
    #plt.plot(xx[:,1], 'ro')
    plt.plot(dfx, 'bo', label='x')
    plt.plot(dfy, 'r^', label='y')
    #plt.legend(loc='best')
    plt.title('todas las variables')
    plt.show()
    
    plt.figure()
    plt.subplot()
    #plt.plot(xx[:,0], 'bo')
    #plt.plot(xx[:,1], 'ro')
    plt.plot(dfx, dfy, 'o')    
    plt.title('por grupos')
    plt.show()
    
    print("Hedges g muestras relacionadas (medidas repetidas) [g_inf, g_sup]: ", Hedges_g(dfx, dfy, muestras_dep=1))
    Hedges_g(dfx, dfy, muestras_dep=True, borders=True, show=True)
    Hedges_g(dfx, dfy, varlabels=dfx.columns, muestras_dep=True, show=True)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.55))
    fig.suptitle("Hedges' g effect size", fontsize=14, fontweight='bold', y=1.06)
    ax.set_title('(independent samples)', y=1.03)
    Hedges_g(dfx, dfy, varlabels=dfx.columns, muestras_dep=False, numbers=True, ax=ax, show=True)
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.85))
    fig.suptitle("Hedges' g effect size", fontsize=14, fontweight='bold', y=1.06)
    ax.set_title('(related samples)', y=1.03)
    ESCI, tabla=Hedges_g(dfx, dfy, varlabels=dfx.columns, muestras_dep=False, numbers=True, show=True, grid=True, ax=ax)
    
    print(tabla)
    
    ESCI, tabla = Hedges_g(dfx, dfy, grouplabels=['G1', 'G2'], varlabels=dfx.columns, muestras_dep=0, leadingZero=True, numbers=True, show=True)
    
    print(tabla)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.75))
    fig.suptitle("Hedges' g effect size", fontsize=14, fontweight='bold', y=1.06)
    ax.set_title('(related samples)', y=1.03)
    ESCI, tabla=Hedges_g(dfx.iloc[:,0:1], dfy.iloc[:,0:1], decVariable=3, decP=3, decES=3, varlabels=dfx.columns[0:1], muestras_dep=False, numbers=True, show=True, ax=ax)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.75))
    fig.suptitle("Hedges' g effect size", fontsize=14, fontweight='bold', y=1.06)
    ax.set_title('(related samples)', y=1.03)
    ESCI, tabla=Hedges_g(dfx.iloc[:,0:1], dfy.iloc[:,0:1], decVariable=3, decP=3, decES=3, varlabels=dfx.columns[0:1], muestras_dep=False, grid=True, numbers=True, show=True, ax=ax)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.75))
    ESCI, tabla = Hedges_g(dfx, dfy, grouplabels=['G1', 'G2'], muestras_dep=False, leadingZero=True, numbers=True, show=True, ax=ax)
    print(tabla)
    
    
    #%%
    
    #para comprobar con nans
    x=np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    y=np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    dfx=pd.DataFrame(x, columns=['var1'])
    dfy=pd.DataFrame(y, columns=['var2'])
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.25))#apiñado
    fig.suptitle("Hedges' g effect size", fontsize=14, fontweight='bold')
    ax.set_title('(related samples)', y=1.03)
    ESCI, tabla=Hedges_g(dfx, dfy, varlabels=dfx.columns, muestras_dep=False, decVariable=4, decP=2, decES=3, numbers=True, show=True, ax=ax)
    
    #%%
    x=np.array([0.892, 0.903, 0.898, 0.878, 0.908, 0.945, 0.926, 0.932, 0.932, 0.879, 0.920, 0.882])
    y=np.array([0.889, 0.908, 0.891, 0.864, 0.929, 0.939, 0.934, 0.928, 0.965, 0.872 ,0.918, 0.872])
    dfx=pd.DataFrame(x, columns=['Tgs_NQ'])
    dfy=pd.DataFrame(y, columns=['Tgs_Q'])
    
    fig, ax = plt.subplots(1, 1, figsize=(4, numVars*0.75), dpi=200)
    fig.suptitle("Hedges' g effect size", fontsize=14, fontweight='bold', y=1.06)
    ax.set_title('(related samples)', y=1.07)
    ESCI, tabla=Hedges_g(dfx, dfy, grouplabels=['G1', 'G2'], varlabels=dfx.columns, muestras_dep=False, pctcl = 95, sdpooled= True, tipose='Nakagawa', decVariable=3, decP=3, decES=3, numbers=True, show=True, ax=ax)
    ax.set_xlim(-2,2)
    plt.show()
    
    #%%
    
    
    