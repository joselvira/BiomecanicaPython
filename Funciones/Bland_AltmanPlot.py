# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:02:27 2017

@author: YoMismo
"""

from __future__ import division, print_function #division #Ensure division returns float

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt


__author__ = 'Jose Luis Lopez Elvira'
__version__ = '1.1.4'
__date__ = '18/03/2021'


#%%
def bland_altman_plot(data1, data2, unidad='', etiquetaCasos=False, regr=0, tcrit_Exacto=False, n_decimales=1, ax=None, show_text=None, show_bias_LOA=False, color_lin=None, *args, **kwargs):
    """Realiza gráfico de Bland-Altman para dos variables con medidas similares.
    Ejemplo de explicación en: https://www.medcalc.org/manual/blandaltman.php
    
    Parameters
    ----------
    data1 : 1D array_like data pandas DataFrame.
    data2 : 1D array_like data pandas DataFrame.
    unidad: opcional, incluye la unidad de medida en las etiquetas de los ejes
    etiquetaCasos: opcional, pone un número al lado de cada caso
    regr: si es mayor que cero, incluye en el gráfico la línea de regresión con 
          el exponente indicado. También presenta el valor de la correlación 
          con su p, de R2 y del error medio cuadrático.
    tcrit_Exacto : con False toma el valor t crítico = 1.96; con True lo 
          calcula a partir de n (#stats.t.ppf(q=0.975, df=n1 + n2-2))
    n_decimales : especifica el número de decimales a mostrar en BIAS y LOA.
    ax : ejes para el grafico resultante.
    show_text : indica si muestra texto informativo.
                Puede ser 'bias_loa', 'regr', 'publication', 'all'.
                defoult=None.
    show_bias_LOA : True/False. Muestra el valor del bias y limits of agreement 
          en el gráfico.
    color_lin: Si se quiere controlar el color de las líneas bias y LOA. Por 
          defecto (None), mantiene color negro para bias y gris para LOA.
          Útil cuando se quieren solapar varios grupos de datos en la misma
          gráfica.
    *args y **kwargs especificaciones de formato para los puntos del grafico.
    
    Returns
    -------
    
    grafico Bland-Altman    
    Bias, media de las diferencias
    LOA, limits of agreement.
    
    Example
    -------    
    bland_altman_plot(s1, s2, lw=0, color='k', s=40)
    bias, LOA = bland_altman_plot(s1, s2, etiquetaCasos= True, regr=2, unidad='m', tcrit_Exacto=True, show_bias_LOA=True, lw=1, color='b', s=20, ax=ax)

    Version history
    ---------------
    '1.1.5', 12/07/2023
            Corregido error al calcular regresión cuando todas las x son iguales.
    
    '1.1.4':
            Añadido el argumento n_decimales para poder especificar el número de decimales cuando muesra el BIAS y LOA.
    
    '1.1.3':
            Exporta los LOA sin multiplicar por 2.
            Representa el texto del bias y LOA en las líneas correspondientes.
            color_lin controla el color del texto de bias_loa.
            Introducido parámetro show_text, que puede ser 'bias_loa', 'regr', 'publication' o 'all'. El parámetro show_bias_LOA
    
    '1.1.2':
            Corregidas las gráficas, con ax en lugar de plt. Antes no funcionaba con varios subplots.
            Con color_lin se puede controlar el color de las líneas bias y LOA. Útil cuando se quieren solapar gráficas de varios conjuntos de datos.
    '1.1.1':
            Cálculo R2 con sklearn, con statsmodels falla por la versión de scikit.
            Quita las filas con valores nulos.
            
    '1.1.0':
            Quitados adaptadores para etiquetas.
            Se puede elegir el tcrit 1.96 o ajustarlo a la n.
    """
    if len(data1) != len(data2):
        raise ValueError('Los dos grupos de datos no tienen la misma longitud.')
    
    #primero agrupa las dos variables en un dataframe para quitar los casos nulos de cualquiera de los dos.
    data= pd.concat([pd.DataFrame(data1), pd.DataFrame(data2)], axis=1).dropna()
    
    data1 = data.iloc[:,0]
    data2 = data.iloc[:,1]
        
    n1        = data1.notnull().sum() #pd.notnull(data1).count() #asi para que no cuente los NaN #np.isfinite(data1).count()
    n2        = data2.notnull().sum() #pd.notnull(data2).count()
          
    mean      = np.mean([data1, data2], axis=0)
    diff      = np.array(data1 - data2)         # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0, ddof=1)            # Standard deviation of the difference
    t_crit    = 1.96 if tcrit_Exacto==False else stats.t.ppf(q=0.975, df=n1 + n2-2) #por defecto 1.96, de esta forma se ajusta a la n
                
    if unidad!='':
        unidad= ' ('+unidad+ ')'
    
    # make plot if not axis was provided
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    #dibuja los puntos
    ax.scatter(mean, diff, zorder=2, *args, **kwargs)
    
    if regr!=0 and sum(mean)!=0.0:
        import seaborn as sns
                
        #linear regresion
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean, diff)
        #plt.plot(mean, slope*mean + intercept , 'r', alpha=.5, lw = 1)
        
        #con cualquier exponente de regresión
        orden=regr
        #import statsmodels.api as sm #para las regresiones #FALLA POR versión de SCIKIT...
        #R2 = sm.OLS(diff, xpoly).fit().rsquared
        
        #Calcula el modelo de regresión para obtener R2
        from sklearn.pipeline import Pipeline #estos para calcular las regresiones
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        X=mean
        y=diff
        polynomial_features = PolynomialFeatures(degree=orden, include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([('polynomial_features', polynomial_features),
                             ('linear_regression', linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)
        
        y_predict = pipeline.predict(X[:, np.newaxis])
        R2 = r2_score(y, y_predict)
        MSE = mean_squared_error(y, y_predict)
        
        #Gráfica de least squares fit line
        if color_lin==None:
            col='black'

        else:
            col=color_lin
        sns.regplot(x=mean, y=diff, scatter=False, order=orden, ax=ax, line_kws={'color':col, 'alpha':0.6, 'lw':2})
        
        cuadroTexto=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.1,rounding_size=.5')
        if show_text in ['regr', 'all']:
            ax.text(0.02, 0.01, 'r= {0:.3f}, p= {1:.3f}, $R^2$= {2:.3f} MSE= {3:.3f}'.format(r_value, p_value, R2, MSE), fontsize=10,
                     horizontalalignment='left', verticalalignment='bottom', color=col, bbox=cuadroTexto, transform=ax.transAxes, zorder=2)
        elif show_text in ['publication']:
            ax.text(0.02, 0.01, 'r= {0:.3f}'.format(r_value), fontsize=10,
                     horizontalalignment='left', verticalalignment='bottom', color=col, bbox=cuadroTexto, transform=ax.transAxes, zorder=2)
                   
        
    #dibuja la línea horizontal del cero
    ax.axhline(0.0, color='grey', linestyle='-', zorder=1, linewidth=1.0, solid_capstyle='round')
    
    if color_lin==None:
        #dibuja la línea horizontal de la media
        ax.axhline(md, color='black', linestyle='-', zorder=1, linewidth=2.0, solid_capstyle='round')
        
        #dibuja las líneas horizontales de los límites de acuerdo
        ax.axhline(md + t_crit*sd, color='gray', zorder=1, linestyle='--', dashes=(5, 2), dash_capstyle='round', linewidth=1.5)
        ax.axhline(md - t_crit*sd, color='gray', zorder=1, linestyle='--', dashes=(5, 2), dash_capstyle='round', linewidth=1.5)
                
    else:
        #dibuja la línea horizontal de la media
        ax.axhline(md, color=color_lin, linestyle='-', zorder=1, linewidth=2.0, solid_capstyle='round')
        
        #dibuja las líneas horizontales de los límites de confianza
        ax.axhline(md + t_crit*sd, color=color_lin, zorder=1, linestyle='--', dashes=(5, 2), dash_capstyle='round', linewidth=1.5)
        ax.axhline(md - t_crit*sd, color=color_lin, zorder=1, linestyle='--', dashes=(5, 2), dash_capstyle='round', linewidth=1.5)
        
        
        
    if etiquetaCasos:
        font = {'family': 'sans',
                    'color':  'red',
                    'weight': 'normal',
                    'size': 8,
                    'alpha': 0.7,
                }
        for num in range(len(data1)):            
            if ~np.isnan(mean[num]) and ~np.isnan(diff[num]):
                plt.text(mean[num], diff[num], str(num), fontdict=font)
    
    etiquetaY='Difference'
    etiquetaY=etiquetaY + unidad
    etiquetaX='Mean'
    etiquetaX=etiquetaX + unidad
    
    ax.set_xlabel(etiquetaX)
    ax.set_ylabel(etiquetaY)
        
    if show_text in ['bias_loa', 'publication', 'all'] or show_bias_LOA:
        if color_lin==None:
            color_lin='black'            
        
        cuadroTexto=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.1,rounding_size=.5')
        ax.text(ax.get_xlim()[1], md+(ax.get_ylim()[1]-ax.get_ylim()[0])/1000, 'Bias {0:.{dec}f}'.format(md, dec=n_decimales), fontsize=12, color=color_lin,
             horizontalalignment='right', verticalalignment='bottom', bbox=cuadroTexto, transform=ax.transData, zorder=2)
        
        ax.text(ax.get_xlim()[1], (md+t_crit*sd)+(ax.get_ylim()[1]-ax.get_ylim()[0])/1000, 'LOA {0:.{dec}f}'.format(md+t_crit*sd, dec=n_decimales), fontsize=10, color=color_lin,
             horizontalalignment='right', verticalalignment='bottom', bbox=cuadroTexto, transform=ax.transData)
        
        ax.text(ax.get_xlim()[1], (md-t_crit*sd)+(ax.get_ylim()[1]-ax.get_ylim()[0])/1000, 'LOA {0:.{dec}f}'.format(md-t_crit*sd, dec=n_decimales), fontsize=10, color=color_lin,
             horizontalalignment='right', verticalalignment='bottom', bbox=cuadroTexto, transform=ax.transData)
    plt.tight_layout()
           
    return(md, t_crit*sd)

#%%        
if __name__ == '__main__':
    import pandas as pd
    #%% Comprobar con los datos de la web https://rpubs.com/Cristina_Gil/B-A_analysis    	
    metodo_A = np.array([1, 5, 10, 20, 50, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    	
    metodo_B = np.array([8, 16, 30, 24, 39, 54, 40, 68, 72, 62, 122, 80, 181, 259, 275, 380, 320, 434, 479, 587, 626, 648, 738, 766, 793, 851, 871, 957, 1001, 960])
    	
    bland_altman_plot(metodo_A, metodo_B, regr=2, tcrit_Exacto=True)
    plt.show()

	#%%
    #Crea un conjunto de medidas con dos instrumentos. El 2º es como el 1º pero con un error gausiano
    np.random.seed(1)
    mu1, sigma1 = 0, 10.5 # media y SD del instrumento 1
    instr1 = np.random.normal(mu1, sigma1, 10)
    
    mu2, sigma2 = 3.1, 10.1 # # media y SD que se añade al instrumento 1
    instr2 = instr1+np.random.normal(mu2, sigma2, 10)
    
    #instr1=pd.DataFrame(instr1)
    #instr2=pd.DataFrame(instr2)
    
    #Muestra los datos
    plt.plot(instr1, 'bo')
    plt.plot(instr2, 'ro')
    plt.show()
    
    #y un scatter plot
    plt.plot(instr1, instr2, 'ro')
    plt.show()
    
    #%%
    #Crea el Bland-Altman plot básico
    bland_altman_plot(instr1, instr2)
    plt.show()
    
    #puede devolver los valores de bias (media de las diferencias) y limits of agreement, y también presentarlos en la gráfica
    bias, LOA = bland_altman_plot(instr1, instr2, show_bias_LOA=True)
    print('Bias = {:.2f}, LOA ={:.2f}'.format(bias, LOA))
    
    #Se puede controlar el color de los puntos (color), su tamaño (s)
    bland_altman_plot(instr1, instr2, color='r', s=80, show_bias_LOA=True)
    
    #Se puede pedir que etiquete cada caso para poder identificarlos
    bland_altman_plot(instr1, instr2, etiquetaCasos=True, color='b', show_bias_LOA=True)
    
    #También puede calcular si existe tendencia en los datos. Presenta la R2 y Pearson
    bland_altman_plot(instr1, instr2, regr=1, color='b', show_bias_LOA=True)
    
    #para poder controlar el aspecto de los ejes, etiquetas, etc. incluirlo en una figura
    fig, ax = plt.subplots()
    bland_altman_plot(instr1, instr2, etiquetaCasos=False, ax=ax, color='k', s=40)
    plt.title('Bland-Altman plot')
    ax.set_ylabel('Bias')
    ax.set_xlabel('Media entre instrumento 1 e instrumento 2')
    ax.set_xlim([-30,30])
    ax.set_ylim([-30,30])
    plt.show()
    
    #También puede calcular si existe tendencia en los datos. El número pasado en regs se utiliza como exponente de la línea de regresión utilizada.
    #Presenta Pearson y p de la correlación lineal y la R2 de la regresión del polinomio con exponente indicado
    bland_altman_plot(instr1, instr2, regr=1, color='b', show_bias_LOA=True)
    bland_altman_plot(instr1, instr2, regr=2, color='b', show_bias_LOA=True)
    
    
    
    #%%
    np.random.seed(9999)
    m1 = np.random.random(500)
    m2 = np.random.random(500)
    
    
    mediadif, LOA= bland_altman_plot(m1, m2, lw=0, color='k', s=40, show_text='all', regr=1, color_lin='grey')
    plt.title('Bland-Altman Plot')
    plt.show()
    
    #%%
    Datos = pd.read_excel(r"F:\Programacion\Python\Mios\Estadistica\EjemploDatos-Bland-AltmanPlot.xlsx", 'Hoja1', index_col=None, na_values=[" "])
    
        
    bland_altman_plot(Datos.iloc[:,0], Datos.iloc[:,1], lw=0, color='k', s=40, show_text='bias_loa')
    plt.title('Bland-Altman Plot')
    plt.show()
    
    
    
    #%%###############################################
    Data1 = np.array([10.3, 5.1, 3.2, 19.1, 8.1, 11.7, 7.1, 13.9, 4.0, 20.1, 27.5, 6.4, 30.1, 13.0, 10.0,
                            16.8, 17.3, 3.0, 25.6, 19.3, 15.0, 27.3, 14.2, 13.0,14.4, 22.1, 19.0, 18.0, 13.0, 25.6,
                            18.4, 12.6, 25.5, 15.7, 20.2, 16.5, 19.3, 10.0, 18.8, 24.0, 22.8])
    
    
    #Create an array with the second data set of group 1
    Data2 = np.array([8.9, 4.0, 8.1, 21.2, 8.1, 12.0, 4.5, 13.9, 4.0, 20.1, 27.5, 6.4, 40.3, 13.0, 10.0, 32.2,
                              17.1, 9.4, 25.2, 18.8, 15.0, 27.3, 21.3, 13.0, 14.4,22.1, 17.9, 3.0, 13.0, 19.0, 18.4,
                              12.6, 25.5, 15.7, 21.2, 16.5, 19.3, 10.0, 30.8, 9.0, 22.8])
    
    Data1=pd.Series(Data1)
    Data2=pd.Series(Data2)
    
    bland_altman_plot(Data2, Data1, unidad='cm')
    plt.title('Bland-Altman Plot')
    plt.show()
    
    #%%###############################################
    
    #cuando las muestras vienen de la misma población normal, cabe esperar que se salga de los límites 5 de cada 100
    mu1, sigma1 = 0, 0.9 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 100)
    mu2, sigma2 = 0, 0.9 # mean and standard deviation
    s2 = np.random.normal(mu2, sigma2, 100)
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, ax=ax, lw=0, color='k', s=40)
    plt.title('Dos muestras normales')
    plt.show()
    
    #%%###############################################
    
    #Cuando son proporcionales sale una línea con pendiente
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 1000)
    
    s2 = s1*1.5
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
        
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, ax=ax, lw=0, color='k', s=40, regr=1, show_text='publication')
    plt.title('proporcionales')
    plt.show()
    
    #%%###############################################
    
    #Cuando son exponenciales sale una curva rara
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 100)
    
    s2 = s1**3
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
        
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, ax=ax, lw=0, color='k', s=40, regr=3, show_text='all')
    plt.title('proporcionales')
    plt.show()
    #%%###############################################
    
    #Cuando son iguales sale una línea horizontal=0
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 100)    
    s2 = s1
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, ax=ax, lw=0, color='k', s=40)
    plt.title('iguales')
    plt.show()
    
    #%%###############################################
    #Cuando son iguales + una diferencia aleatoria normal sale una nube que tiende hacia abajo
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 100)
    mu2, sigma2 = 0.1, 0.9 # mean and standard deviation
    s2 = s1+np.random.normal(mu2, sigma2, 100)
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    kargs={'lw':0, 'color':'b','s':40, 'alpha':0.5}
    #bland_altman_plot(s1, s2, etiquetaCasos=True, ax=ax, **kargs, idioma='esp')#en python 2 no funciona
    bland_altman_plot(s1, s2, etiquetaCasos=True, ax=ax, lw=0, color='b', regr=1, alpha=0.5, s=40)
    plt.title('iguales + una diferencia aleatoria normal')
    plt.show()
    
    ################################################
    #%%
    #Cuando son iguales + una diferencia constante sale una línea horizontal= a la diferencia cte.
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 100)
    mu2, sigma2 = 0.1, 0.9 # mean and standard deviation
    s2 = s1+10
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, etiquetaCasos=True, ax=ax, lw=0, color='k', s=40)
    plt.title('iguales + una diferencia constante')
    plt.show()
    
    ################################################
    #%%
    #Prueba cuando hay algún nan entre datos
    mu1, sigma1 = 0, 0.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 10)
    mu2, sigma2 = 0.1, 0.9 # mean and standard deviation
    s2 = s1+np.random.normal(mu2, sigma2, 10)
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    s1[4]=np.nan
    s2[4]=np.nan    
    
    #s2[6]=np.nan 
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, etiquetaCasos=True, ax=ax, lw=0, color='k', s=40)
    plt.title('iguales + una diferencia aleatoria normal')
    plt.show()
    
    ################################################
    #%% Calcula con correlación
    #Cuando son iguales + una diferencia aleatoria normal sale una nube que tiende hacia abajo
    #np.random.seed(1)
    mu1, sigma1 = 0, 10.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 50)
    mu2, sigma2 = 3.1, 10.1 # mean and standard deviation
    s2 = s1+np.random.normal(mu2, sigma2, 50)
    
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=150)
    bland_altman_plot(s1, s2, etiquetaCasos=False, ax=ax, lw=0, color='k', regr=1, s=40)
    plt.title('iguales + una diferencia aleatoria normal')
    plt.show()
    
        
    #%% con varios subplots
    mu1, sigma1 = 0, 10.5 # mean and standard deviation
    s1 = np.random.normal(mu1, sigma1, 50)
    mu2, sigma2 = 3.1, 10.1 # mean and standard deviation
    s2 = s1+np.random.normal(mu2, sigma2, 50)
    s1=pd.Series(s1)
    s2=pd.Series(s2)
    
    
    mu12, sigma12 = 10, 25.5 # mean and standard deviation
    s12 = np.random.normal(mu12, sigma12, 30)
    mu22, sigma22 = 15.1, 20.1 # mean and standard deviation
    s22 = s12+np.random.normal(mu22, sigma22, 30)
    s12=pd.Series(s12)
    s22=pd.Series(s22)
    
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5), dpi=150) #, constrained_layout=True
   
    
    bland_altman_plot(s1, s2, etiquetaCasos=False, ax=ax[0], lw=0, color='k', regr=1, s=40, show_text='bias_loa', n_decimales=3)
    #plt.xlim(0.244, 0.252)
    ax[0].set_title('Gráfica1')
    
    
    bland_altman_plot(s12, s22, etiquetaCasos=False, ax=ax[1], lw=0, color='k', regr=1, s=40)
    #plt.xlim(0.244, 0.252)
    ax[1].set_title('Gráfica2')
    
    
    #%% Con color uniforme para cada conjunto de datos
    fig, ax = plt.subplots(figsize=(5,5), dpi=150) #, constrained_layout=True
   
    
    bland_altman_plot(s1, s2, etiquetaCasos=False, ax=ax, regr=1, s=70, lw=0, color='b', alpha=0.6, color_lin='b', show_text='bias_loa')
    #plt.xlim(0.244, 0.252)
        
    
    bland_altman_plot(s12, s22, etiquetaCasos=False, ax=ax, regr=1, s=70, lw=0, color='r', alpha=0.6, color_lin='r', show_text='bias_loa')
    #plt.xlim(0.244, 0.252)
    ax.set_title('Gráfica2')
    
# %%
       
    
