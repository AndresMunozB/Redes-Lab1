# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:12:11 2016

@author: Francisco
"""



import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace
from scipy.fftpack import fft, ifft



#==============================================================================
# Qué hace la función?: Trabajo con los datos de audio, obtengo la cantidad de datos y determino el tiempo
# Parámetros de entrada: Matriz con los datos de la amplitud del audio
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y un vector con los tiempos de la señal
#==============================================================================
def obtener_datos_1(info):
    señal = info[:,0]#obtengo el vector de solo un canal de audio
    largo_señal = len(señal)#obtengo el largo de la señal
    largo_señal = float(largo_señal)
    tiempo = float(largo_señal)/float(rate)#genero el tiempo total del audio
    x = arange(0,tiempo,1.0/float(rate))#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    return señal,largo_señal,x


#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_audio_respecto_tiempo(señal,x):
    plt.plot(x,señal,"--")
    plt.title("Audio con respecto al tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.show
    
    
#==============================================================================
# Qué hace la función?: Genera la transformada de fourier de una señal dada
# Parámetros de entrada: Vector con la señal a transformar y el largo de la señal
# Parámetros de salida: Vector con la amplitud transformada y un vector con las frecuencias
#==============================================================================    
def obtener_transformada_fourier(señal,largo_señal):
    transFour = fft(señal,largo_señal)#eje Y
    transFourN = transFour/largo_señal#eje y normalizado
    
    aux = linspace(0.0,1.0,largo_señal/2+1)#obtengo las frecuencias
    xfourier = rate/2*aux#genero las frecuencias dentro del espectro real
    yfourier = transFourN[0.0:largo_señal/2+1]#genero la parte necesaria para graficar de la transformada
    return xfourier,yfourier
    
    
#==============================================================================
# Qué hace la función?: Grafica la transformada de fourier de una función
# Parámetros de entrada: -vector de amplitudes (EJE Y) y vector con frecuencias (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_transformada_1(xfourier,yfourier):
    plt.plot(xfourier,abs(yfourier))
    plt.title("Amplitud respecto a la frecuencia (fft)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.show


#==============================================================================
# Qué hace la función?: Aplica la transformada de fourier inversa a una señal
# Parámetros de entrada: Señal con amplitudes y largo de la señal
# Parámetros de salida: Vector con la transformada inversa
#==============================================================================
def aplicar_inversa_fourier(yfourier,largo_señal):
    inverTransFour = ifft(yfourier*largo_señal,largo_señal)
    return inverTransFour



#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#============================================================================== 
def graficar_audio_respecto_tiempo_2(x,inverTransFour):
    plt.plot(x,inverTransFour,"--")
    plt.title("Audio con respecto al tiempo (ifft)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.show


#==============================================================================
# Qué hace la función?: Obtiene la amplitud maxima de un vector de amplitudes
# Parámetros de entrada: Vector con amplitudes
# Parámetros de salida: Valor maximo del vector
#==============================================================================
def obt_valor_maximo(yfourier):
    valor_maximo = max(yfourier)#obtengo la amplitud maxima
    return valor_maximo
    
#==============================================================================
# Qué hace la función?: Busca la posició en el vector donde se encuentra un valor dado
# Parámetros de entrada: Valor a buscar y arreglo de datos
# Parámetros de salida: Posicion en la que se encuentra dicho valor
#==============================================================================
#FUNCION PARA BUSCAR LA POSICION DE LA AMPLITUD MAXIMA
def buscar_Pos_Maximo(maximo,arreglo):
    for i in range(len(arreglo)):
        if arreglo[i] == maximo:
            return i


#==============================================================================
# Qué hace la función?: Genera los datos necesarios para eliminar el ruido de una vector de audio
# Parámetros de entrada: Vector con amplitudes y el valor con la amplitud maxima
# Parámetros de salida: Vector con amplitudes sin ruido
#==============================================================================
def generar_datos_nuevo_espectro(yfourier,valor_maximo):
    total_amplitudes = len(yfourier)#obtengo la cantidad total de datos
    total_amplitudes = total_amplitudes/2 #esto debido a que la mitad de los datos son negativos y no nos sirven
    quince_amplitudes = total_amplitudes*0.075#obtengo el 15 porciento del total de las amplitudes
    quince_amplitudes = int(quince_amplitudes)

    sin_ruido = np.zeros(total_amplitudes*2)#GENERO UNA MATRIZ DE CEROS
    posicion = buscar_Pos_Maximo(valor_maximo, yfourier)#ENCUENTRO LA POSICION DONDE SE ENCUENTRA LA MAXIMA AMPLITUD
    minima_posicion = posicion-quince_amplitudes#GENERO LA POSICION MINIMA DEL RANGO DE 15%
    maxima_posicion = posicion+quince_amplitudes#GEENERO LA POSICON MAXIMA DEL RANGO DE 15%
    print(minima_posicion)
    print(maxima_posicion)
    
    if (minima_posicion < 0):
        minima_posicion = 0
        
    if (maxima_posicion > total_amplitudes):
        maxima_posicion = total_amplitudes
    
    sin_ruido[minima_posicion:maxima_posicion] = yfourier[minima_posicion:maxima_posicion]
    return sin_ruido


#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#============================================================================== 
def graficar_audio_respecto_tiempo_3(x,inver_tans_four_sin_ruido):
    plt.plot(x,inver_tans_four_sin_ruido,"--")
    plt.title("Audio con respecto al tiempo sin ruido (ifft)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.show


#==============================================================================
# Qué hace la función?: Grafica la transformada de fourier de una función
# Parámetros de entrada: -vector de amplitudes (EJE Y) y vector con frecuencias (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================
def graficar_transformada_2(xfourier,yfourier):
    plt.plot(xfourier,abs(yfourier))
    plt.title("Amplitud respecto a la frecuencia sin ruido (fft)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.show
    


#==============================================================================
# inicio del codigo a ejecutar
# PARA GENERAR LOS GRAFICOS ES NECESARIO DESCOMENTAR LINEAS DE CODIGO!!!!!
#==============================================================================

# PUNTO 1, IMPORTAR LA SEÑAL DE AUDIO
rate,info=read("beacon.wav")


# PUNTO 2, GRAFICAR EL AUDIO CON RESPECTO AL TIEMPO
# obtener datos:
señal,largo_señal,x = obtener_datos_1(info)
#FINALMENTE GRAFICO LA FUNCION DEL AUDIO CON RESPECTO AL TIEMPO
#graficar_audio_respecto_tiempo(señal,x)


#PUNTO 3, UTILIZANDO LA TRANSFORMADA DE FOURIER:
# 3.A, GRAFICAR LA SEÑAL EN EL DOMINIO DE LA FRECUENCIA
xfourier,yfourier = obtener_transformada_fourier(señal,largo_señal)
#FINALMENTE GRAFICO LA FUNCION DE LA AMPLITUD CON RESPECTO A LA FRECUENCIA
#graficar_transformada_1(xfourier,yfourier)

# 3.B APLICAR LA INVERSA A LA FUNCIÓN ANTERIOR
inverTransFour = aplicar_inversa_fourier(yfourier,largo_señal)
#FINALMENTE GRAFICO LA FUNCION DEL AUDIO CON RESPECTO AL TIEMPO en base a la antitransformada
#graficar_audio_respecto_tiempo_2(x,inverTransFour)


# 4 EN EL DOMINIO DE LA FRECUENCIA
# 4.A ANALIZAR EL ESPECTRO Y ANALIZAR LOS COMPONENTES CON MAYOR AMPLITUD
valor_maximo = obt_valor_maximo(yfourier)


# 4.B GENERAR UN NUEVO ESPECTRO DENTRO DEL MARGEN DEL 15% DE LA MAXIMA AMPLITUD
sin_ruido = generar_datos_nuevo_espectro(yfourier,valor_maximo)
#graficar_transformada_2(xfourier,sin_ruido)

# 4.C CALCULAR LA TRANSFORMADA INVERSA DE LOS DATOS SIN RUIDO

inver_tans_four_sin_ruido = aplicar_inversa_fourier(sin_ruido,largo_señal)
#FINALMENTE GRAFICO LA FUNCION DEL AUDIO CON RESPECTO AL TIEMPO en base a la antitransformada sin ruido
graficar_audio_respecto_tiempo_3(x,inver_tans_four_sin_ruido)


