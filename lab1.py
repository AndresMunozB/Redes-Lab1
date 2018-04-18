# -*- coding: utf-8 -*-

import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
from scipy import fft, ifft, arange
from pylab import savefig
import matplotlib.pyplot as plt

#==============================================================================
# Función: En base a los datos que entrega beacon.wav se obtiene			   
# los datos de la señal, la cantidad de datos que esta tiene, y el tiempo que  
# que dura el audio.														   
# Parámetros de entrada: Matriz con los datos de la amplitud del audio.
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y 
# un vector con los tiempos de la señal.
#==============================================================================
def getDatos(info,rate):
	#Datos del audio.
	signal = info[:,0]
	#print(signal)
	#Largo de todos los datos.
	len_signal = len(signal)
	#Transformado a float.
	len_signal = float(len_signal)
	#Duración del audio.
	time = len_signal/float(rate)
	#print(time)
	#Eje x para el gráfico, de 0 a la duración del audio.
	t = linspace(0, time, len_signal)
	#print(t)
	return signal, len_signal, t

#=============================================================================
# Función: Grafica los datos del audio en función del tiempo.
# Parámetros de entrada: El arreglo con los datos del tiempo y los datos de la
# señal.
# Parámetros de salida: Ninguno, pero se muestra un gráfico por pantalla.
#=============================================================================
def graphTime(t, signal):
	plt.plot(t,signal)
	plt.title("Audio con respecto al tiempo")
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	savefig("Audio con respecto al tiempo.png")
	plt.show()

#=============================================================================
# Función: Función que se encarga de obtener la transformada de Fourier de los
# datos de la señal.
# Parámetros de entrada: Un arreglo con los datos de la señal, y el largo de 
# este arreglo.
# Parámetros de salida: Dos arreglos, uno con los valores del eje x y otro con
# los valores del eje y.
#=============================================================================
def fourierTransformation(signal, len_signal):
	fourierT = fft(signal)
	fourierNorm = fourierT/len_signal
	xfourier = np.fft.fftfreq(len(fourierNorm),1/rate)
	return xfourier, fourierNorm


#===============================================================================
# Función: Grafica la transformada de Fourier, usando los arreglos de la función
# anterior.
# Parámetros de entrada: arreglo con los valores del eje x y arreglo con los 
# valores del eje y.
# Parámetros de salida: Ninguno, se muestra un gráfico por pantalla.
#===============================================================================
def graphTransformation(xfourier,yfourier):
    
	plt.title("Amplitud vs Frecuencia")
	plt.xlabel("Frecuencia [Hz]")
	plt.ylabel("Amplitud [dB]")
	plt.plot(xfourier,abs(yfourier))
	savefig("Amplitud vs Frecuencia.png")
	plt.show()

#===============================================================================
# Función: Obtiene la anti-transformada de fourier del arreglo yfourier.
# Parámetros de entrada: Arreglo yfourier y el largo de este arreglo.
# Parametros de salida: Un arreglo con los valores de la anti transformada de 
# Fourier.
#===============================================================================
def getInverseFourier(yfourier,len_freq):
		fourierTInv = ifft(yfourier)*len_freq
		return fourierTInv

#===============================================================================
# Función: Grafica la anti-transformada de fourier.
# Parámetros de entrada: Arreglo del tiempo y arreglo de la anti-transformada
# de Fourier.
# Parámetros de salida: Ninguno, se muestra un gráfico por pantalla.
#===============================================================================
def graphWithInverse(time, invFourier):
	plt.title("Amplitud vs Inversa de Fourier")
	plt.xlabel("Tiempo[s] (IFFT)")
	plt.ylabel("Amplitud [dB]")
	plt.plot(time,invFourier)
	savefig("Amplitud vs Inversa de Fourier.png")
	plt.show()

#===============================================================================
# Función: Obtiene el máximo del arreglo con los valores de Fourier.
# Parámetros de entrada: Arreglo con los valores de la transformada de fourier.
# Parámetros de salida: Valor máximo del arreglo.
#===============================================================================
def getMax(yfourier):
	maxValue = max(yfourier)
	return maxValue

#========================================================================================
# Función: Obtiene el índice del valor máximo que se entrega como parámetro,
# dentro del arreglo yfourier.
# Parámetros de entrada: Valor a buscar y arreglo yfourier (valores de las amplitudes de
# la transformada de fourier)
# Parámetros de salida: Indice del valor que se quería encontrar.
#========================================================================================
def getIndexValue(value, yfourier):
	index = 0
	for i in range(len(yfourier)):
		if (yfourier[i] == value):
			index = i
	return i

#========================================================================================
# Función: Esta función hace el truncamiento del 15% de los datos, para reducir el ruido
# de la señal.
# Parámetros de entrada: Valor máximo del arreglo de las amplitudes (eje Y de la transformada)
# Parámetros de salida: El arreglo de las amplitudes pero con el ruido reducido.
#========================================================================================
def removeNoise(maxValue, yfourier):
	n_amplitude = len(yfourier)
	n_amplitude = int(n_amplitude)
	
	fifteen_percent_ampl = n_amplitude*0.15
	fifteen_percent_ampl = int(fifteen_percent_ampl)
	withoutNoise = np.zeros(n_amplitude,np.complex256)
	pos = getIndexValue(maxValue,yfourier)
	min_pos = pos - fifteen_percent_ampl
	max_pos = pos + fifteen_percent_ampl
	withoutNoise[min_pos:max_pos] = yfourier[min_pos:max_pos]
	return withoutNoise

#================================================================================================
# Función: Grafica la señal sin ruido.
# Parámetros de entrada: Arreglo de los valores del tiempo y arreglo datos de la señal sin ruido.
# Parámetros de salida: Ninguno, pero muestra un gráfico por pantalla.
#===============================================================================================	
def graphWithoutNoise(t,inverseWithoutNoise):
	plt.plot(t,inverseWithoutNoise,"--")
	plt.title("Audio con respecto al tiempo sin ruido (ifft)")
	plt.xlabel("Tiempo [s] (IFFT)")
	plt.ylabel("Amplitud [dB]")
	savefig("Audio con respecto al tiempo sin ruido (ifft).png")
	plt.show()

def showMenu():
	print("\n\t\t\tMENU PARA SELECCIÓN DE GRÁFICOS\n")
	print("   1) Gráfico del audio original: Amplitud vs Tiempo")
	print("   2) Gráfico de la Transformada de Fourier: Amplitud vs Frecuencia")
	print("   3) Gráfico de la Anti-Transformada fourier : Amplitud vs Tiempo (IFFT) ")
	print("   4) Gráfico de la Tranformada truncado al 15%: Amplitud vs Frecuencia")
	print("   5) Grafico de la Anti-Transformada truncado al 15%:Amplitud vs Frecuencia")

rate,info = read("beacon.wav")
señal, largo_señal, t = getDatos(info,rate)
xfourier, yfourier = fourierTransformation(señal, largo_señal)
invFourier = getInverseFourier(yfourier,len(xfourier))
max_value = getMax(yfourier)
without_noise = removeNoise(max_value, yfourier)
inv_without_noise = getInverseFourier(without_noise,len(t))
menu = 0
while menu != str(6):
	showMenu()
	menu = input("Ingrese una opción: ")
	if(menu == "1"):
		graphTime(t,señal)
	elif(menu == "2"):
		graphTransformation(xfourier,yfourier)
	elif(menu == "3"):
		graphWithInverse(t,invFourier)
	elif(menu == "4"):
		graphTransformation(xfourier,without_noise)
	elif(menu == "5"):
		graphWithoutNoise(t, inv_without_noise)
		write("beacon2.wav",rate,inv_without_noise.astype(info.dtype))
	else:
		print("Opción inválida")