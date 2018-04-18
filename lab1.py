# -*- coding: utf-8 -*-

import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
from scipy import fft, ifft, arange
import matplotlib.pyplot as plt

#print(rate) ##frecuencia de muestreo
#print(info)


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

def graphTime(t, signal):
	plt.plot(t,signal)
	plt.title("Audio con respecto al tiempo")
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	plt.show()

def fourierTransformation(signal, len_signal):
	fourierT = fft(signal)
	fourierNorm = fourierT/len_signal
	#print("FN", fourierNorm)
	"""k = arange(len_signal)
	tiempo = len_signal/rate
	frq = k/tiempo"""
	xfourier = np.fft.fftfreq(len(fourierNorm),1/rate)
	return xfourier, fourierNorm

def graphTransformation(xfourier,yfourier):
    
	plt.title("Amplitud vs Frecuencia")
	plt.xlabel("Frecuencia [Hz]")
	plt.ylabel("Amplitud [dB]")
	plt.plot(xfourier,abs(yfourier))
	plt.show()

def getInverseFourier(yfourier,len_freq):
		fourierTInv = ifft(yfourier)*len_freq
		return fourierTInv

def graphWithInverse(time, invFourier):
	plt.title("Amplitud vs Inversa de Fourier")
	plt.xlabel("Tiempo[s] (IFFT)")
	plt.ylabel("Amplitud [dB]")
	plt.plot(time,invFourier)
	plt.show()

def getMax(yfourier):
	maxValue = max(yfourier)
	#print("Maximo: ", maxValue)
	return maxValue

def getIndexValue(value, yfourier):
	index = 0
	for i in range(len(yfourier)):
		if (yfourier[i] == value):
			index = i
	return i

def removeNoise(maxValue, yfourier):
	n_amplitude = len(yfourier)
	n_amplitude = int(n_amplitude/2)
	
	print("n_amplitude: ",n_amplitude)
	fifteen_percent_ampl = n_amplitude*0.15
	fifteen_percent_ampl = int(fifteen_percent_ampl)
	print("fifteen: ", fifteen_percent_ampl)
	withoutNoise = np.zeros(n_amplitude*2,np.complex256)
	pos = getIndexValue(maxValue,yfourier)
	#print("Indice: ", pos)
	min_pos = pos - fifteen_percent_ampl
	max_pos = pos + fifteen_percent_ampl
	#print("Posicion minima: ", min_pos)
	#print("Posicion maxima: ", max_pos)
	#print(len(yfourier))
	#print(len(withoutNoise))
	#print("Without Noise:", withoutNoise)
	withoutNoise[min_pos:max_pos] = yfourier[min_pos:max_pos]
	return withoutNoise
	
def graphWithoutNoise(t,inverseWithoutNoise):
    plt.plot(t,inverseWithoutNoise,"--")
    plt.title("Audio con respecto al tiempo sin ruido (ifft)")
    plt.xlabel("Tiempo [s] (IFFT)")
    plt.ylabel("Amplitud [dB]")
    plt.show()


def showMenu():
	print("    MENU\n\n")
	print("1) Gráfico del audio original: Amplitud vs Tiempo")
	print("2) Gráfico de la Transformada de Fourier: Amplitud vs Frecuencia")
	print("3) Gráfico de la Anti-Transformada fourier : Amplitud vs Tiempo (IFFT) ")
	print("4) Gráfico de la Tranformada truncado al 15%: Amplitud vs Frecuencia")
	print("5) Grafico de la Anti-Transformada truncado al 15%: Amplitud vs Frecuencia")

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
	else:
		print("Opción inválida")





		

	

"""




print("WN:",without_noise)

"""

#write("beacon3.wav",rate,inv_without_noise)