# -*- coding: utf-8 -*-

import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
from scipy import fft, ifft, arange
import matplotlib.pyplot as plt
rate,info = read("beacon.wav")
print(rate) ##frecuencia de muestreo
print(info)


def getDatos(info,rate):
	#Datos del audio.
	signal = info[:,0]
	print(signal)
	#Largo de todos los datos.
	len_signal = len(signal)
	#Transformado a float.
	len_signal = float(len_signal)
	#Duración del audio.
	time = len_signal/float(rate)
	print(time)
	#Eje x para el gráfico, de 0 a la duración del audio.
	t = linspace(0, time, len_signal)
	print(t)
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
	print("FN", fourierNorm)
	k = arange(len_signal)
	tiempo = len_signal/rate
	frq = k/tiempo
	xfourier = np.fft.fftfreq(len(fourierNorm),1/rate)
	return xfourier, fourierNorm

def graphTransformation(xfourier,yfourier):
    
	plt.title("Amplitud vs Frecuencia")
	plt.xlabel("Frecuencia [Hz]")
	plt.ylabel("Amplitud [dB]")
	plt.plot(xfourier,abs(yfourier))
	plt.show()

def getInverseFourier(yfourier):
    fourierTInv = ifft(yfourier)
    return fourierTInv

def graphWithInverse(time, invFourier):
	plt.title("Amplitud vs Inversa de Fourier")
	plt.xlabel("Tiempo[s]")
	plt.ylabel("Amplitud [dB]")
	plt.plot(time,invFourier)
	plt.show()

def getMax(listValues):
	maxValue = max(listValues)
	return maxValue

def getIndexValue(value, listValues):
	for i in range(len(listValues)):
		if (listValues[i] == value):
			return i

def removeNoise(maxValue, listFourier):
	n_amplitude = len(listFourier)
	fifteen_percent_ampl = n_amplitude*0.075
	fifteen_percent_ampl = int(fifteen_percent_ampl)

	withoutNoise = np.zeros(n_amplitude)
	pos = getIndexValue(maxValue,listFourier)
	min_pos = pos - fifteen_percent_ampl
	max_pos = pos + fifteen_percent_ampl
	print("Posicion minima: ", min_pos)
	print("Posicion maxima: ", max_pos)
	print(len(listFourier))
	print(len(withoutNoise))
	print("Without Noise:", withoutNoise)
	withoutNoise[min_pos:max_pos] = listFourier[min_pos:max_pos]
	return withoutNoise
	
def graphWithoutNoise(t,inverseWithoutNoise):
    plt.plot(t,inverseWithoutNoise,"--")
    plt.title("Audio con respecto al tiempo sin ruido (ifft)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.show()

señal, largo_señal, t = getDatos(info,rate)
xfourier, yfourier = fourierTransformation(señal, largo_señal)
graphTime(t,señal)
graphTransformation(xfourier,yfourier)
invFourier = getInverseFourier(yfourier)
graphWithInverse(t,invFourier)
max_value = getMax(yfourier)
without_noise = removeNoise(max_value, yfourier)
print("WN:",without_noise)
inv_without_noise = getInverseFourier(without_noise)
graphWithoutNoise(t, inv_without_noise)



