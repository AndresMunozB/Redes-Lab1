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

def graphTime(signal, x):
	plt.plot(x,signal)
	plt.title("Audio con respecto al tiempo")
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	plt.show

def fourierTransformation(signal, len_signal):
	fourierT = fft(signal, len_signal)
	fourierT_norm = fourierT/len_signal

	aux = linspace(0.0, 1.0, len_signal/2+1)
	xfourier = rate/2*aux
	yfourier = fourierT_norm[0.0: len_signal/2+1]
	return xfourier, yfourier

"""def graficar_transformada_1(xfourier,yfourier):
    plt.plot(xfourier,abs(yfourier))
    plt.title("Amplitud respecto a la frecuencia (fft)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.show

def aplicar_inversa_fourier(yfourier,len_signal):
    fourierTInv = ifft(yfourier*len_signal,len_signal)
    return inverTransFour


dimension = info[0].size
print(dimension)
print(info[0])
#data: datos del audio(arreglo de numpy)
if(dimension == 1):
	data =  info
	perfect = 1
else:
	data = info[:,dimension-1]
	perfect = 0

#timp = tiempo que dura todo el audio
timp = len(data)/rate
#print('data: ',data)

t = linspace(0,timp,len(data)) #linspace(start,stop,number)
#print(len(data))
#print('timp: ',timp)
plt.title('Audio')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [dB]')
plt.plot(t,data)
plt.show()




large = len(data)
#print(large)
k = arange(large)
#print('k: ',k)


T = large/rate
frq = k/timp
Y = fft(data)

#print(Y)
largeY= len(Y)
y2 = np.fft.fftfreq(largeY,1/rate)


#plt.plot(frq,abs(Y))
#plt.show()


otro = ifft(Y) #transformada inversa 
u = k/otro	
print(u)

plt.plot(y2, frq)
plt.show()"""

señal, largo_señal, x = getDatos(info,rate)
xfourier, yfourier = fourierTransformation(señal, largo_señal)




