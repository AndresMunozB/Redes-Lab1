import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
from scipy import fft, ifft, arange
import matplotlib.pyplot as plt
rate,info = read("beacon.wav")
#print(rate) ##frecuencia de muestreo
#print(info)

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
#plt.plot(t,data)
#plt.show()




large = len(data)
#print(large)
k = arange(large)
#print('k: ',k)


T = large/rate
frq = k/timp
h = 1/frq

print ('h: ', h) 
Y = fft(data)
#print(Y)
largeY= len(Y)


#plt.plot(frq,abs(Y))
#plt.show()


otro = ifft(Y) #transformada inversa 
u = k/otro	
print(u)

plt.plot(u,otro)
plt.show()


