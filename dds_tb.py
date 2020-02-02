import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *
import math
from decimal import *

NO_BITS_DAC = 14
NO_CORDIC_IT = 12 # ?
NO_SAMPLES = 4096
SAMPLING_FREQ = 100e6 # Hz
UPSAMPLE = 60

class quantizator():
    def __init__(self, nbits, offset = 0.5):
        self.nbits = nbits
        self.offset = offset
        self.levels = -2**(nbits-1)+1.0*np.arange(0,2**nbits+1,1)+offset
        self.levels[-1] = np.inf
        self.delta = 2.0/2**(nbits)
        self.missingCodes = np.zeros(2**nbits+1)

    def stepErrors(self, maxErr):
        self.levels += (np.random.uniform(-maxErr, maxErr, (1,2**self.nbits+1)))[0]

    def quantizeNormalized(self, data):
        level = np.argmax(self.levels > data*2**(self.nbits-1))
        level += self.missingCodes[level]
        level += -2**(self.nbits-1)
        return self.delta*level

    def quantizeInteger(self, data):
        level = np.argmax(self.levels > data*2**(self.nbits-1))
        level += self.missingCodes[level]        
        level += -2**(self.nbits-1)
        return level

    def __call__(self, data, normalize = True):
        if normalize:
            vquant = np.vectorize(self.quantizeNormalized)
        else:
            vquant = np.vectorize(self.quantizeInteger)
        return vquant(data)


def DAC_nrz(samples, upsample):
    quant = quantizator(NO_BITS_DAC)
    # quant.stepErrors(0.12)
    new_samples = np.array([samples] * upsample).transpose().flatten()
    dither = 100 * np.random.uniform(0,quant.delta**2/12,NO_SAMPLES*UPSAMPLE)
    new_samples = quant(new_samples+dither)
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples)*upsample))
    return (new_samples, time)

def FFT_an(val):
    spec = np.fft.fft(val)
    freqs = np.fft.fftfreq(len(val),1.0/SAMPLING_FREQ/UPSAMPLE)
    spec = spec[freqs >= 0]
    spec[spec == 0] = 1e-18
    freqs = freqs[freqs >= 0]
    spec_db = 20 * np.log10(np.abs(spec) / len(val) * 2)
    plt.plot(freqs/1e6, spec_db)
    plt.xlim(0,250)
    plt.ylim(-75,5)
    plt.show()
    return freqs, spec

Nbits = 8
Npoints = 4096
M = 128
dM = 15
T = 1

quant = quantizator(Nbits)
quant.stepErrors(0.05)

A = 0.5 * quant.delta
Ampl = 1

t = T*np.arange(0,Npoints,1)

y3 = Ampl*np.sin(2*np.pi*t*M/Npoints)


y1 = A*np.sin(2*np.pi*t*M/Npoints)
y2 = A*np.sin(2*np.pi*t*(M+dM)/Npoints)

y3=y1+y2
plt.plot(y3)
plt.show()


analog, time = DAC_nrz(y3, UPSAMPLE)
# analog=quant(y3)

Y = 20 * np.log10(1e-25+abs(np.fft.fft(analog)))-20*np.log10(Npoints)
plt.plot(Y[0:int(Npoints/2)])
fftNoiseFloor = 6.02*Nbits+1.76+10*np.log10(Npoints/2)
plt.plot([0,int(Npoints/2)],[-fftNoiseFloor, -fftNoiseFloor])
plt.show()

plt.plot(time, analog)
plt.title("Izlaz DAC vreme")
plt.show()

fft_freqs, fft_vals = FFT_an(analog)


