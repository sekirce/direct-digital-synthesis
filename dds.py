import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *
import math
from decimal import *

NO_BITS_DAC = 14
NO_CORDIC_IT = 12 # ?
NO_SAMPLES = 1000
SAMPLING_FREQ = 100e6 # Hz
UPSAMPLE = 30
# quatizator class

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

# CORDIC

def CORDIC_scaling_factor_nth_iteration(n):
    scaling_factor = 1
    for i in range(0,n):
        scaling_factor *= 1.0 / math.sqrt(1.0+2.0**(-2*i))
    return scaling_factor

# phi in degrees
def CORDIC(phi):

    quadrant = (int(phi/91)%4)
    phi1 = phi - quadrant *90
    theta =  phi1 * np.pi/180    # radians

    x =CORDIC_scaling_factor_nth_iteration(NO_CORDIC_IT)
    y = 0
    z = theta

    for i in range (0, NO_CORDIC_IT):
        sigma = (1 if z >= 0 else -1)
        xnew = x - sigma*y*2**(-i)
        y = y + sigma*x*2**(-i)
        x = xnew
        z = z - sigma * math.atan(2**(-i))
    if quadrant == 0:
        return x
    elif quadrant == 1:
        return -y
    elif quadrant == 2:
        return -x
    elif quadrant == 3:
        return y


def PhaseSamples(f0):
    phi0 = np.zeros(NO_SAMPLES)
    step = f0 / SAMPLING_FREQ
    noise = np.random.uniform(0, 0.0005,NO_SAMPLES)
    phi0 += noise
    return (np.cumsum([step] * NO_SAMPLES + phi0)%1)


def DAC_nrz(samples, upsample):
    new_samples = np.array([samples] * upsample).transpose().flatten()
    quant = quantizator(NO_BITS_DAC)
    new_samples = quant(new_samples)
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples)*upsample))
    return (new_samples, time)

# CORDIC cosine generator

def FFT_dig(val):
    spec = np.fft.fft(val)
    freqs = np.fft.fftfreq(len(val),1.0/SAMPLING_FREQ)
    spec = spec[freqs >= 0]
    spec[spec == 0] = 1e-18
    freqs = freqs[freqs >= 0]
    spec_db = 20 * np.log10(np.abs(spec) / len(val) * 2)
    plt.plot(freqs/SAMPLING_FREQ, spec_db)
    plt.title("fft")
    plt.xlim(0,0.5)
    plt.ylim(-100,5)
    plt.grid(True)
    plt.show()
    return freqs, spec





# sin(x)/x correction FIR filter

fmax = 0.4 				# max. frequency of interest
n = 7					# amplitude precision of +/- 0.05 dB
f = fmax * np.arange(0, n+1, 1)/n

H_sinc = np.sin(np.pi*f)/(np.pi*f+1e-15)
H_sinc[0] = 1.0

f_goal = np.zeros(2*(n+1))
f_goal[0::2] = f
f_goal[1::2] = f+1e-3
f_goal *= 2				# In firls function Nyquist frequency is 1

goal = np.zeros(2*(n+1))
goal[0::2] = 1.0/H_sinc
goal[1::2] = 1.0/H_sinc

h_invsinc = firls(7, f_goal, goal)						# firls FIR filter order od 6
w,H = freqz(tuple(h_invsinc),tuple([1]), worN = NO_SAMPLES)

H_sinc = np.sin(w/2)/(w/2+1e-15)
H_sinc[0] = 1.0

H_times_Hsinc = 20*np.log10(np.absolute(H_sinc * H))
plt.plot(w/(2*np.pi), H_times_Hsinc)
plt.xlim(0,0.4)
plt.ylim(-0.1,0.1)

plt.show()

plt.plot(w/(2*np.pi), H_times_Hsinc)
plt.xlim(0,0.4)
plt.ylim(-0.05,0.05)

plt.show()

plt.plot(w/(2*np.pi),np.absolute(1.0/H_sinc), 'k', aa=True, linewidth=2.5,label=r'$x/\sin(x)$')
plt.plot(w/(2*np.pi),np.absolute(H), 'gray', aa=True, linewidth=2.5,label=r'$|H(\mathrm{j} F)|$')
plt.legend(loc='upper left')
df = 0.005

for i in range(0,len(goal),2):
    plt.plot((f_goal[i]/2-df,f_goal[i+1]/2+df),[goal[i], goal[i+1]], 'r',aa=True, linewidth=2.5)

plt.xlim(0,0.5)
plt.ylim(1,1.4)
plt.grid()

plt.xlabel(r'$F$')
plt.ylabel(r'$|H(\mathrm{j} F)|$')

plt.tight_layout()
plt.show()


freq = 1.1354e6
angles = PhaseSamples(freq) *360
plt.plot(angles)
plt.show()

val = np.array([])
for i in angles:
    val = np.append(val, [CORDIC(i)])





fft_freqs = np.fft.fftfreq(len(val), 1.0/SAMPLING_FREQ)
FFT_dig(val)

val = np.convolve(val,h_invsinc,mode='same')
analog, time = DAC_nrz(val, UPSAMPLE)



plt.plot(time, analog)
plt.show()

FFT_dig(analog)

