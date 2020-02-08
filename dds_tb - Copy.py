
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *

# from decimal import *

import FixedPoint as fp

NO_BITS_DAC = 14
NO_CORDIC_IT = 12 # ?
NO_BITS_CORDIC = 40     # M = W 
NO_SAMPLES = 4096
SAMPLING_FREQ = 100e6 # Hz
UPSAMPLE = 60



def float_to_precision(number, no_bits_int, no_bits_fraction):
    PREC = fp.FXfamily(no_bits_int, no_bits_fraction)
    return float(PREC(number))


# ================================================================================================================================================

# D Grujic quanizator ADC class

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

# ================================================================================================================================================

# Grujic CORDIC functions

def CORDIC_iteration(x, y, z, i, mode, scaled=False):
    """
    Calculate one iteration of CORDIC algorithm.
    x, y, z - inputs
    i - iteration number
    mode - "rotation" for CORDIC in rotation mode,
            otherwise calculates in vector mode
    scaled - True or False. If False, scaling factor is 1.
    Returns (x, y, z)
    """
    sgn = lambda a: (a>=0) - (a<0)
    if mode=="rotation":
        sigma = sgn(z)
    else:
        sigma = -sgn(y)
    xnew = x - sigma*y*2**(-i)
    ynew = y + sigma*x*2**(-i)
    z = z - sigma * math.atan(2**(-i))
    if scaled:
        ki = CORDIC_Ki(i)
        xnew *= ki
        ynew *= ki
    return (xnew, ynew, z)

def CORDIC_Ki(i):
    """
    Calculate the scaling factor for i-th iteration
    """
    k = 1.0/math.sqrt(1.0+2.0**(-2*i))
    return k

def CORDIC_Kn(n):
    """
    Calculate the overall scaling factor for n iterations
    """
    k = 1
    for i in range(0,n):
        k *= CORDIC_Ki(i)
    return k

# ================================================================================================================================================

# CORDIC copied

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

# ================================================================================================================================================

# generate phase

def generate_phase(f0):
    phi0 = np.zeros(NO_SAMPLES)
    step = f0 / SAMPLING_FREQ
    # float_to_precision
    noise = np.random.uniform(0, 0.000005,NO_SAMPLES)
    phi0 += noise

    return (np.cumsum([step] * NO_SAMPLES + phi0)%1)

# ================================================================================================================================================

# FFT functions with output integrated

def FFT_digital(val):
    spec = np.fft.fft(val)
    freqs = np.fft.fftfreq(len(val),1.0/SAMPLING_FREQ)
    spec = spec[freqs >= 0]
    spec[spec == 0] = 1e-18
    freqs = freqs[freqs >= 0]
    spec_db = 20 * np.log10(np.abs(spec) / len(val) * 2)
    return freqs, spec

def FFT_analog(val):
    spec = np.fft.fft(val)
    freqs = np.fft.fftfreq(len(val),1.0/SAMPLING_FREQ/UPSAMPLE)
    spec = spec[freqs >= 0]
    spec[spec == 0] = 1e-18
    freqs = freqs[freqs >= 0]
    # spec_db = 20 * np.log10(np.abs(spec) / len(val) * 2)
    return freqs, spec

# ================================================================================================================================================

# DAC functions

def DAC_nrz(samples, upsample):
    quant = quantizator(NO_BITS_DAC)
    # quant.stepErrors(0.12)
    new_samples = np.array([samples] * upsample).transpose().flatten()
    dither = 0 * np.random.uniform(0,quant.delta**2/12,NO_SAMPLES*UPSAMPLE)
    new_samples = quant(new_samples+dither)
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples)*upsample))
    return (new_samples, time)

def DAC_brz(samples, upsample):
    quant = quantizator(NO_BITS_DAC)
    # quant.stepErrors(0.12)

    upsample1 = int(upsample/4)
    upsample2 = int(upsample/2)
    upsample3 = upsample-upsample2
    upsample2 -=upsample1
    new_samples = np.array([samples]*upsample1 + [-samples]*upsample2 + [0*samples]*upsample3)
    new_samples = new_samples.transpose().flatten()
    dither = 0 * np.random.uniform(0,quant.delta**2/12,NO_SAMPLES*UPSAMPLE)
    new_samples = quant(new_samples+dither)
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples) * upsample))
    return (new_samples, time)

# ================================================================================================================================================

# filter functions
# change parameters

def LP_cheby_2nd():
    Wn = 40e6
    Wa = 59e6
    b, a = cheby2(11, 60, Wn = Wa, analog=True)
    w, H = freqs(b,a, worN = np.linspace(0,100e6,100))
    # plt.plot(w,20*np.log10(np.abs(H)))
    # plt.show()
    return b,a

def BP_cheby_2nd():
    Wa1 = 91e6
    Wa2 = 159e6
    b, a = cheby2(8, 60, Wn = (Wa1, Wa2), btype="bandpass", analog = True)
    w, H = freqs(b, a, worN = np.linspace(60e+6,200e+6,300))
    # plt.plot(w,20*np.log10(np.abs(H)))
    # plt.show()
    return b,a

   
# ================================================================================================================================================

# sin(x)/x correction FIR filter

fmax = 0.4              # max. frequency of interest
n = 7                   # fitting dots
# amplitude precision of +/- 0.05 dB
f = fmax * np.arange(0, n+1, 1)/n

FIR_FILTER_ORDER = 7

H_sinc = np.sin(np.pi*f)/(np.pi*f+1e-15)
H_sinc[0] = 1.0

f_goal = np.zeros(2*(n+1))
f_goal[0::2] = f
f_goal[1::2] = f+1e-3
f_goal *= 2             # In firls function Nyquist frequency is 1

goal = np.zeros(2*(n+1))
goal[0::2] = 1.0/H_sinc
goal[1::2] = 1.0/H_sinc


h_invsinc = firls(FIR_FILTER_ORDER, f_goal, goal)                      
w,H = freqz(tuple(h_invsinc),tuple([1]), worN = NO_SAMPLES)

H_sinc = np.sin(w/2)/(w/2+1e-15)
H_sinc[0] = 1.0

H_times_Hsinc =  20*np.log10(np.absolute(H_sinc * H))

# plt.plot(w/(2*np.pi), H_times_Hsinc)
# plt.xlim(0,0.4)
# plt.ylim(-0.1,0.1)

# plt.show()



# test function

signal_freq = 10e6
generated_phase = generate_phase(signal_freq) *360        # in degrees
# plt.plot(generated_phase)
# plt.show()

cordic_sine = np.array([])
for i in generated_phase:
    cordic_sine = np.append(cordic_sine, [CORDIC(i)])

plt.plot(cordic_sine)
plt.show()

my_freqs, my_spec = FFT_digital(cordic_sine)
my_spec_db = 20 * np.log10(np.abs(my_spec) / len(cordic_sine) * 2)

plt.plot(my_freqs/SAMPLING_FREQ, my_spec_db)
plt.show()


cordic_sine_through_FIR = np.convolve(cordic_sine,h_invsinc,mode='same')
analog, time = DAC_nrz(cordic_sine_through_FIR, UPSAMPLE)


# plt.plot(time, analog)
# plt.title("Izlaz DAC vreme")
# plt.show()


fft_freqs, fft_vals = FFT_analog(analog)
b,a = LP_cheby_2nd()
w1,H1 = freqs(b, a, worN = fft_freqs)
filtered = fft_vals*H1
# plt.plot(fft_freqs/1e6,20*np.log10(np.abs(filtered)/len(filtered)))
# plt.xlim(0,250)
# plt.ylim(-200, 5)
# plt.show()
# plt.plot(np.fft.ifft(filtered))
# plt.show()


# 3rd zone
# bipolar return to zero

analog, time = DAC_brz(cordic_sine, UPSAMPLE)
# plt.plot(time, analog)
# plt.title("DAC_brz")
# plt.show()

fft_freqs, fft_mag = FFT_analog(analog)
b, a = BP_cheby_2nd()
w1,H1 = freqs(b,a,worN = fft_freqs)
filtered = fft_mag*H1
# plt.plot(fft_freqs/1e6,20*np.log10(np.abs(filtered)/len(filtered)))
# plt.xlim(0,250)
# plt.ylim(-200, 5)
# plt.title("filtrirani fft")
# plt.show()
# plt.plot(np.fft.ifft(filtered))
# plt.title("filtrirani trans")
# plt.show()

