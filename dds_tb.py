# authors:
# Arsovic Aleksandar
# Vukovic Aleksandar

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *
from matplotlib import rc
rc('text', usetex=True)

# https://github.com/rwpenney/spfpm/blob/master/FixedPoint.py
import FixedPoint as fp

NO_BITS_DAC = 14 
NO_CORDIC_IT = 12
NO_BITS_PHASE_ACC = 40     # M = W 
NO_SAMPLES = 4096
SAMPLING_FREQ = 100e6 # Hz
UPSAMPLE = 40
FIR_FILTER_ORDER = 7

# fixed float point function

def float_to_precision(number, no_bits_fraction, no_bits_int):
    PREC = fp.FXfamily(no_bits_fraction, no_bits_int )
    return float(PREC(number))

# ================================================================================================================================================

# D Grujic quanizator ADC class

class quantizator():
    def __init__(self, nbits, offset = 0):
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

# CORDIC copied

def CORDIC_scaling_factor_nth_iteration(n):
    scaling_factor = 1
    for i in range(0,n):
        scaling_factor *= 1.0 / math.sqrt(1.0+2.0**(-2*i))
    return scaling_factor

# phi in degrees
def CORDIC(phi):
    quadrant = int((phi+math.pi)/(math.pi/2))%4
    theta = phi + math.pi - quadrant * math.pi/2
    x =CORDIC_scaling_factor_nth_iteration(NO_CORDIC_IT)
    y = 0
    z = theta

    for i in range (0, NO_CORDIC_IT):
        sigma = (1 if z >= 0 else -1)
        xnew = x - sigma*y*2**(-i)
        y = y + sigma*x*2**(-i)
        x = xnew
        z = z - sigma * math.atan(2**(-i))
    if quadrant == 2:
        return x
    elif quadrant == 3:
        return -y
    elif quadrant == 0:
        return -x
    elif quadrant == 1:
        return y

# ================================================================================================================================================

# generate phase

def generate_phase(f0):
    phi0 = np.zeros(NO_SAMPLES)
    step = float_to_precision(f0 / SAMPLING_FREQ *2 , NO_BITS_PHASE_ACC - 3,3) 
    noise = np.random.normal(0, 0.00005,NO_SAMPLES) 
    # Dithering noise in phase acc is off
    #phi0 += noise
    array_precision = []
    for i in (np.cumsum([step] * NO_SAMPLES + phi0)%2-1)*math.pi :
        array_precision.append(float_to_precision(i, NO_BITS_PHASE_ACC - 3,3))
    return array_precision

# ================================================================================================================================================

# FFT functions with output integrated

def FFT(val):   # not upsampled
    spec = np.fft.fft(val)
    freqs = np.fft.fftfreq(len(val),1.0/SAMPLING_FREQ)
    spec = spec[freqs >= 0]
    spec[spec == 0] = 1e-18
    freqs = freqs[freqs >= 0]
    spec_db = 20 * np.log10(np.abs(spec) / len(val) * 2)
    return freqs, spec

def FFT_upsampled(val):    # upsampled
    spec = np.fft.fft(val)
    freqs = np.fft.fftfreq(len(val),1.0/SAMPLING_FREQ/UPSAMPLE)
    spec = spec[freqs >= 0]
    spec[spec == 0] = 1e-18
    freqs = freqs[freqs >= 0]
    spec_db = 20 * np.log10(np.abs(spec) / len(val) * 2)
    return freqs, spec

# ================================================================================================================================================

# DAC functions quantizator

def DAC_nrz(samples, upsample):
    quant = quantizator(NO_BITS_DAC)
    new_samples = np.array([samples] * upsample).transpose().flatten()
    dither = 1 * np.random.normal(0,0.05,NO_SAMPLES*UPSAMPLE)
    #new_samples = quant(new_samples+dither)
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples)*upsample))
    return (new_samples, time)

def DAC_brz(samples, upsample):
    quant = quantizator(NO_BITS_DAC)

    upsample1 = int(upsample/4)
    upsample2 = int(upsample/2)
    upsample3 = upsample-upsample2
    upsample2 -=upsample1
    new_samples = np.array([samples]*upsample1 + [-samples]*upsample2 +
                           [0*samples]*upsample3)
    new_samples = new_samples.transpose().flatten()
    dither = 0 * np.random.normal(0,0.05,NO_SAMPLES*UPSAMPLE)
    new_samples = quant(new_samples+dither)
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples) * upsample))
    return (new_samples, time)

# ================================================================================================================================================

# DAC functions quantizator

def DAC_nrz_fp(samples, upsample):
    
    samples_transposed = np.array([samples] * upsample).transpose().flatten()
    dither = 0 * np.random.normal(0,0.15,NO_SAMPLES*UPSAMPLE)
    samples_fp = []
    for i in range(0,len(samples_transposed)):
        samples_fp.append(float_to_precision(samples_transposed[i]+dither[i], NO_BITS_DAC-2,3))
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples)*upsample))
    return (samples_fp, time)

def DAC_brz_fp(samples, upsample):
    
    upsample1 = int(upsample/4)
    upsample2 = int(upsample/2)
    upsample3 = upsample-upsample2
    upsample2 -=upsample1
    new_samples = np.array([samples]*upsample1 + [-samples]*upsample2 +
                           [0*samples]*upsample3)
    samples_transposed = new_samples.transpose().flatten()
    dither = 0 * np.random.normal(0,0.15,NO_SAMPLES*UPSAMPLE)
    samples_fp = []
    for i in range(0,len(samples_transposed)):
        samples_fp.append(float_to_precision(samples_transposed[i]+dither[i], NO_BITS_DAC-2,3))
    time = np.cumsum([1.0/SAMPLING_FREQ/upsample] * (len(samples) * upsample))
    return (samples_fp, time)

# ================================================================================================================================================

# filter functions
# change parameters

def LP_cheby_2nd():
    Wn = 40e6
    Wa = 59e6
    b, a = cheby2(11, 60, Wn = Wa, analog=True)
    w, H = freqs(b,a, worN = np.linspace(0,100e6,100))
    return b,a

def BP_cheby_2nd():
    Wa1 = 91e6
    Wa2 = 159e6
    b, a = cheby2(8, 60, Wn = (Wa1, Wa2), btype="bandpass", analog = True)
    w, H = freqs(b, a, worN = np.linspace(60e+6,200e+6,300))
    return b,a
   
# ================================================================================================================================================

# sin(x)/x correction FIR filter

fmax = 0.4              # max. frequency of interest
n = 7                   # fitting dots
# amplitude precision of +/- 0.05 dB
f = fmax * np.arange(0, n+1, 1)/n

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

# ================================================================================================================================================
plt.plot(w/(2*np.pi),np.absolute(1.0/H_sinc), 'k', aa=True, linewidth=2.5,label=r'$x/\sin(x)$')
plt.plot(w/(2*np.pi),np.absolute(H), 'gray', aa=True, linewidth=2.5,label=r'$|H(\mathrm{j} F)|$')
df = 0.005
for i in range(0,len(goal),2):
    plt.plot((f_goal[i]/2-df,f_goal[i+1]/2+df),[goal[i], goal[i+1]], 'r',aa=True, linewidth=2.5)
#plt.title("Characteristics of sinc(x) and correction")
plt.xlabel("$F=f/f_s$" )
plt.ylabel("$H$ [dB]")
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('slike/sinx_korekcija.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

H_times_Hsinc =  20*np.log10(np.absolute(H_sinc * H))

# ================================================================================================================================================
plt.plot(w/(2*np.pi), H_times_Hsinc)
plt.xlim(0,0.4)
plt.ylim(-0.1,0.1)
plt.xlabel("$F=f/f_s$" )
plt.ylabel("$H$ [dB]")
#plt.title("Part of FIR filter characteristic of significance")
plt.plot([0,w[-1]], [0.05, 0.05], 'r', label=r'limits')
plt.plot([0,w[-1]], [-0.05, -0.05], 'r')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('slike/varijacija_amplitude_fir.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# main function

signal_freq = 10e6
generated_phase = generate_phase(signal_freq)

# ================================================================================================================================================
plt.plot(generated_phase)
#plt.title("Output of phase accumulator")
plt.xlabel("$t$ [s]")
plt.ylabel("$\\varphi$ [rad]")
plt.xlim(0,100)
plt.grid(True)
plt.savefig('slike/fazni_akumulator_izlaz.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

cordic_sine = np.array([])
for i in generated_phase:
    cordic_sine = np.append(cordic_sine, [CORDIC(i)])

# ================================================================================================================================================
plt.plot(cordic_sine)
#plt.title("Sine generated by CORDIC algorithm")
#plt.ylabel("");
plt.xlabel("$t$ [s]")
plt.xlim(0,100)
plt.grid(True)
plt.savefig('slike/cordic_sample.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

my_freqs, my_spec = FFT(cordic_sine)
my_spec_db = 20 * np.log10(np.abs(my_spec) / len(cordic_sine) * 2)

# ================================================================================================================================================
plt.plot(my_freqs/SAMPLING_FREQ, my_spec_db, label=b'spectrum')
plt.ylim(-80,10)
#plt.title("Spectrum of CORDIC sine")
plt.ylabel("$H$ [dB]")
plt.xlabel("$F=f/f_s$" )
plt.plot([0,my_freqs[-1]/SAMPLING_FREQ], [-35, -35], 'r',label=r'referent')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('slike/cordic_fft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

cordic_sine_through_FIR = np.convolve(cordic_sine,h_invsinc,mode='same')
analog, time = DAC_nrz_fp(cordic_sine_through_FIR/1.2, UPSAMPLE)

# ================================================================================================================================================
plt.plot(time*1e6, analog)
#plt.title("Output of DA convertor nrz")
plt.xlim(0,1)
plt.ylabel("$H$")
plt.xlabel("$t$ [us]")
plt.savefig('slike/output_da_nrz.pdf')
plt.grid(True)
plt.savefig('slike/da_sample_no_filter.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

fft_freqs, fft_vals = FFT_upsampled(analog)
b,a = LP_cheby_2nd()
w_lp,H_lp = freqs(b, a, worN = fft_freqs)

# ================================================================================================================================================
plt.plot(w_lp/1e6,20*np.log10(np.abs(H_lp)))
#plt.title("Transfer characteristic of LP filter")
plt.ylabel("$H$ [dB]")
plt.xlabel("$f$ [MHz]")
plt.axvline(x=40, color='b', linestyle='--')
plt.axvline(x=60, color='r', linestyle='--')
plt.xlim(0,300)
plt.grid(True)
plt.savefig('slike/lp_fft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

filtered = fft_vals*H_lp
t = np.arange(0, 1000*NO_SAMPLES/SAMPLING_FREQ, 1000*2/SAMPLING_FREQ/UPSAMPLE)

# ================================================================================================================================================
plt.plot(fft_freqs/1e6,20*np.log10(np.abs(filtered)/len(filtered)), label=b'spectrum')
plt.xlim(0,250)
plt.ylim(-200, 5)
#plt.title("Spectrum of LP filtered output of DA convertor nrz")
plt.axvline(x=50, color='k', linestyle='--')
plt.plot([0,fft_freqs[-1]], [-60, -60], 'r', label=r'suppression lvl needed')
plt.ylabel("$H$ [dB]")
plt.xlabel("$f$ [MHz]")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('slike/da_fft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================


# ================================================================================================================================================
plt.plot(t, np.fft.ifft(filtered))
plt.xlim(0.0005,0.001)
#plt.title("Filtered output of DA convertor nrz")
plt.ylabel("$v$")
plt.xlabel("$t$ [us] ")
plt.grid(True)
plt.savefig('slike/da_ifft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================


# 3rd zone
# bipolar return to zero

analog, time = DAC_brz_fp(cordic_sine, UPSAMPLE)

# ================================================================================================================================================
plt.plot(time*1e6, analog)
#plt.title("Output of raw DA convertor brz")
plt.ylabel("$H$")
plt.xlim(0,1)
plt.xlabel("$t$ [s]")
plt.grid(True)
plt.savefig('slike/da_brz_sample.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

fft_freqs, fft_mag = FFT_upsampled(analog)
b, a = BP_cheby_2nd()
w_bp,H_bp = freqs(b,a,worN = fft_freqs)

# ================================================================================================================================================
plt.plot(w_bp/1e6,20*np.log10(np.abs(H_bp)))
#plt.title("Transfer characteristic of BP filter")
plt.ylabel("$H$ [dB]")
plt.xlabel("$f$ [MHz]")
plt.xlim(0,300)
plt.axvline(x=90, color='r', linestyle='--')
plt.axvline(x=100, color='b', linestyle='--')
plt.axvline(x=140, color='b', linestyle='--')
plt.axvline(x=160, color='r', linestyle='--')
plt.grid(True)
plt.savefig('slike/bp_fft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

filtered = fft_mag*H_bp

# ================================================================================================================================================
plt.plot(fft_freqs/1e6,20*np.log10(np.abs(filtered)/len(filtered)),label=b'spectrum')
plt.xlim(0,250)
plt.ylim(-200, 5)
#plt.title("Spectrum of BP filtered DA convertor brz")
plt.ylabel("$H$ [dB]")
plt.xlabel("$f$ [MHz]")
plt.axvline(x=100, color='k', linestyle='--')
plt.axvline(x=150, color='k', linestyle='--')
plt.plot([0,fft_freqs[-1]], [-60, -60], 'r', label=r'suppression lvl needed')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('slike/da_brz_fft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

# ================================================================================================================================================
plt.plot(np.fft.ifft(filtered))
#plt.title("Filtered DA convertor brz")
plt.ylabel("$v$")
plt.xlabel("$t$ [us]")
plt.xlim(0,1000)
plt.grid(True)
plt.savefig('slike/da_brz_ifft.pdf',bbox_inches='tight')
plt.show()
# ================================================================================================================================================

