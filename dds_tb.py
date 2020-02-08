
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *

from decimal import *

NO_BITS_DAC = 14
NO_CORDIC_IT = 12 # ?
NO_SAMPLES = 4096
SAMPLING_FREQ = 100e6 # Hz
UPSAMPLE = 60

# D Grujic class

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

# D Grujic CORDIC functions

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

def print_iteration(i, x, y, z):
    """
    Print intermediate results
    """
    res = '{0: <3}'.format(str(i))
    res += "{:12.8f} {:12.8f} {:12.8f}".format(x,y,z) 
    print(res)





###############
# Input values
###############

angle = 13 * math.pi/180.0

startAngle = 18 * math.pi/180.0

scaled = False
x = math.cos(startAngle)
y = math.sin(startAngle)
z = angle
n = 12

###############
# Functions
###############