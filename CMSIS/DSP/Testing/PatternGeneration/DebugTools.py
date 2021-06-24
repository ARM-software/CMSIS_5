import struct
import numpy as np
# Tools to read the generated pattern files and the hex output files

def readQ31Pattern(r):
    l = []
    with open(r, 'r') as f:
       f.readline()
       nb = int(f.readline())
       for i in range(nb):
          f.readline()
          # Read the hex and interpret the sign
          r=int(f.readline(),16)
          r=struct.unpack('<i', struct.pack('<I',r))[0]
          l.append(r)
    l = (1.0*np.array(l)) / 2**31
    #print(l)
    return(l)

def readQ15Pattern(r):
    l = []
    with open(r, 'r') as f:
       f.readline()
       nb = int(f.readline())
       for i in range(nb):
          f.readline()
          # Read the hex and interpret the sign
          r=int(f.readline(),16)
          r=struct.unpack('<h', struct.pack('<H',r))[0]
          l.append(r)
    l = (1.0*np.array(l)) / 2**15
    #print(l)
    return(l)

def hex2float(h):
    return(struct.unpack('<f', struct.pack('<I', int(h,16)))[0])

def hex2f16(h):
    return(struct.unpack('<e', struct.pack('<H', int(h,16)))[0])


def readF32Pattern(r):
    l = []
    with open(r, 'r') as f:
       f.readline()
       nb = int(f.readline())
       for i in range(nb):
          f.readline()
          l.append(hex2float(f.readline()))
    l = (1.0*np.array(l))
    #print(l)
    return(l)

def readF16Pattern(r):
    l = []
    with open(r, 'r') as f:
       f.readline()
       nb = int(f.readline())
       for i in range(nb):
          f.readline()
          l.append(hex2f16(f.readline()))
    l = (1.0*np.array(l))
    #print(l)
    return(l)

# Read the hex and interpret the sign
def hexToQ31(s):
   r = int(s,0)
   r=struct.unpack('<i', struct.pack('<I',r))[0]
   return(r)

# Read the hex and interpret the sign
def hexToQ15(s):
   r = int(s,0)
   r=struct.unpack('<i', struct.pack('<I',r))[0]
   return(r)

def readQ31Output(path):
   sig = np.loadtxt(path, delimiter=',',dtype="int64",converters= {0:hexToQ31})
   sig = 1.0 * sig / 2**31
   return(sig)

def readQ15Output(path):
   sig = np.loadtxt(path, delimiter=',',dtype="int64",converters= {0:hexToQ15})
   sig = 1.0 * sig / 2**15
   return(sig)


def readF32Output(path):
   sig = np.loadtxt(path, delimiter=',',dtype="float",converters= {0:hex2float})
   sig = 1.0 * sig
   return(sig)

def readF16Output(path):
   sig = np.loadtxt(path, delimiter=',',dtype="float",converters= {0:hex2f16})
   sig = 1.0 * sig
   return(sig)

def SNR(ref,sig):
  energy = np.dot(ref,np.conj(ref))
  error = np.dot(ref-sig,np.conj(ref-sig))
  snr = 10 * np.log10(energy/error)
  return(snr)