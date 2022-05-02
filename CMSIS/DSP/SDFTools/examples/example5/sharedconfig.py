import sys
import numpy as np 


FS=16000
NBCHANNELS=2 # stereo
# You can try with 120
AUDIO_INTERRUPT_LENGTH = 192

# MFCC Description
sample_rate = 16000
FFTSize = 1024
AudioOverlap = 256
numOfDctOutputs = 13
    
freq_min = 64
freq_high = sample_rate / 2
numOfMelFilters = 20

# NB MFCC Outputs to cover one second of signal with the window overlap
nbMFCCOutputs = int(np.floor((sample_rate - FFTSize) / AudioOverlap))

if nbMFCCOutputs%2 == 1:
    nbMFCCOutputs = nbMFCCOutputs + 1 
    
