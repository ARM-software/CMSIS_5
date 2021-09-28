import numpy as np
import cmsisdsp.datatype as dt

def frequencyToMelSpace(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)

def melSpaceToFrequency(mels):
    return 700.0 * (np.exp(mels / 1127.0) - 1.0)

def melFilterMatrix(dtype,fmin, fmax, numOfMelFilters,fs,FFTSize):

    filters = np.zeros((numOfMelFilters,int(FFTSize/2+1)))
    zeros = np.zeros(int(FFTSize // 2 ))


    fmin_mel = frequencyToMelSpace(fmin)
    fmax_mel = frequencyToMelSpace(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=numOfMelFilters+2)


    linearfreqs = np.linspace( 0, fs/2.0, int(FFTSize // 2 + 1) )
    spectrogrammels = frequencyToMelSpace(linearfreqs)[1:]


    filtPos=[]
    filtLen=[]
    totalLen = 0
    packedFilters = []
    for n in range(numOfMelFilters):

      
      upper = (spectrogrammels - mels[n])/(mels[n+1]-mels[n]) 
      lower = (mels[n+2] - spectrogrammels)/(mels[n+2]-mels[n+1])


      filters[n, :] = np.hstack([0,np.maximum(zeros,np.minimum(upper,lower))])
      nb = 0 
      startFound = False
      for sample in filters[n, :]:
        if not startFound and sample != 0.0:
            startFound = True 
            startPos = nb

        if startFound and sample == 0.0:
           endPos = nb - 1 
           break
        nb = nb + 1 
      filtLen.append(endPos - startPos+1)
      totalLen += endPos - startPos + 1
      filtPos.append(startPos)
      packedFilters += list(filters[n, startPos:endPos+1])

    return filtLen,filtPos,dt.convert(packedFilters,dtype)


def dctMatrix(dtype,numOfDctOutputs, numOfMelFilters):
   
    result = np.zeros((numOfDctOutputs,numOfMelFilters))
    s=(np.linspace(1,numOfMelFilters,numOfMelFilters) - 0.5)/numOfMelFilters

    for i in range(0, numOfDctOutputs):
        result[i,:]=np.cos(i * np.pi*s) * np.sqrt(2.0/numOfMelFilters)
        
    return dt.convert(result.reshape(numOfDctOutputs*numOfMelFilters),dtype)


