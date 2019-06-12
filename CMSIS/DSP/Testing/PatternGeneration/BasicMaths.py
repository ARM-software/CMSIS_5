import os.path
import numpy as np
import struct
import itertools


def createMissingDir(destPath):
  theDir=os.path.normpath(destPath)
  if not os.path.exists(theDir):
      os.makedirs(theDir)

def float_to_hex(f):
    """ Convert and x86 float to an ARM unsigned long int.
  
    Args:
      f (float): value to be converted
    Raises:
      Nothing 
    Returns:
      str : representation of the hex value
    """
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def to_q31(v):
    r = int(round(v * 2**31))
    if (r > 0x07FFFFFFF):
      r = 0x07FFFFFFF
    if (r < -0x080000000):
      r = -0x080000000
    return hex(struct.unpack('<I', struct.pack('<i', r))[0])

def to_q15(v):
    r = int(round(v * 2**15))
    if (r > 0x07FFF):
      r = 0x07FFF
    if (r < -0x08000):
      r = -0x08000
    return hex(struct.unpack('<H', struct.pack('<h', r))[0])

def to_q7(v):
    r = int(round(v * 2**7))
    if (r > 0x07F):
      r = 0x07F
    if (r < -0x080):
      r = -0x080
    return hex(struct.unpack('<B', struct.pack('<b', r))[0])

class Config:
    def __init__(self,patternDir,paramDir,ext):
      self._patternDir = "%s%s" % (patternDir,ext.upper())
      self._paramDir = "%s%s" % (paramDir,ext.upper())
      self._ext = ext 

      createMissingDir(self._patternDir)
      createMissingDir(self._paramDir)

    
    def inputP(self,i):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,self._ext)))

    def refP(self,i):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,self._ext)))

    def paramP(self,i):
        """ Path to a parameters from the ID
      
        Args:
          i (int): ID to the params
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        return(os.path.join(self._paramDir,"Params%d.txt" % i))

    def _writeVectorF32(self,i,data):
        """ Write pattern data
        
        The format is recognized by the text framework script.
        First line is the sample width (B,H or W for 8,16 or 32 bits)
        Second line is number of samples
        Other lines are hexadecimal representation of the samples in format
        which can be read on big endian ARM.
        
          Args:
            j (int): ID of pattern file
            data (array): Vector containing the data
          Raises:
            Nothing 
          Returns:
            Nothing
        """
        with open(i,"w") as f:
            # Write sample dimension nb sample header
            #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
            f.write("W\n%d\n" % len(data))
            for v in data:
                f.write("// %f\n" % v)
                f.write("%s\n" % float_to_hex(v))

    def _writeVectorQ31(self,i,data):
        """ Write pattern data
        
        The format is recognized by the text framework script.
        First line is the sample width (B,H or W for 8,16 or 32 bits)
        Second line is number of samples
        Other lines are hexadecimal representation of the samples in format
        which can be read on big endian ARM.
        
          Args:
            j (int): ID of pattern file
            data (array): Vector containing the data
          Raises:
            Nothing 
          Returns:
            Nothing
        """
        with open(i,"w") as f:
            # Write sample dimension nb sample header
            #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
            f.write("W\n%d\n" % len(data))
            for v in data:
                f.write("// %f\n" % v)
                f.write("%s\n" % to_q31(v))

    def _writeVectorQ15(self,i,data):
        """ Write pattern data
        
        The format is recognized by the text framework script.
        First line is the sample width (B,H or W for 8,16 or 32 bits)
        Second line is number of samples
        Other lines are hexadecimal representation of the samples in format
        which can be read on big endian ARM.
        
          Args:
            j (int): ID of pattern file
            data (array): Vector containing the data
          Raises:
            Nothing 
          Returns:
            Nothing
        """
        with open(i,"w") as f:
            # Write sample dimension nb sample header
            #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
            f.write("H\n%d\n" % len(data))
            for v in data:
                f.write("// %f\n" % v)
                f.write("%s\n" % to_q15(v))

    def _writeVectorQ7(self,i,data):
        """ Write pattern data
        
        The format is recognized by the text framework script.
        First line is the sample width (B,H or W for 8,16 or 32 bits)
        Second line is number of samples
        Other lines are hexadecimal representation of the samples in format
        which can be read on big endian ARM.
        
          Args:
            j (int): ID of pattern file
            data (array): Vector containing the data
          Raises:
            Nothing 
          Returns:
            Nothing
        """
        with open(i,"w") as f:
            # Write sample dimension nb sample header
            #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
            f.write("B\n%d\n" % len(data))
            for v in data:
                f.write("// %f\n" % v)
                f.write("%s\n" % to_q7(v))

    def writeVector(self,j,data):
        if (self._ext == "f32"):
          self._writeVectorF32(self.refP(j),data)
        if (self._ext == "q31"):
          self._writeVectorQ31(self.refP(j),data)
        if (self._ext == "q15"):
          self._writeVectorQ15(self.refP(j),data)
        if (self._ext == "q7"):
          self._writeVectorQ7(self.refP(j),data)

    def writeInput(self,j,data):
        if (self._ext == "f32"):
          self._writeVectorF32(self.inputP(j),data)
        if (self._ext == "q31"):
          self._writeVectorQ31(self.inputP(j),data)
        if (self._ext == "q15"):
          self._writeVectorQ15(self.inputP(j),data)
        if (self._ext == "q7"):
          self._writeVectorQ7(self.inputP(j),data)

    def writeParam(self,j,data):
        """ Write pattern data
        
        The format is recognized by the text framework script.
        First line is the sample width (B,H or W for 8,16 or 32 bits)
        Second line is number of samples
        Other lines are hexadecimal representation of the samples in format
        which can be read on big endian ARM.
        
          Args:
            j (int): ID of parameter file
            data (array): Vector containing the data
          Raises:
            Nothing 
          Returns:
            Nothing
        """
        i=self.paramP(j)
        with open(i,"w") as f:
            # Write sample dimension nb sample header
            #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
            f.write("%d\n" % len(data))
            for v in data:
                f.write("%d\n" % v)





def writeTests(config):
    NBSAMPLES=256

    data1=np.random.randn(NBSAMPLES)
    data2=np.random.randn(NBSAMPLES)
    data3=np.random.randn(1)
    
    data1 = data1/max(data1)
    data2 = data1/max(data2)

    config.writeInput(1, data1)
    config.writeInput(2, data2)
    
    ref = data1 + data2
    config.writeVector(1, ref)
    
    ref = data1 - data2
    config.writeVector(2, ref)
    
    ref = data1 * data2
    config.writeVector(3, ref)
    
    ref = -data1
    config.writeVector(4, ref)
    
    ref = data1 + 0.5
    config.writeVector(5, ref)
    
    ref = data1 * 0.5
    config.writeVector(6, ref)
    
    nb = 3
    ref = np.array([np.dot(data1[0:nb] ,data2[0:nb])])
    config.writeVector(7, ref)
    
    nb = 8
    ref = np.array([np.dot(data1[0:nb] ,data2[0:nb])])
    config.writeVector(8, ref)
    
    nb = 9
    ref = np.array([np.dot(data1[0:nb] ,data2[0:nb])])
    config.writeVector(9, ref)
    
    ref = abs(data1)
    config.writeVector(10, ref)


PATTERNDIR = os.path.join("Patterns","DSP","BasicMaths","BasicMaths")
PARAMDIR = os.path.join("Parameters","DSP","BasicMaths","BasicMaths")

configf32=Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Config(PATTERNDIR,PARAMDIR,"q15")
configq7=Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
writeTests(configq7)

# Params just as example
someLists=[[1,3,5],[1,3,5],[1,3,5]]

r=np.array([element for element in itertools.product(*someLists)])
configf32.writeParam(1, r.reshape(81))

