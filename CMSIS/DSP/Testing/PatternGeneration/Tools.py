import os.path
import struct
import numpy as np

def normalize(a):
  return(a/np.max(np.abs(a)))

TAILONLY = 1
BODYONLY = 2
BODYANDTAIL = 3

# Datatype formats
F64 = 64 
F32 = 0
F16 = 16
Q31 = 31 
Q15 = 15
Q7 = 7

def loopnb(format,loopkind):
    nb = 0
    if loopkind == TAILONLY:
        if format == 64:
            nb = 1 
        if format == 0 or format == 31:
            nb = 3 
        if format == 15 or format == 16:
            nb = 7
        if format == 7:
            nb = 15
    if loopkind == BODYONLY:
        if format == 64:
            nb = 4 
        if format == 0 or format == 31:
            nb = 8 
        if format == 15 or format == 16:
            nb = 16
        if format == 7:
            nb = 32
    if loopkind == BODYANDTAIL:
        if format == 64:
            nb = 5
        if format == 0 or format == 31:
            nb = 11 # 9
        if format == 15 or format == 16:
            nb = 23 # 17
        if format == 7:
            nb = 47 # 33

    return(nb)

# Tools to generate pattern files

def createMissingDir(destPath):
  theDir=os.path.normpath(destPath)
  if not os.path.exists(theDir):
      os.makedirs(theDir)

# Pack an array of boolean into uint32
def packset(a):
    b = np.packbits(a)
    newSize = int(np.ceil(b.shape[0] / 4.0)) * 4
    c = np.copy(b)
    c.resize(newSize)
    #print(c)
    vecSize = round(newSize/4)
    c=c.reshape(vecSize,4)
    #print(c)
    r = np.zeros(vecSize)
    result = []
    for i in range(0,vecSize):
        #print(c[i,:])
        #print("%X %X %X %X" % (c[i,0],c[i,1],c[i,2],c[i,3]))
        d = (c[i,0] << 24) | (c[i,1] << 16) | (c[i,2] << 8) | c[i,3] 
        result.append(np.uint32(d))
    return(result) 

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

def float16_to_hex(f):
    """ Convert and x86 float to an ARM unsigned long int.
  
    Args:
      f (float): value to be converted
    Raises:
      Nothing 
    Returns:
      str : representation of the hex value
    """
    return hex(struct.unpack('<H', struct.pack('<e', f))[0])

def float64_to_hex(f):
    """ Convert and x86 float to an ARM unsigned long int.
  
    Args:
      f (float): value to be converted
    Raises:
      Nothing 
    Returns:
      str : representation of the hex value
    """
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def to_q63(v):
    r = int(round(v * 2**63))
    if (r > 0x07FFFFFFFFFFFFFFF):
      r = 0x07FFFFFFFFFFFFFFF
    if (r < -0x08000000000000000):
      r = -0x08000000000000000
    return ("0x%s" % format(struct.unpack('<Q', struct.pack('<q', r))[0],'016X'))

def to_q31(v):
    r = int(round(v * 2**31))
    if (r > 0x07FFFFFFF):
      r = 0x07FFFFFFF
    if (r < -0x080000000):
      r = -0x080000000
    return ("0x%s" % format(struct.unpack('<I', struct.pack('<i', r))[0],'08X'))

def to_q15(v):
    r = int(round(v * 2**15))
    if (r > 0x07FFF):
      r = 0x07FFF
    if (r < -0x08000):
      r = -0x08000
    return ("0x%s" % format(struct.unpack('<H', struct.pack('<h', r))[0],'04X'))

def to_q7(v):
    r = int(round(v * 2**7))
    if (r > 0x07F):
      r = 0x07F
    if (r < -0x080):
      r = -0x080
    return ("0x%s" % format(struct.unpack('<B', struct.pack('<b', r))[0],'02X'))

def s8(r):
  return ("0x%s" % format(struct.unpack('<B', struct.pack('<b', r))[0],'02X'))

def s16(r):
  return ("0x%s" % format(struct.unpack('<H', struct.pack('<h', r))[0],'04X'))

def s32(r):
  return ("0x%s" % format(struct.unpack('<I', struct.pack('<i', r))[0],'08X'))

def u32(r):
  return ("0x%s" % format(struct.unpack('<I', struct.pack('<I', r))[0],'08X'))

class Config:
    def __init__(self,patternDir,paramDir,ext):
      self._patternDir = "%s%s" % (patternDir,ext.upper())
      self._paramDir = "%s%s" % (paramDir,ext.upper())
      self._ext = ext 
      self._overwrite=True

      createMissingDir(self._patternDir)
      createMissingDir(self._paramDir)

    def setOverwrite(self,v):
        self._overwrite=v

    def canOverwrite(self,path):
        return(self._overwrite or not os.path.exists(path))

    def inputP(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,self._ext)))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,self._ext)))

    def inputS32P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"s32")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"s32")))

    def inputS16P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"s16")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"s16")))

    def inputS8P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"s8")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"s8")))

    def inputF32P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"f32")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"f32")))

    def inputF16P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"f16")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"f16")))

    def inputQ31P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"q31")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"q31")))

    def inputQ15P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"q15")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"q15")))

    def inputQ7P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"q7")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"q7")))

    def inputU32P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"u32")))
        else:
          return(os.path.join(self._patternDir,"Input%d_%s.txt" % (i,"u32")))

    def refP(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,self._ext)))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,self._ext)))

    def refS8P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"s8")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"s8")))

    def refS16P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"s16")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"s16")))

    def refS32P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"s32")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"s32")))

    def refQ63P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"q63")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"q63")))

    def refQ31P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"q31")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"q31")))

    def refF32P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"f32")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"f32")))

    def refF16P(self,i,name=None):
        """ Path to a reference pattern from the ID
      
        Args:
          i (int): ID to the reference pattern
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._patternDir,"%s%d_%s.txt" % (name,i,"f16")))
        else:
          return(os.path.join(self._patternDir,"Reference%d_%s.txt" % (i,"f16")))

    def paramP(self,i,name=None):
        """ Path to a parameters from the ID
      
        Args:
          i (int): ID to the params
        Raises:
          Nothing 
        Returns:
          str : path to the file where to generate the pattern data
        """
        if name:
          return(os.path.join(self._paramDir,"%s%d.txt" % (name,i)))
        else:
          return(os.path.join(self._paramDir,"Params%d.txt" % i))

    def _writeVectorF64(self,i,data):
        """ Write pattern data
        
        The format is recognized by the text framework script.
        First line is the sample width (B,H or W,D for 8,16,32 or 64 bits)
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("D\n%d\n" % len(data))
              for v in data:
                  f.write("// %f\n" % v)
                  f.write("%s\n" % float64_to_hex(v))

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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("W\n%d\n" % len(data))
              for v in data:
                  f.write("// %f\n" % v)
                  f.write("%s\n" % float_to_hex(v))

    def _writeVectorF16(self,i,data):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("H\n%d\n" % len(data))
              for v in data:
                  f.write("// %f\n" % v)
                  f.write("%s\n" % float16_to_hex(v))

    def _writeVectorQ63(self,i,data):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("D\n%d\n" % len(data))
              for v in data:
                  f.write("// %f\n" % v)
                  f.write("%s\n" % to_q63(v))

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
        if self.canOverwrite(i):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("H\n%d\n" % len(data))
              for v in data:
                  f.write("// %f\n" % v)
                  f.write("%s\n" % to_q15(v))

    def _writeVectorS16(self,i,data):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("H\n%d\n" % len(data))
              for v in data:
                  f.write("// %d\n" % v)
                  f.write("%s\n" % s16(v))

    def _writeVectorS32(self,i,data):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("W\n%d\n" % len(data))
              for v in data:
                  f.write("// %d\n" % v)
                  f.write("%s\n" % s32(v))

    def _writeVectorU32(self,i,data):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("W\n%d\n" % len(data))
              for v in data:
                  f.write("// %s\n" % v)
                  f.write("%s\n" % u32(v))

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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("B\n%d\n" % len(data))
              for v in data:
                  f.write("// %f\n" % v)
                  f.write("%s\n" % to_q7(v))

    def _writeVectorS8(self,i,data):
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
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("B\n%d\n" % len(data))
              for v in data:
                  f.write("// %d\n" % v)
                  f.write("%s\n" % s8(v))

    def writeReference(self,j,data,name=None):
        if (self._ext == "f64"):
          self._writeVectorF64(self.refP(j,name),data)
        if (self._ext == "f32"):
          self._writeVectorF32(self.refP(j,name),data)
        if (self._ext == "f16"):
          self._writeVectorF16(self.refP(j,name),data)
        if (self._ext == "q63"):
          self._writeVectorQ63(self.refP(j,name),data)
        if (self._ext == "q31"):
          self._writeVectorQ31(self.refP(j,name),data)
        if (self._ext == "q15"):
          self._writeVectorQ15(self.refP(j,name),data)
        if (self._ext == "q7"):
          self._writeVectorQ7(self.refP(j,name),data)
        if (self._ext == "u32"):
          self._writeVectorU32(self.refP(j,name),data)
        if (self._ext == "s8"):
          self._writeVectorS8(self.refP(j,name),data)

    def writeReferenceQ63(self,j,data,name=None):
        self._writeVectorQ63(self.refQ63P(j,name),data)

    def writeReferenceQ31(self,j,data,name=None):
        self._writeVectorQ31(self.refQ31P(j,name),data)

    def writeReferenceS8(self,j,data,name=None):
        self._writeVectorS8(self.refS8P(j,name),data)

    def writeReferenceS16(self,j,data,name=None):
        self._writeVectorS16(self.refS16P(j,name),data)
       
    def writeReferenceS32(self,j,data,name=None):
        self._writeVectorS32(self.refS32P(j,name),data)

    def writeReferenceF32(self,j,data,name=None):
        self._writeVectorF32(self.refF32P(j,name),data)

    def writeReferenceF16(self,j,data,name=None):
        self._writeVectorF16(self.refF16P(j,name),data)

    def writeInput(self,j,data,name=None):
        if (self._ext == "f64"):
          self._writeVectorF64(self.inputP(j,name),data)
        if (self._ext == "f32"):
          self._writeVectorF32(self.inputP(j,name),data)
        if (self._ext == "f16"):
          self._writeVectorF16(self.inputP(j,name),data)
        if (self._ext == "q31"):
          self._writeVectorQ31(self.inputP(j,name),data)
        if (self._ext == "q15"):
          self._writeVectorQ15(self.inputP(j,name),data)
        if (self._ext == "q7"):
          self._writeVectorQ7(self.inputP(j,name),data)
        if (self._ext == "u32"):
          self._writeVectorU32(self.inputP(j,name),data)
        if (self._ext == "s8"):
          self._writeVectorS8(self.inputP(j,name),data)

    def writeInputF32(self,j,data,name=None):
        self._writeVectorF32(self.inputF32P(j,name),data)

    def writeInputF16(self,j,data,name=None):
        self._writeVectorF16(self.inputF16P(j,name),data)

    def writeInputQ31(self,j,data,name=None):
        self._writeVectorQ31(self.inputQ31P(j,name),data)

    def writeInputQ15(self,j,data,name=None):
        self._writeVectorQ15(self.inputQ15P(j,name),data)

    def writeInputQ7(self,j,data,name=None):
        self._writeVectorQ7(self.inputQ7P(j,name),data)

    def writeInputS32(self,j,data,name=None):
        self._writeVectorS32(self.inputS32P(j,name),data)

    def writeInputS16(self,j,data,name=None):
        self._writeVectorS16(self.inputS16P(j,name),data)

    def writeInputS8(self,j,data,name=None):
        self._writeVectorS8(self.inputS8P(j,name),data)

    def writeInputU32(self,j,data,name=None):
        self._writeVectorU32(self.inputU32P(j,name),data)

    def writeParam(self,j,data,name=None):
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
        i=self.paramP(j,name)
        if self.canOverwrite(i):
          with open(i,"w") as f:
              # Write sample dimension nb sample header
              #np.savetxt(i, data, newline="\n", header="W\n%d" % len(data),comments ="" )
              f.write("%d\n" % len(data))
              for v in data:
                  f.write("%d\n" % v)



