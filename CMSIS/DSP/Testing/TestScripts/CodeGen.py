import TestScripts.Parser
import sys
import os.path
import math

groupCode="""class %s : public Client::Group
{
   public:
     %s(Testing::testID_t id):Client::Group(id)
 %s
     { 
        %s
     }
    private:
        %s
};
"""

suiteCode="""
#include \"%s.h\"
    %s::%s(Testing::testID_t id):Client::Suite(id)
    {
        %s
    }
"""

def decodeHex(hexstr,bits,mask):
    value = int(hexstr,16) & mask
    if value & (1 << (bits-1)):
         value -= 1 << bits
    return value

def createMissingDir(destPath):
  theDir=os.path.normpath(os.path.dirname(destPath))
  if not os.path.exists(theDir):
      os.makedirs(theDir)
      
class CodeGen:
    """ Code generation from the parsed description file
    
    Generate include files, cpp files, and .txt files for test control
    Generation depends on the mode (fpga or semihosting)
    """

    def __init__(self,patternDir,paramDir,fpga):
        """ Create a CodeGen object
      
        Args:
          patternDir (str) : where pattern must be read
              Used to generate include file in fpag mode
          fpga (bool) : false in semihosting mode
        Raises:
          Nothing 
        Returns:
          CodeGen : CodeGen object
        """
        self._fpga = fpga
        self._patternDir = patternDir
        self._paramDir = paramDir
        self._currentPaths = [self._patternDir]
        self._currentParamPaths = [self._paramDir]
        self._alignment=8
   
    def _genGroup(self,root,fi):
        """ Generate header definition for a group of tests
      
        Args:
          root (TreeElem) : group object to use
          fi (file) : Header file (TestDesc.h)
              Where to write include definitions for the classes
        Raises:
          Nothing 
        Returns:
          Nothing
        """

        # Get the CPP class name
        theClass = root.data["class"]
        # Get the group ID
        theId = root.id
        
        varInit = ""
        testInit = ""
        testDecl = ""
       
        # Generate initializer for the constructors
        # All the member variables corresponding to the
        # suites and groups contained in this group.
        i = 1
        for c in root:
            theData = c.data
            theMemberId = c.id
            if not theData["deprecated"]:
               theMemberClass = theData["class"]
               theMemberVar = theMemberClass + "Var"
               varInit += (",%s(%d)\n" % (theMemberVar,theMemberId))
            i = i + 1

        # Generate initialization code for the constructor
        i = 1
        for c in root:
            theData = c.data
            if theData["deprecated"]:
               testInit += "this->addContainer(NULL);"
            else:
               theMemberClass = theData["class"]
               theMemberVar = theMemberClass+"Var"
               testInit += "this->addContainer(&%s);\n" % (theMemberVar)
            i = i + 1

        # Generate member variables for the groups and the suites
        # contained in this group. 
        i = 1
        for c in root:
            theData = c.data
            if not theData["deprecated"]:
               theMemberClass = theData["class"]
               theMemberVar = theMemberClass+"Var"
               theMemberId = c.id
               testDecl += "%s %s;\n" % (theMemberClass,theMemberVar)
            i = i + 1

        fi.write(groupCode % (theClass,theClass,varInit,testInit,testDecl) )

    def _genSuite(self,root,thedir,sourceFile):
        """ Generate source definition for suite
      
        Args:
          root (TreeElem) : suite object to use
          fi (file) : Source file (TestDesc.cpp)
              Where to write source definitions for the classes
        Raises:
          Nothing 
        Returns:
          Nothing
        """

        # Get the CPP class
        theClass = root.data["class"]
        testInit = ""
        testDecl=""
        declared={}

        # Generate constructor for the class
        # Test functions are added in the constructor.
        # Keep track of the list of test functions 
        # (since a test function can be reused by several tests)
        for c in root:
            theData = c.data
            theId = c.id
            theTestName = theData["class"]
            if not theData["deprecated"]:
               testInit += "this->addTest(%d,(Client::test)&%s::%s);\n" % (theId,theClass,theTestName)
            # To be able to deprecated tests without having to change the code
            # we dump const declaration even for deprecated functions
            if theTestName not in declared:
                  testDecl += "void %s();\n" % (theTestName)
                  declared[theTestName] = 1

        # Generate the suite constructor.
        sourceFile.write(suiteCode % (theClass,theClass,theClass,testInit) )

        # Generate specific header file for this suite
        with open(os.path.join(thedir,"%s_decl.h" % theClass),"w") as decl:
            # Use list of test functions to
            # declare them in the header.
            # Since we don't want to declare the functino several times,
            # that's why we needed to keep track of functions in the code above:
            # to remove redundancies since the function
            # can appear several time in the root object.
            decl.write(testDecl)
            # Generate the pattern ID for patterns
            # defined in this suite
            decl.write("\n// Pattern IDs\n")
            i = 0
            for p in root.patterns:
                newId = "static const int %s=%d;\n" % (p[0],i)
                decl.write(newId)
                i = i + 1

            # Generate output ID for the output categories
            # defined in this suite
            decl.write("\n// Output IDs\n")
            i = 0
            for p in root.outputs:
                newId = "static const int %s=%d;\n" % (p[0],i)
                decl.write(newId)
                i = i + 1

            # Generate test ID for the test define din this suite
            i = 0
            decl.write("\n// Test IDs\n")
            for c in root:
              theData = c.data
              theId = c.id
              
              # To be able to deprecated tests without having to change the code
              # we dump const declaration even for deprecated functions
              #if not theData["deprecated"]:
              theTestName = theData["class"]
              defTestID = "static const int %s_%d=%d;\n" % (theTestName.upper(),theId,theId)
              decl.write(defTestID)
              i = i + 1

        

    def _genCode(self,root,dir,sourceFile,headerFile):
        """ Generate code for the tree of tests
      
        Args:
          root (TreeElem) : root object containing the tree
          sourceFile (file) : Source file (TestDesc.cpp)
              Where to write source definitions for the classes
          headerFile (file) : Heade5 file (TestDesc.h)
              Where to write header definitions for the classes
        Raises:
          Nothing 
        Returns:
          Nothing
        """
        deprecated = root.data["deprecated"]
        if not deprecated:
           if root.kind == TestScripts.Parser.TreeElem.GROUP:
               # For a group, first iterate on the children then on the
               # parent because the parent definitino in the header is
               # dependent on the children definition (member varaiables)
               # So children must gbe defined before.
               for c in root:
                   self._genCode(c,dir,sourceFile,headerFile)
               self._genGroup(root,headerFile)
   
           # For a suite, we do not need to recurse since it is a leaf
           # of the tree of tests and is containing a list of tests to
           # generate
           if root.kind == TestScripts.Parser.TreeElem.SUITE:
               self._genSuite(root,dir,sourceFile)

    def getSuites(self,root,listOfSuites):
        """ Extract the list of all suites defined in the tree
      
        Args:
          root (TreeElem) : root object containing the tree
          listOfSuites (list) : list of suites
        Raises:
          Nothing 
        Returns:
          list : list of suites
        """
        deprecated = root.data["deprecated"]
        if not deprecated:
           theClass = root.data["class"]
           if root.kind == TestScripts.Parser.TreeElem.SUITE:
              # Append current suite to the list and return the result
              listOfSuites.append(theClass)
              return(listOfSuites)
           elif root.kind == TestScripts.Parser.TreeElem.GROUP:
             # Recurse on the children to get the listOfSuite.
             # getSuites is appending to the list argument.
             # So the line listOfSuites = self.getSuites(c,listOfSuites)
             # is a fold.
             for c in root: 
               listOfSuites = self.getSuites(c,listOfSuites)
             return(listOfSuites)
           else:
             return(listOfSuites)
        else:
          return(listOfSuites)

    def _genText(self,root,textFile):
        """ Generate the text file which is driving the tests in semihosting

        Format of file is:
        node kind (group, suite or test)
        node id (group id, suite id, test id)
        y or n to indicate if this node has some parameters
          if y there is the parameter ID
        y or n to indicate if entering this node is adding a new
        folder to the path used to get input files
          if y, the path for the node is added.

        Then, if node is a suite, description of the suite is generated.
         Number of parameters or 0
           It is the number of arguments used by function setUp (arity).
           It is not the number of samples in a paramater file
           The number of samples is always a multiple
           For instance if width and height are the parameters then number
           of parameters (arity) is 2.
           But each parameter file may contain several combination of (width,height)
           So the lenght of those files will be a multiple of 2.

         The number of patterns path is generated
         The patterns names are generated

         The number of output names is generated
         The output names are generated

         The number of parameter path is generated
         The parameter description are generated
         p
         path if a path
         g
         gen data if a generator
         If a generator:
           Generator kind (only 1 = cartesian product generator)
           Number of input samples (number of values to encode all parameters)
           Number of output samples (number of generated values)
           Number of dimensions
           For each dimension
             Length
             Samples
        For instance, with A=[1,2] and B=[1,2,3].
        Input dimension is 1+2 + 1 + 3 = 7 (number of values needed to encode the two lists
        with their length).
        Number of output dimensions is 2 * 3 = 6

        So, for a test (which is also a TreeElement in the structure),
        only node kind and node id are generated followed by 
        param description and n for folder (since ther is never any folder)
        
        In the test description file, there is never any way to change the pattern
        folder for a test.
        
      
        Args:
          root (TreeElem) : root object containing the tree
          textFile (file) : where to write the driving description
        Raises:
          Nothing 
        Returns:
          Nothing
        """
        deprecated = root.data["deprecated"]
        if not deprecated:
           textFile.write(str(root.kind))
           textFile.write(" ")
           textFile.write(str(root.id))
           textFile.write("\n")

           if "PARAMID" in root.data:
              if root.parameterToID:
                textFile.write("y\n")
                paramId = root.parameterToID[root.data["PARAMID"]]
                textFile.write(str(paramId))
                textFile.write("\n")
              elif root.parent.parameterToID:
                textFile.write("y\n")
                paramId = root.parent.parameterToID[root.data["PARAMID"]]
                textFile.write(str(paramId))
                textFile.write("\n")
              else:
                textFile.write("n\n")
           else:
              textFile.write("n\n")

           # Always dump even if there is no path for a test
           # so for a test it will always be n
           # It is done to simplify parsing on C side
           if root.path:
             textFile.write("y\n")
             textFile.write(root.path)
             textFile.write('\n')
           else:
             textFile.write("n\n")

          
           if root.kind == TestScripts.Parser.TreeElem.SUITE:
              # Here we dump the number of parameters used
              # for the tests / benchmarks
              if root.params:
                 textFile.write("%d\n" % len(root.params.full))
              else:
                 textFile.write("0\n")

              # Generate patterns path
              textFile.write("%d\n" % len(root.patterns))
              for (patid,patpath) in root.patterns:
                 #textFile.write(patid)
                 #textFile.write("\n")
                 textFile.write(patpath.strip())
                 textFile.write("\n")

              # Generate output names
              textFile.write("%d\n" % len(root.outputs))
              for (outid,outname) in root.outputs:
                 #textFile.write(patid)
                 #textFile.write("\n")
                 textFile.write(outname.strip())
                 textFile.write("\n")

              # Generate parameters path or generator
              textFile.write("%d\n" % len(root.parameters))
              for (paramKind,parid,parpath) in root.parameters:
                 if paramKind == TestScripts.Parser.TreeElem.PARAMFILE:
                    textFile.write("p\n")
                    textFile.write(parpath.strip())
                    textFile.write("\n")
                 # Generator kind (only 1 = cartesian product generator)
                 # Number of input samples (dimensions + vectors)
                 # Number of samples generated when run
                 # Number of dimensions
                 # For each dimension
                 #   Length
                 #   Samples
                 if paramKind == TestScripts.Parser.TreeElem.PARAMGEN:
                    textFile.write("g\n")
                    dimensions = len(parpath)
                    nbOutputSamples = 1 
                    nbInputSamples = 0
                    for c in parpath:
                      nbOutputSamples = nbOutputSamples * len(c["INTS"])
                      nbInputSamples = nbInputSamples + len(c["INTS"]) + 1
                    textFile.write("1")
                    textFile.write("\n")
                    textFile.write(str(nbInputSamples))
                    textFile.write("\n")
                    textFile.write(str(nbOutputSamples))
                    textFile.write("\n")
                    textFile.write(str(dimensions))
                    textFile.write("\n")
                    for c in parpath:
                      textFile.write(str(len(c["INTS"])))
                      textFile.write("\n")
                      for d in c["INTS"]:
                        textFile.write(str(d))
                        textFile.write("\n")

           # Iterate on the children
           for c in root:
               self._genText(c,textFile)

    def _write64(self,v,f):
      """ Write four integers into a C char array to represent word32
      
      It is used to dump input patterns in include files
      or test drive in include file
            
      Args:
        v (int) : The int64 to write
        f (file) : the opended file
      Raises:
        Nothing 
      Returns:
        Nothing
      """
      a=[0,0,0,0,0,0,0,0]
      a[0]= v & 0x0FF
      v = v >> 8 
      a[1]= v & 0x0FF
      v = v >> 8 
      a[2]= v & 0x0FF
      v = v >> 8 
      a[3]= v & 0x0FF
      v = v >> 8 
      a[4]= v & 0x0FF
      v = v >> 8 
      a[5]= v & 0x0FF
      v = v >> 8 
      a[6]= v & 0x0FF
      v = v >> 8 
      a[7]= v & 0x0FF
      v = v >> 8 
      f.write("%d,%d,%d,%d,%d,%d,%d,%d,\n" % (a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]))

    def _write32(self,v,f):
      """ Write four integers into a C char array to represent word32
      
      It is used to dump input patterns in include files
      or test drive in include file
            
      Args:
        v (int) : The int32 to write
        f (file) : the opended file
      Raises:
        Nothing 
      Returns:
        Nothing
      """
      a=[0,0,0,0]
      a[0]= v & 0x0FF
      v = v >> 8 
      a[1]= v & 0x0FF
      v = v >> 8 
      a[2]= v & 0x0FF
      v = v >> 8 
      a[3]= v & 0x0FF
      v = v >> 8 
      f.write("%d,%d,%d,%d,\n" % (a[0],a[1],a[2],a[3]))


    def _write16(self,v,f):
      """ Write 2 integers into a C char array to represent word32
      
      It is used to dump input patterns in include files
      or test drive in include file
            
      Args:
        v (int) : The int3 to write
        f (file) : the opended file
      Raises:
        Nothing 
      Returns:
        Nothing
      """
      a=[0,0]
      a[0]= v & 0x0FF
      v = v >> 8 
      a[1]= v & 0x0FF
      f.write("%d,%d,\n" % (a[0],a[1]))

    def _write8(self,v,f):
      """ Write 4 integers into a C char array to represent word8
      
      It is used to dump input patterns in include files
      or test drive in include file
            
      Args:
        v (int) : The int to write
        f (file) : the opended file
      Raises:
        Nothing 
      Returns:
        Nothing
      """
      a=[0]
      a[0]= v & 0x0FF
      f.write("%d,\n" % (a[0]))

    def _writeString(self,v,f):
        """ Write a C string into a C char array to represent as a list of int
        
        It is used to dump input patterns in include files
        or test drive in include file
              
        Args:
          v (str) : The string to write
          f (file) : the opended file
        Raises:
          Nothing 
        Returns:
          Nothing
        """
        for c in v:
          f.write("'%c'," % c)
        f.write("'\\0',\n")




    def convertToInt(self,k,s):
      v = 0
      if k == "D":
        v = decodeHex(s,64,0x0FFFFFFFFFFFFFFFF)
      if k == "W":
        v = decodeHex(s,32,0x0FFFFFFFF)
      if k == "H":
        v = decodeHex(s,16,0x0FFFF)
      if k == "B":
        v = decodeHex(s,8,0x0FF)
      return(v)

    def addPattern(self,includeFile,path):
        """ Add a pattern to the include file
        
        It is writing sequence of int into a C char array
        to represent the pattern.

        Assuming the C chr array is aligned, pattern offset are
        aligned too.

        So, some padding with 0 may also be generated after the patterm
              
        Args:
          includeFile (file) : Opened include file
          path (str) : Path to file containing the data
        Raises:
          Nothing 
        Returns:
          (int,int) : The pattern offset in the array and the number of samples
        """
        # Current offset in the array which is the offset for the
        # pattern being added to this array
        returnOffset = self._offset

        # Read the pattern for the pattern file
        with open(path,"r") as pat:
          # Read pattern word size (B,H or W for 8, 16 or 32 bits)
          # Patterns are containing data in hex format (big endian for ARM)
          # So there is no conversion to do.
          # Hex data is read and copied as it is in the C array
          
          k =pat.readline().strip()
          sampleSize=1
          if k == 'D':
              sampleSize = 8 
          if k == 'W':
              sampleSize = 4 
          if k == 'H':
              sampleSize = 2 
          # Read number of samples
          nbSamples = int(pat.readline().strip())
          # Compute new offset based on pattern length only
          newOffset = self._offset + sampleSize * nbSamples
          # Compute padding due to alignment constraint
          pad = self._alignment*math.ceil(newOffset / self._alignment) - newOffset
          # New offset in the array
          self._offset=newOffset + pad

          # Generate all samples into the C array
          for i in range(nbSamples):
            # In pattern file we have a comment giving the
            # true value (for instance float)
            # and then a line giving the hex data
            # We Ignore comment
            pat.readline()
            # Then we read the Value 
            v = self.convertToInt(k,pat.readline())
            # Depending on the word size, this hex must be writen to 
            # the C array as 4,2 or 1 number.
            if k == 'D':
               self._write64(v,includeFile)
            if k == 'W':
               self._write32(v,includeFile)
            if k == 'H':
               self._write16(v,includeFile)
            if k == 'B':
               self._write8(v,includeFile)
               #includeFile.write("%d,\n" % v)
          # Add the padding to the pattern
          for i in range(pad):
               includeFile.write("0,\n")
        
        return(returnOffset,nbSamples)

    def addParameter(self,includeFile,path):
        """ Add a parameter array to the include file
        
        It is writing sequence of int into a C char array
        to represent the pattern.

        Assuming the C chr array is aligned, pattern offset are
        aligned too.

        So, some padding with 0 may also be generated after the patterm
              
        Args:
          includeFile (file) : Opened include file
          path (str) : Path to file containing the data
        Raises:
          Nothing 
        Returns:
          (int,int) : The pattern offset in the array and the number of samples
        """
        # Current offset in the array which is the offset for the
        # pattern being added to this array
        returnOffset = self._offset

        # Read the pattern for the pattern file
        with open(path,"r") as pat:
          # Read pattern word size (B,H or W for 8, 16 or 32 bits)
          # Patterns are containing data in hex format (big endian for ARM)
          # So there is no conversion to do.
          # Hex data is read and copied as it is in the C array
          
          sampleSize = 4 
          # Read number of samples
          nbSamples = int(pat.readline().strip())

          # Compute new offset based on pattern length only
          newOffset = self._offset + sampleSize * nbSamples
          # Compute padding due to alignment constraint
          pad = self._alignment*math.ceil(newOffset / self._alignment) - newOffset
          # New offset in the array
          self._offset=newOffset + pad

          # Generate all samples into the C array
          for i in range(nbSamples):
            # In pattern file we have a comment giving the
            # true value (for instance float)
            # and then a line giving the hex data
            # Then we read the Value 
            v = int(pat.readline().strip(),0)
            # Depending on the word size, this hex must be writen to 
            # the C array as 4,2 or 1 number.
            self._write32(v,includeFile)
          # Add the padding to the pattern
          for i in range(pad):
               includeFile.write("0,\n")
        
        return(returnOffset,nbSamples)

    def _genDriver(self,root,driverFile,includeFile):
        """ Generate the driver file and the pattern file

        Args:
          root (TreeElement) : Tree of test descriptions
          driverFile (file) : where to generate C array for test descriptions
          includeFile (file) : where to generate C array for patterns
        Raises:
          Nothing 
        Returns:
          Nothing
        """

        #if root.parent:
        #   print(root.parent.data["message"])
        #print("  ",root.data["message"])
        #print(self._currentPaths)
        
        deprecated = root.data["deprecated"]
        # We follow a format quite similar to what is generated in _genText
        # for the text description
        # But here we are using an offset into the pattern array
        # rather than a path to a pattern file.
        # It is the only difference
        # But for outputs we still need the path so the logic is the same
        # Path for output is required to be able to extract data from the stdout file
        # and know where the data must be written to.
        # Please refer to comments of _genText for description of the format

        oldPath = self._currentPaths.copy()
        oldParamPath = self._currentParamPaths.copy()
        if not deprecated:
           # We write node kind and node id
           self._write32(root.kind,driverFile)
           self._write32(root.id,driverFile)

           

           if "PARAMID" in root.data:
              if root.parameterToID:
                driverFile.write("'y',")
                paramId = root.parameterToID[root.data["PARAMID"]]
                self._write32(int(paramId),driverFile)
              elif root.parent.parameterToID:
                driverFile.write("'y',")
                paramId = root.parent.parameterToID[root.data["PARAMID"]]
                self._write32(int(paramId),driverFile)
              else:
                driverFile.write("'n',")
           else:
              driverFile.write("'n',")

           # We write a folder path
           # if folder changed in test description file
           # Always dumped for a test even if no path for
           # a test. So a test will always have n
           # It is done to simplify parsing on C side
           if root.path:
             self._currentPaths.append(root.path)
             self._currentParamPaths.append(root.path)
             driverFile.write("'y',")
             self._writeString(root.path,driverFile)
           else:
             driverFile.write("'n',\n")

           if root.kind == TestScripts.Parser.TreeElem.SUITE:
              # Number of parameters
              if root.params:
                 self._write32(len(root.params.full),driverFile)
              else:
                 self._write32(0,driverFile)

              # Patterns offsets are written
              # and pattern length since the length is not available in a file
              # like for semihosting version
              self._write32(len(root.patterns),driverFile)
              for (patid,patpath) in root.patterns:
                 temp = self._currentPaths.copy()
                 temp.append(patpath)
                
                 includeFile.write("// " + os.path.join(*temp) + "\n")
                 offset,nbSamples=self.addPattern(includeFile,os.path.join(*temp))

                 #driverFile.write(patpath)
                 #driverFile.write("\n")
                 self._write32(offset,driverFile)
                 self._write32(nbSamples,driverFile)

              # Generate output names
              self._write32(len(root.outputs),driverFile)
              for (outid,outname) in root.outputs:
                 #textFile.write(patid)
                 #textFile.write("\n")
                 self._writeString(outname.strip(),driverFile)

              # Parameters offsets are written
              # and parameter length since the length is not available in a file
              # like for semihosting version
              self._write32(len(root.parameters),driverFile)
              for (paramKind,parid,parpath) in root.parameters:
                 if paramKind == TestScripts.Parser.TreeElem.PARAMFILE:
                    temp = self._currentParamPaths.copy()
                    temp.append(parpath)

                    includeFile.write("// " + os.path.join(*temp) + "\n")
                    offset,nbSamples=self.addParameter(includeFile,os.path.join(*temp))
   
                    #driverFile.write(patpath)
                    #driverFile.write("\n")
                    driverFile.write("'p',")
                    self._write32(offset,driverFile)
                    self._write32(nbSamples,driverFile)
                 # TO DO
                 if paramKind == TestScripts.Parser.TreeElem.PARAMGEN:
                    temp = self._currentParamPaths.copy()
                   
                    includeFile.write("// " + os.path.join(*temp) + "\n")
                    
                    driverFile.write("'g',")
                    dimensions = len(parpath)
                    nbOutputSamples = 1 
                    nbInputSamples = 0
                    for c in parpath:
                      nbOutputSamples = nbOutputSamples * len(c["INTS"])
                      nbInputSamples = nbInputSamples + len(c["INTS"]) + 1

                    self._write32(1,driverFile)
                    self._write32(nbInputSamples,driverFile)
                    self._write32(nbOutputSamples,driverFile)
                    self._write32(dimensions,driverFile)

                    for c in parpath:
                      self._write32(len(c["INTS"]),driverFile)
                      for d in c["INTS"]:
                        self._write32(d,driverFile)


           # Recurse on the children
           for c in root:
               self._genDriver(c,driverFile,includeFile)

        self._currentPaths = oldPath.copy()
        self._currentParamPaths = oldParamPath.copy()

    def genCodeForTree(self,root):
      """ Generate all files from the trees of tests
      
      Args:
        root (TreeElement) : Tree of test descriptions
      Raises:
        Nothing 
      Returns:
        Nothing
      """
      # Get a list of all suites contained in the tree
      suites = self.getSuites(root,[])

      
      # Generate .cpp and .h files neded to run the tests
      with open("GeneratedSource/TestDesc.cpp","w") as sourceFile:
       with open("GeneratedInclude/TestDesc.h","w") as headerFile:
         headerFile.write("#include \"Test.h\"\n")
         headerFile.write("#include \"Pattern.h\"\n")

         sourceFile.write("#include \"Test.h\"\n")
         for s in suites:
          headerFile.write("#include \"%s.h\"\n" % s)
         self._genCode(root,"GeneratedInclude",sourceFile,headerFile)
       
      # Generate a driver file for semihosting
      # (always generated for debug purpose since it is the reference format)
      with open("TestDesc.txt","w") as textFile:
           self._genText(root,textFile)

      # If fpga mode we need to generate
      # a include file version of the driver file and of 
      # the pattern files.
      # Driver file is similar in this case but different from semihosting
      # one.
      if not self._fpga:
         with open("GeneratedInclude/TestDrive.h","w") as driverFile:
            driverFile.write("// Empty driver include in semihosting mode")
         with open("GeneratedInclude/Patterns.h","w") as includeFile:
            includeFile.write("// Empty pattern include in semihosting mode")
      else:
        with open("GeneratedInclude/TestDrive.h","w") as driverFile:
          driverFile.write("#ifndef _DRIVER_H_\n")
          driverFile.write("#define _DRIVER_H_\n")
          driverFile.write("__ALIGNED(8) const char testDesc[]={\n")
          self._offset=0
          with open("GeneratedInclude/Patterns.h","w") as includeFile:
            includeFile.write("#ifndef _PATTERNS_H_\n")
            includeFile.write("#define _PATTERNS_H_\n")
            includeFile.write("__ALIGNED(8) const char patterns[]={\n")
            self._genDriver(root,driverFile,includeFile)
            includeFile.write("};\n")
            includeFile.write("#endif\n")
          driverFile.write("};\n")
          driverFile.write("#endif\n")

           
 