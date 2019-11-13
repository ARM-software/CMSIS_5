import os.path
import numpy as np
import itertools

# This python module is containing definitions useful for
# generating test patterns.
import Tools


# This function is generating patterns for an addition
def writeTests(config,format):
    # In this test, we can use the same inputs for several tests.
    # For instance, we could test an addition on 3 samples and an addition
    # on 9 samples to be sure that the vectorized code with tail is
    # working.
    # So, we generate two long patterns and in the different tests we may load
    # only a subset of the samples.
    NBSAMPLES=256

    # Two random arrays with gaussian distribution
    data1=np.random.randn(NBSAMPLES)
    data2=np.random.randn(NBSAMPLES)
    
    # We normalize the data to ensure that the q31, q15 and q7 patterns won't
    # be already saturated.
    data1 = Tools.normalize(data1)
    data2 = Tools.normalize(data2)

    # The input patterns are written. The writeInput function of the config object is
    # doing a lot:
    # It is converting the float data to the right format (float, q31, q15 or q7)
    # depending on the config object.
    # It is generating a text file with the right format as recognized by the test framework
    # It is naming the file using the PATTERNDIR (defined below), the id (1 or 2 in this example)
    # and using "Input".
    #
    # So first file is named "Input1_f32.txt"
    config.writeInput(1, data1)
    config.writeInput(2, data2)
    
    # We compute the reference pattern
    ref = data1 + data2

    # Write reference is similar to writeInput.
    # The created file will be named "Reference1_f32.txt"
    config.writeReference(1, ref)
       
# This function is generating patterns for all the types
def generatePatterns():

    # We define the path to the patterns.
    # This path must be compatible with the folder directives used in the desc.txt
    # test description file.
    # By default, the root folder for pattern is Patterns and the root one for
    # Parameters is Parameters.
    # So both path defines in desc.txt are relative to those root folders.
    #
    # The last folder will be completed with the type.
    # So for instance we will get ExampleCategoryF32, ExampleCategoryQ31 ...
    PATTERNDIR = os.path.join("Patterns","Example","ExampleCategory","ExampleCategory")
    PARAMDIR = os.path.join("Parameters","Example","ExampleCategory","ExampleCategory")

    # config object for each type are created
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    
    # Test patterns for each config are generated.
    # Second argument may be used to vary the content fo files
    # depending on the type.
    #
    # For instance, in Tools there is Tools.loopnb which can be used
    # like Tools.loopnb(format,Tools.TAILONLY)
    # It is giving a number of iterations corresponding to the case (Tail only, body only, body and tail)
    # Since the number of lanes depends on the type, testing vectorized code is requiring the use of
    # different lengths according to the type.
    writeTests(configf32,0)
    writeTests(configq31,31)
    writeTests(configq15,15)
    writeTests(configq7,7)
   
# Useful to be able to use this file as a script or to import it from another script
# and use the generatePatterns function
if __name__ == '__main__':
  generatePatterns()