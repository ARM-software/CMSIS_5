#from distutils.core import setup, Extension
from setuptools import setup, Extension,find_packages
from distutils.util import convert_path
import glob
import numpy
import sys
import os
import os.path
import re
import pathlib

here = pathlib.Path(__file__).parent.resolve()
ROOT = here
ROOT=""

includes = [os.path.join(ROOT,"Include"),os.path.join(ROOT,"PrivateInclude"),os.path.join("PythonWrapper","cmsisdsp_pkg","src")]

if sys.platform == 'win32':
  cflags = ["-DWIN","-DCMSISDSP","-DUNALIGNED_SUPPORT_DISABLE"] 
else:
  cflags = ["-Wno-attributes","-Wno-unused-function","-Wno-unused-variable","-Wno-implicit-function-declaration","-DCMSISDSP","-D__GNUC_PYTHON__"]

transform = glob.glob(os.path.join(ROOT,"Source","TransformFunctions","*.c"))

# Files are present when creating the source distribution
# but they are not copied to the source distribution
# When doing pip install those files are not prevent
# and it should not fail
try:
  transform.remove(os.path.join(ROOT,"Source","TransformFunctions","TransformFunctions.c"))
  transform.remove(os.path.join(ROOT,"Source","TransformFunctions","TransformFunctionsF16.c"))
except:
  pass

support = glob.glob(os.path.join(ROOT,"Source","SupportFunctions","*.c"))

try:
  support.remove(os.path.join(ROOT,"Source","SupportFunctions","SupportFunctions.c"))
  support.remove(os.path.join(ROOT,"Source","SupportFunctions","SupportFunctionsF16.c"))
except:
  pass

fastmath = glob.glob(os.path.join(ROOT,"Source","FastMathFunctions","*.c"))
try:
  fastmath.remove(os.path.join(ROOT,"Source","FastMathFunctions","FastMathFunctions.c"))
except:
  pass

filtering = glob.glob(os.path.join(ROOT,"Source","FilteringFunctions","*.c"))
try:
  filtering.remove(os.path.join(ROOT,"Source","FilteringFunctions","FilteringFunctions.c"))
  filtering.remove(os.path.join(ROOT,"Source","FilteringFunctions","FilteringFunctionsF16.c"))
except:
  pass

matrix = glob.glob(os.path.join(ROOT,"Source","MatrixFunctions","*.c"))
try:
  matrix.remove(os.path.join(ROOT,"Source","MatrixFunctions","MatrixFunctions.c"))
  matrix.remove(os.path.join(ROOT,"Source","MatrixFunctions","MatrixFunctionsF16.c"))
except:
  pass

statistics = glob.glob(os.path.join(ROOT,"Source","StatisticsFunctions","*.c"))
try:
  statistics.remove(os.path.join(ROOT,"Source","StatisticsFunctions","StatisticsFunctions.c"))
  statistics.remove(os.path.join(ROOT,"Source","StatisticsFunctions","StatisticsFunctionsF16.c"))
except:
  pass

complexf = glob.glob(os.path.join(ROOT,"Source","ComplexMathFunctions","*.c"))
try:
  complexf.remove(os.path.join(ROOT,"Source","ComplexMathFunctions","ComplexMathFunctions.c"))
  complexf.remove(os.path.join(ROOT,"Source","ComplexMathFunctions","ComplexMathFunctionsF16.c"))
except:
  pass

basic = glob.glob(os.path.join(ROOT,"Source","BasicMathFunctions","*.c"))
try:
  basic.remove(os.path.join(ROOT,"Source","BasicMathFunctions","BasicMathFunctions.c"))
  basic.remove(os.path.join(ROOT,"Source","BasicMathFunctions","BasicMathFunctionsF16.c"))
except:
  pass

controller = glob.glob(os.path.join(ROOT,"Source","ControllerFunctions","*.c"))
try:
  controller.remove(os.path.join(ROOT,"Source","ControllerFunctions","ControllerFunctions.c"))
except:
  pass

common = glob.glob(os.path.join(ROOT,"Source","CommonTables","*.c"))
try:
  common.remove(os.path.join(ROOT,"Source","CommonTables","CommonTables.c"))
  common.remove(os.path.join(ROOT,"Source","CommonTables","CommonTablesF16.c"))
except:
  pass

interpolation = glob.glob(os.path.join(ROOT,"Source","InterpolationFunctions","*.c"))
try:
  interpolation.remove(os.path.join(ROOT,"Source","InterpolationFunctions","InterpolationFunctions.c"))
  interpolation.remove(os.path.join(ROOT,"Source","InterpolationFunctions","InterpolationFunctionsF16.c"))
except:
  pass

quaternion = glob.glob(os.path.join(ROOT,"Source","QuaternionMathFunctions","*.c"))
try:
  quaternion.remove(os.path.join(ROOT,"Source","QuaternionMathFunctions","QuaternionMathFunctions.c"))
except:
  pass

distance = glob.glob(os.path.join(ROOT,"Source","DistanceFunctions","*.c"))
try:
  distance.remove(os.path.join(ROOT,"Source","DistanceFunctions","DistanceFunctions.c"))
except:
  pass

bayes = glob.glob(os.path.join(ROOT,"Source","BayesFunctions","*.c"))
try:
  bayes.remove(os.path.join(ROOT,"Source","BayesFunctions","BayesFunctions.c"))
except:
  pass

svm = glob.glob(os.path.join(ROOT,"Source","SVMFunctions","*.c"))
try:
  svm.remove(os.path.join(ROOT,"Source","SVMFunctions","SVMFunctions.c"))
except:
  pass

# Add dependencies
transformMod = transform + common + basic + complexf + fastmath + matrix + statistics
statisticsMod = statistics + common + fastmath + basic
interpolationMod = interpolation + common
filteringMod = filtering + common + support + fastmath + basic
controllerMod = controller + common

matrixMod = matrix 
supportMod = support 
complexfMod = complexf + fastmath + common + basic
basicMod = basic
quaternionMod = quaternion
fastmathMod = basic + fastmath + common
distanceMod = distance + common + basic + statistics + fastmath
bayesMod = bayes + fastmath + common + statistics + basic
svmMod = svm + fastmath + common + basic


filteringMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_filtering.c"))
matrixMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_matrix.c"))
supportMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_support.c"))
statisticsMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_statistics.c"))
complexfMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_complexf.c"))
basicMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_basic.c"))
controllerMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_controller.c"))
transformMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_transform.c"))
interpolationMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_interpolation.c"))
quaternionMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_quaternion.c"))
fastmathMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_fastmath.c"))
distanceMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_distance.c"))
bayesMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_bayes.c"))
svmMod.append(os.path.join("PythonWrapper","cmsisdsp_pkg","src","cmsisdsp_svm.c"))




missing=set([
  ])

def notf16(number):
  if re.search(r'f16',number):
     return(False)
  if re.search(r'F16',number):
     return(False)
  return(True)

def isnotmissing(src):
  name=os.path.splitext(os.path.basename(src))[0]
  return(not (name in missing))

# If there are too many files, the linker command is failing on Windows.
# So f16 functions are removed since they are not currently available in the wrapper.
# A next version will have to structure this wrapper more cleanly so that the
# build can work even with more functions

filtering = list(filter(isnotmissing,list(filter(notf16, filteringMod))))
matrix = list(filter(isnotmissing,list(filter(notf16, matrixMod))))
support = list(filter(isnotmissing,list(filter(notf16, supportMod))))
statistics = list(filter(isnotmissing,list(filter(notf16, statisticsMod))))
complexf = list(filter(isnotmissing,list(filter(notf16, complexfMod))))
basic = list(filter(isnotmissing,list(filter(notf16, basicMod))))
controller = list(filter(isnotmissing,list(filter(notf16, controllerMod))))
transform = list(filter(isnotmissing,list(filter(notf16, transformMod))))
interpolation = list(filter(isnotmissing,list(filter(notf16, interpolationMod))))
quaternion = list(filter(isnotmissing,list(filter(notf16, quaternionMod))))
fastmath = list(filter(isnotmissing,list(filter(notf16, fastmathMod))))
distance = list(filter(isnotmissing,list(filter(notf16, distanceMod))))
bayes = list(filter(isnotmissing,list(filter(notf16, bayesMod))))
svm = list(filter(isnotmissing,list(filter(notf16, svmMod))))

#for l in filtering:
#  print(os.path.basename(l))
#quit()

def mkModule(name,srcs,funcDir,newCflags=[]):
  localinc = os.path.join(ROOT,"Source",funcDir)
  return(Extension(name,
                    sources = (srcs
                              )
                              ,
                    include_dirs =  [localinc] + includes + [numpy.get_include()],
                    extra_compile_args = cflags + newCflags
                              ))

flagsForCommonWithoutFFT=["-DARM_DSP_CONFIG_TABLES", 
    "-DARM_FAST_ALLOW_TABLES", 
    "-DARM_ALL_FAST_TABLES"]

moduleFiltering = mkModule('cmsisdsp_filtering',filtering,"FilteringFunctions",flagsForCommonWithoutFFT)
moduleMatrix = mkModule('cmsisdsp_matrix',matrix,"MatrixFunctions")
moduleSupport = mkModule('cmsisdsp_support',support,"SupportFunctions")
moduleStatistics = mkModule('cmsisdsp_statistics',statistics,"StatisticsFunctions",flagsForCommonWithoutFFT)
moduleComplexf= mkModule('cmsisdsp_complexf',complexf,"ComplexMathFunctions")
moduleBasic = mkModule('cmsisdsp_basic',basic,"BasicMathFunctions")
moduleController = mkModule('cmsisdsp_controller',controller,"ControllerFunctions",flagsForCommonWithoutFFT)
moduleTransform = mkModule('cmsisdsp_transform',transform,"TransformFunctions")
moduleInterpolation = mkModule('cmsisdsp_interpolation',interpolation,"InterpolationFunctions",flagsForCommonWithoutFFT)
moduleQuaternion = mkModule('cmsisdsp_quaternion',quaternion,"QuaternionMathFunctions")
moduleFastmath = mkModule('cmsisdsp_fastmath',fastmath,"FastMathFunctions",flagsForCommonWithoutFFT)
moduleDistance = mkModule('cmsisdsp_distance',distance,"DistanceFunctions",flagsForCommonWithoutFFT)
moduleBayes = mkModule('cmsisdsp_bayes',bayes,"BayesFunctions",flagsForCommonWithoutFFT)
moduleSVM = mkModule('cmsisdsp_svm',svm,"SVMFunctions",flagsForCommonWithoutFFT)




def build():
  if sys.version_info.major < 3:
      print('setup.py: Error: This package only supports Python 3.', file=sys.stderr)
      sys.exit(1)
  
  main_ns = {}
  ver_path = convert_path(os.path.join("cmsisdsp","version.py"))
  with open(ver_path) as ver_file:
      exec(ver_file.read(), main_ns)

  setup (name = 'cmsisdsp',
         version = main_ns['__version__'],
         packages=["cmsisdsp","cmsisdsp.sdf","cmsisdsp.sdf.nodes","cmsisdsp.sdf.nodes.host","cmsisdsp.sdf.scheduler"],
         description = 'CMSIS-DSP Python API',
         long_description=open("PythonWrapper_README.md").read(),
         long_description_content_type='text/markdown',
         ext_modules = [moduleFiltering ,
                        moduleMatrix,
                        moduleSupport,
                        moduleStatistics,
                        moduleComplexf,
                        moduleBasic,
                        moduleController,
                        moduleTransform,
                        moduleInterpolation, 
                        moduleQuaternion,
                        moduleFastmath,
                        moduleDistance,
                        moduleBayes,
                        moduleSVM
                        ],
         include_package_data=True,
         author = 'Copyright (C) 2010-2022 ARM Limited or its affiliates. All rights reserved.',
         author_email = 'christophe.favergeon@arm.com',
         url="https://arm-software.github.io/CMSIS_5/DSP/html/index.html",
         python_requires='>=3.6',
         license="License :: OSI Approved :: Apache Software License",
         platforms=['any'],
         classifiers=[
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: Implementation :: CPython",
                "Programming Language :: C",
                "License :: OSI Approved :: Apache Software License",
                "Operating System :: OS Independent",
                "Development Status :: 4 - Beta",
                "Topic :: Software Development :: Embedded Systems",
                "Topic :: Scientific/Engineering :: Mathematics",
                "Environment :: Console",
                "Intended Audience :: Developers",
          ],
          keywords=['development','dsp','cmsis','cmsis-dsp','Arm','signal processing','maths'],
          install_requires=['numpy>=1.19',
          'networkx>=2.5',
          'jinja2>= 2.0, <3.0',
          'sympy>=1.6',
          'markupsafe<2.1'
          ],
          project_urls={  # Optional
             'Bug Reports': 'https://github.com/ARM-software/CMSIS_5/issues',
             'Source': 'https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP',
            }
          )
       

build()