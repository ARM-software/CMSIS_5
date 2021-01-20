from distutils.core import setup, Extension
import glob
import numpy
import config
import sys
import os
from config import ROOT
import re

includes = [os.path.join(ROOT,"Include"),os.path.join(ROOT,"PrivateInclude"),os.path.join("cmsisdsp_pkg","src")]

if sys.platform == 'win32':
  cflags = ["-DWIN",config.cflags,"-DUNALIGNED_SUPPORT_DISABLE"] 
else:
  cflags = ["-Wno-attributes","-Wno-unused-function","-Wno-unused-variable","-Wno-implicit-function-declaration",config.cflags,"-D__GNUC_PYTHON__"]

transform = glob.glob(os.path.join(ROOT,"Source","TransformFunctions","*.c"))
#transform.remove(os.path.join(ROOT,"Source","TransformFunctions","arm_dct4_init_q15.c"))
#transform.remove(os.path.join(ROOT,"Source","TransformFunctions","arm_rfft_init_q15.c"))
transform.remove(os.path.join(ROOT,"Source","TransformFunctions","TransformFunctions.c"))
transform.remove(os.path.join(ROOT,"Source","TransformFunctions","TransformFunctionsF16.c"))

support = glob.glob(os.path.join(ROOT,"Source","SupportFunctions","*.c"))
support.remove(os.path.join(ROOT,"Source","SupportFunctions","SupportFunctions.c"))
support.remove(os.path.join(ROOT,"Source","SupportFunctions","SupportFunctionsF16.c"))

fastmath = glob.glob(os.path.join(ROOT,"Source","FastMathFunctions","*.c"))
fastmath.remove(os.path.join(ROOT,"Source","FastMathFunctions","FastMathFunctions.c"))

filtering = glob.glob(os.path.join(ROOT,"Source","FilteringFunctions","*.c"))
filtering.remove(os.path.join(ROOT,"Source","FilteringFunctions","FilteringFunctions.c"))
filtering.remove(os.path.join(ROOT,"Source","FilteringFunctions","FilteringFunctionsF16.c"))

matrix = glob.glob(os.path.join(ROOT,"Source","MatrixFunctions","*.c"))
matrix.remove(os.path.join(ROOT,"Source","MatrixFunctions","MatrixFunctions.c"))
matrix.remove(os.path.join(ROOT,"Source","MatrixFunctions","MatrixFunctionsF16.c"))

statistics = glob.glob(os.path.join(ROOT,"Source","StatisticsFunctions","*.c"))
statistics.remove(os.path.join(ROOT,"Source","StatisticsFunctions","StatisticsFunctions.c"))
statistics.remove(os.path.join(ROOT,"Source","StatisticsFunctions","StatisticsFunctionsF16.c"))

complexf = glob.glob(os.path.join(ROOT,"Source","ComplexMathFunctions","*.c"))
complexf.remove(os.path.join(ROOT,"Source","ComplexMathFunctions","ComplexMathFunctions.c"))
complexf.remove(os.path.join(ROOT,"Source","ComplexMathFunctions","ComplexMathFunctionsF16.c"))

basic = glob.glob(os.path.join(ROOT,"Source","BasicMathFunctions","*.c"))
basic.remove(os.path.join(ROOT,"Source","BasicMathFunctions","BasicMathFunctions.c"))
basic.remove(os.path.join(ROOT,"Source","BasicMathFunctions","BasicMathFunctionsF16.c"))

controller = glob.glob(os.path.join(ROOT,"Source","ControllerFunctions","*.c"))
controller.remove(os.path.join(ROOT,"Source","ControllerFunctions","ControllerFunctions.c"))

common = glob.glob(os.path.join(ROOT,"Source","CommonTables","*.c"))
common.remove(os.path.join(ROOT,"Source","CommonTables","CommonTables.c"))
common.remove(os.path.join(ROOT,"Source","CommonTables","CommonTablesF16.c"))

interpolation = glob.glob(os.path.join(ROOT,"Source","InterpolationFunctions","*.c"))
interpolation.remove(os.path.join(ROOT,"Source","InterpolationFunctions","InterpolationFunctions.c"))
interpolation.remove(os.path.join(ROOT,"Source","InterpolationFunctions","InterpolationFunctionsF16.c"))

#modulesrc = glob.glob(os.path.join("cmsisdsp_pkg","src","*.c"))
modulesrc = []
modulesrc.append(os.path.join("cmsisdsp_pkg","src","cmsismodule.c"))

allsrcs = support + fastmath + filtering + matrix + statistics + complexf + basic
allsrcs = allsrcs + controller + transform + modulesrc + common+ interpolation

def notf16(number):
  if re.search(r'f16',number):
     return(False)
  if re.search(r'F16',number):
     return(False)
  return(True)

# If there are too many files, the linker command is failing on Windows.
# So f16 functions are removed since they are not currently available in the wrapper.
# A next version will have to structure this wrapper more cleanly so that the
# build can work even with more functions
srcs = list(filter(notf16, allsrcs))

module1 = Extension(config.extensionName,
                    sources = (srcs
                              )
                              ,
                    include_dirs =  includes + [numpy.get_include()],
                    #extra_compile_args = ["-Wno-unused-variable","-Wno-implicit-function-declaration",config.cflags]
                    extra_compile_args = cflags
                              )

setup (name = config.setupName,
       version = '0.0.1',
       description = config.setupDescription,
       ext_modules = [module1],
       author = 'Copyright (C) 2010-2020 ARM Limited or its affiliates. All rights reserved.',
       url="https://github.com/ARM-software/CMSIS_5",
       classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ])
