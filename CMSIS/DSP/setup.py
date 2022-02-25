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

#distance = glob.glob(os.path.join(ROOT,"Source","DistanceFunctions","*.c"))
#distance.remove(os.path.join(ROOT,"Source","DistanceFunctions","DistanceFunctions.c"))


# Add dependencies
transformMod = transform + common + basic + complexf + fastmath + matrix + statistics
statisticsMod = statistics + common + fastmath
interpolationMod = interpolation + common
filteringMod = filtering + common + support + fastmath
controllerMod = controller + common

matrixMod = matrix 
supportMod = support 
complexfMod = complexf + fastmath + common
basicMod = basic
quaternionMod = quaternion
fastmathMod = fastmath + common


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




missing=set(["arm_abs_f64"
,"arm_absmax_f64"
,"arm_absmax_no_idx_f32"
,"arm_absmax_no_idx_f64"
,"arm_absmin_f64"
,"arm_absmin_no_idx_f32"
,"arm_absmin_no_idx_f64"
,"arm_add_f64"
,"arm_barycenter_f32"
,"arm_braycurtis_distance_f32"
,"arm_canberra_distance_f32"
,"arm_chebyshev_distance_f32"
,"arm_chebyshev_distance_f64"
,"arm_circularRead_f32"
,"arm_cityblock_distance_f32"
,"arm_cityblock_distance_f64"
,"arm_cmplx_mag_f64"
,"arm_cmplx_mag_squared_f64"
,"arm_cmplx_mult_cmplx_f64"
,"arm_copy_f64"
,"arm_correlate_f64"
,"arm_correlation_distance_f32"
,"arm_cosine_distance_f32"
,"arm_cosine_distance_f64"
,"arm_dot_prod_f64"
,"arm_entropy_f32"
,"arm_entropy_f64"
,"arm_euclidean_distance_f32"
,"arm_euclidean_distance_f64"
,"arm_exponent_f32"
,"arm_fill_f64"
,"arm_fir_f64"
,"arm_fir_init_f64"
,"arm_gaussian_naive_bayes_predict_f32"
,"arm_jensenshannon_distance_f32"
,"arm_kullback_leibler_f32"
,"arm_kullback_leibler_f64"
,"arm_logsumexp_dot_prod_f32"
,"arm_logsumexp_f32"
,"arm_mat_cholesky_f32"
,"arm_mat_cholesky_f64"
,"arm_mat_init_f32"
,"arm_mat_ldlt_f32"
,"arm_mat_ldlt_f64"
,"arm_mat_mult_f64"
,"arm_mat_solve_lower_triangular_f32"
,"arm_mat_solve_lower_triangular_f64"
,"arm_mat_solve_upper_triangular_f32"
,"arm_mat_solve_upper_triangular_f64"
,"arm_mat_sub_f64"
,"arm_mat_trans_f64"
,"arm_max_f64"
,"arm_max_no_idx_f32"
,"arm_max_no_idx_f64"
,"arm_mean_f64"
,"arm_merge_sort_f32"
,"arm_merge_sort_init_f32"
,"arm_min_f64"
,"arm_min_no_idx_f32"
,"arm_min_no_idx_f64"
,"arm_minkowski_distance_f32"
,"arm_mult_f64"
,"arm_negate_f64"
,"arm_offset_f64"
,"arm_power_f64"
,"arm_scale_f64"
,"arm_sort_f32"
,"arm_sort_init_f32"
,"arm_spline_f32"
,"arm_spline_init_f32"
,"arm_std_f64"
,"arm_sub_f64"
,"arm_svm_linear_init_f32"
,"arm_svm_linear_predict_f32"
,"arm_svm_polynomial_init_f32"
,"arm_svm_polynomial_predict_f32"
,"arm_svm_rbf_init_f32"
,"arm_svm_rbf_predict_f32"
,"arm_svm_sigmoid_init_f32"
,"arm_svm_sigmoid_predict_f32"
,"arm_var_f64"
,"arm_vexp_f32"
,"arm_vexp_f64"
,"arm_vlog_f64"
,"arm_vsqrt_f32"
,"arm_weighted_sum_f32"
,"arm_circularRead_q15"
,"arm_circularRead_q7"
,"arm_div_q63_to_q31"
,"arm_fir_sparse_q15"
,"arm_fir_sparse_q31"
,"arm_fir_sparse_q7"
,"arm_mat_init_q15"
,"arm_mat_init_q31"
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

#for l in filtering:
#  print(os.path.basename(l))
#quit()

def mkModule(name,srcs):
  return(Extension(name,
                    sources = (srcs
                              )
                              ,
                    include_dirs =  includes + [numpy.get_include()],
                    extra_compile_args = cflags
                              ))

moduleFiltering = mkModule('cmsisdsp_filtering',filtering)
moduleMatrix = mkModule('cmsisdsp_matrix',matrix)
moduleSupport = mkModule('cmsisdsp_support',support)
moduleStatistics = mkModule('cmsisdsp_statistics',statistics)
moduleComplexf= mkModule('cmsisdsp_complexf',complexf)
moduleBasic = mkModule('cmsisdsp_basic',basic)
moduleController = mkModule('cmsisdsp_controller',controller)
moduleTransform = mkModule('cmsisdsp_transform',transform)
moduleInterpolation = mkModule('cmsisdsp_interpolation',interpolation)
moduleQuaternion = mkModule('cmsisdsp_quaternion',quaternion)
moduleFastmath = mkModule('cmsisdsp_fastmath',fastmath)




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
                        moduleFastmath
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
          'jinja2>= 3.0',
          'sympy>=1.6'],
          project_urls={  # Optional
             'Bug Reports': 'https://github.com/ARM-software/CMSIS_5/issues',
             'Source': 'https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP',
            }
          )
       

build()