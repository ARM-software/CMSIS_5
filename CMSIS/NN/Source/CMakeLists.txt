cmake_minimum_required (VERSION 3.6)
cmake_policy(SET CMP0077 NEW)
project(CMSISNN)

# Needed to find the config modules
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP)
# CMSIS-NN is dependent on features in the CMSIS-DSP
# and must use similar logic for configuring the build
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP/Source)

include(config)
include(configDsp)

# Select which parts of the CMSIS-DSP must be compiled.
# There are some dependencies between the parts but they are not tracked
# by this cmake. So, enabling some functions may require to enable some
# other ones.
option(FULLYCONNECTED       "Fully Connected"       ON)
option(CONVOLUTION          "Convolutions"          ON)
option(ACTIVATION           "Activations"           ON)
option(POOLING              "Pooling"               ON)
option(SOFTMAX              "Softmax"               ON)
option(BASICMATHSNN         "Basic Maths for NN"    ON)

# When OFF it is the default behavior : all tables are included.
option(NNSUPPORT            "NN Support"            ON)


###########################
#
# CMSIS NN
#
###########################

# NN Sources
SET(NN ${ROOT}/CMSIS/NN)

list(APPEND CMAKE_MODULE_PATH ${NN}/Source)




add_library(CMSISNN INTERFACE)

if (BASICMATHSNN)
  add_subdirectory(BasicMathFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNBasicMaths)
endif()

if (FULLYCONNECTED)
  add_subdirectory(FullyConnectedFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNFullyConnected)
endif()

if (CONVOLUTION)
  add_subdirectory(ConvolutionFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNConvolutions)
endif()

if (ACTIVATION)
  add_subdirectory(ActivationFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNActivation)
endif()

if (POOLING)
  add_subdirectory(PoolingFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNPooling)
endif()

if (SOFTMAX)
  add_subdirectory(SoftmaxFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNSoftmax)
endif()

if (NNSUPPORT)
  add_subdirectory(NNSupportFunctions)
  target_link_libraries(CMSISNN INTERFACE CMSISNNSupport)
endif()

### Includes
target_include_directories(CMSISDSP INTERFACE "${NN}/Include")



