cmake_minimum_required (VERSION 3.6)
cmake_policy(SET CMP0077 NEW)
project(CMSISNN)

include(config)
# Needed to find the config modules
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP)


# Select which parts of the CMSIS-DSP must be compiled.
# There are some dependencies between the parts but they are not tracked
# by this cmake. So, enabling some functions may require to enable some
# other ones.
option(FULLYCONNECTED        "Fully Connected"              ON)
option(CONVOLUTION           "Convolutions"              ON)
option(ACTIVATION           "Activations"              ON)
option(POOLING           "Pooling"              ON)
option(SOFTMAX           "Softmax"              ON)

# When OFF it is the default behavior : all tables are included.
option(NNSUPPORT            "NN Support"    ON)


###########################
#
# CMSIS NN
#
###########################

# DSP Sources
SET(NN ${ROOT}/CMSIS/NN)

list(APPEND CMAKE_MODULE_PATH ${NN}/Source)

add_library(CMSISNN INTERFACE)

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



