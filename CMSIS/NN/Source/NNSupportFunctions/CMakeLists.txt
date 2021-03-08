cmake_minimum_required (VERSION 3.6)

project(CMSISNNSupport)

include(configLib)

file(GLOB SRC "./*_s8.c")
add_library(CMSISNNSupport STATIC ${SRC})
target_sources(CMSISNNSupport PUBLIC arm_q7_to_q15_with_offset.c)

configLib(CMSISNNSupport ${ROOT})
configDsp(CMSISNNSupport ${ROOT})

### Includes
target_include_directories(CMSISNNSupport PUBLIC "${NN}/Include")
target_include_directories(CMSISNNSupport PUBLIC "${ROOT}/CMSIS/DSP/Include")



