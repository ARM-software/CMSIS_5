cmake_minimum_required (VERSION 3.6)

project(CMSISNNPooling)

include(configLib)

file(GLOB SRC "./*_s8.c")

add_library(CMSISNNPooling STATIC ${SRC})

configLib(CMSISNNPooling ${ROOT})
configDsp(CMSISNNPooling ${ROOT})

### Includes
target_include_directories(CMSISNNPooling PUBLIC "${NN}/Include")
target_include_directories(CMSISNNPooling PUBLIC "${ROOT}/CMSIS/DSP/Include")



