cmake_minimum_required (VERSION 3.6)

project(CMSISNNConvolutions)

include(configLib)

file(GLOB SRC "./*_s8*.c")
add_library(CMSISNNConvolutions STATIC ${SRC})

configLib(CMSISNNConvolutions ${ROOT})
configDsp(CMSISNNConvolutions ${ROOT})

### Includes
target_include_directories(CMSISNNConvolutions PUBLIC "${NN}/Include")
target_include_directories(CMSISNNConvolutions PUBLIC "${ROOT}/CMSIS/DSP/Include")



