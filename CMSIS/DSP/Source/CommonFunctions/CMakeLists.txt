cmake_minimum_required (VERSION 3.6)

project(CMSISDSPCommonFunctions)

include(configLib)
include(configDsp)

file(GLOB SRC "./*_*.c")

add_library(CMSISDSPCommonFunctions STATIC ${SRC})

configLib(CMSISDSPCommonFunctions ${ROOT})
configDsp(CMSISDSPCommonFunctions ${ROOT})

### Includes
target_include_directories(CMSISDSPCommonFunctions PUBLIC "${DSP}/Include")



