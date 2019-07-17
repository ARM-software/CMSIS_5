cmake_minimum_required (VERSION 3.6)

project(CMSISDSPSVM)

include(config)
include(configDsp)

file(GLOB SRC "./*_*.c")

add_library(CMSISDSPSVM STATIC ${SRC})

configLib(CMSISDSPSVM ${ROOT})
configDsp(CMSISDSPSVM ${ROOT})

### Includes
target_include_directories(CMSISDSPSVM PUBLIC "${DSP}/Include")


if (NEON OR NEONEXPERIMENTAL)
    target_sources(CMSISDSPSVM PRIVATE "${DSP}/ComputeLibrary/Source/arm_cl_tables.c")
endif()

