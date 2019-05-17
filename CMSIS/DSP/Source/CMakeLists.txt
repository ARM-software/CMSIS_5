cmake_minimum_required (VERSION 3.6)

project(CMSISDSP)

# Needed to find the config module
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)


###########################
#
# CMSIS DSP
#
###########################

# DSP Sources
SET(DSP ".")
SET(COMMON "${DSP}/CommonTables")

file(GLOB SOURCES_STATS "${DSP}/StatisticsFunctions/*_*.c")
file(GLOB SOURCES_BASIC "${DSP}/BasicMathFunctions/*_*.c")
file(GLOB SOURCES_COMPLEX "${DSP}/ComplexMathFunctions/*_*.c")
file(GLOB SOURCES_FAST "${DSP}/FastMathFunctions/*_*.c")
file(GLOB SOURCES_FILTERING "${DSP}/FilteringFunctions/*_*.c")
file(GLOB SOURCES_MATRIX "${DSP}/MatrixFunctions/*_*.c")
file(GLOB SOURCES_STATS "${DSP}/StatisticsFunctions/*_*.c")
file(GLOB SOURCES_SUPPORT "${DSP}/SupportFunctions/*_*.c")
file(GLOB SOURCES_TRANSFORM "${DSP}/TransformFunctions/*_*.c")

SET(CMSISDSPSRC ${SOURCES_MATRIX} ${SOURCES_FILTERING} ${SOURCES_SUPPORT} ${SOURCES_COMPLEX} ${SOURCES_TRANSFORM} ${SOURCES_FAST} ${SOURCES_BASIC} ${SOURCES_STATS})

add_library(CMSISDSP STATIC ${COMMON}/arm_common_tables.c ${COMMON}/arm_const_structs.c)

include(config)

### Sources
target_sources(CMSISDSP PRIVATE ${CMSISDSPSRC})

### Includes
target_include_directories(CMSISDSP PUBLIC "${DSP}/../Include")



