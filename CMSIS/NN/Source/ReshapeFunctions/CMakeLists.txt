cmake_minimum_required (VERSION 3.6)

project(CMSISNNReshape)

include(config)

file(GLOB SRC "./*_*.c")

add_library(CMSISNNReshape STATIC ${SRC})

configLib(CMSISNNReshape ${ROOT})
configDsp(CMSISNNReshape ${ROOT})