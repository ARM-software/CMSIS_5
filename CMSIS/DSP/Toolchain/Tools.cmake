SET(COMPILERVERSION "")
if (ARMAC6)
    include(Toolchain/AC6)
endif()

if (GCC)
    include(Toolchain/GCC)
endif()