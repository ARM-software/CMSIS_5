SET(COMPILERVERSION "")

if (ARMAC6)
    include(Toolchain/AC6)
endif()

if (ARMAC5)
    include(Toolchain/AC5)
endif()

if (GCC)
    include(Toolchain/GCC)
endif()

if ((MSVC) OR (HOST))
    function(compilerSpecificCompileOptions PROJECTNAME ROOT)
    endfunction()
endif()