function(configDsp project root)

if (CONFIGTABLE)
    # Public because initialization for FFT may be defined in client code 
    # and needs access to the table.
    target_compile_definitions(${project} PUBLIC ARM_DSP_CONFIG_TABLES)
endif()

if (LOOPUNROLL)
  target_compile_definitions(${project} PRIVATE ARM_MATH_LOOPUNROLL)
endif()

if (ROUNDING)
  target_compile_definitions(${project} PRIVATE ARM_MATH_ROUNDING)
endif()

if (MATRIXCHECK)
  target_compile_definitions(${project} PRIVATE ARM_MATH_MATRIX_CHECK)
endif()

endfunction()