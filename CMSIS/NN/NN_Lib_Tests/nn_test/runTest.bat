@echo off

set UVEXE=C:\Keil_v5\UV4\UV4.EXE

echo Building NN Test for Cortex-M7 Single Precision
%UVEXE% -rb -j0 arm_nnexamples_nn_test.uvprojx -t "ARMCM7_SP" -o "NN_Test_ARMCM7_SP_build.log"

echo Running NN Test for Cortex-M7 Single Precision
%UVEXE% -d arm_nnexamples_nn_test.uvprojx -t "ARMCM7_SP" -j0

type NN_TEST.log
del NN_TEST.log

echo Building NN Test for Cortex-M4 FP
%UVEXE% -rb -j0 arm_nnexamples_nn_test.uvprojx -t "ARMCM4_FP" -o "NN_Test_ARMCM4_FP_build.log"

echo Running NN Test for Cortex-M4 FP
%UVEXE% -d arm_nnexamples_nn_test.uvprojx -t "ARMCM4_FP" -j0

type NN_TEST.log
del NN_TEST.log

echo Building NN Test for Cortex-M3
%UVEXE% -rb -j0 arm_nnexamples_nn_test.uvprojx -t "ARMCM3" -o "NN_Test_ARMCM3_build.log"

echo Running NN Test for Cortex-M3
%UVEXE% -d arm_nnexamples_nn_test.uvprojx -t "ARMCM3" -j0

type NN_TEST.log
del NN_TEST.log