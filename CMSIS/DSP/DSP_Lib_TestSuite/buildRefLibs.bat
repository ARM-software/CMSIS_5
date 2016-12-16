@echo off

SET TMP=C:\Temp
SET TEMP=C:\Temp
SET UVEXE=C:\Keil\UV4\UV4.EXE
set CURDIR=%CD%

if .%1==. goto help
for %%a in (ARM GCC) do if %1==%%a goto startBuild
goto help

:startBuild
echo.
echo Building DSP Reference Libraries %1

if %1==ARM                goto buildARM
if %1==GCC                goto buildGCC
goto err

:buildARM
:buildGCC
cd .\RefLibs\%1
REM @echo on
echo   Building DSP Reference Library for Cortex-M0 Little Endian
%UVEXE% -rb -j0  RefLibs.uvprojx -t "cortexM0l"    -o "RefLib_cortexM0l_build.log"

echo   Building DSP Reference Library for Cortex-M3 Little Endian
%UVEXE% -rb -j0  RefLibs.uvprojx -t "cortexM3l"    -o "RefLib_cortexM3l_build.log"

echo   Building DSP Reference Library for Cortex-M4 Little Endian
%UVEXE% -rb -j0  RefLibs.uvprojx -t "cortexM4l"    -o "RefLib_cortexM4l_build.log"

echo   Building DSP Reference Library for Cortex-M4 with FPU Little Endian
%UVEXE% -rb -j0  RefLibs.uvprojx -t "cortexM4lf"   -o "RefLib_cortexM4lf_build.log"

echo   Building DSP Reference Library for Cortex-M7 Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "cortexM7l"     -o "RefLib_cortexM7l_build.log"

echo   Building DSP Reference Library for Cortex-M7 with single precision FPU Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "cortexM7lfsp"  -o "RefLib_cortexM7lfsp_build.log"

echo   Building DSP Reference Library for Cortex-M7 with double precision FPU Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "cortexM7lfdp"  -o "RefLib_cortexM7lfdp_build.log"

echo   Building DSP Reference Library for ARMv8-M Baseline Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MBLl"     -o "RefLib_ARMv8MBLl_build.log"

echo   Building DSP Reference Library for ARMv8-M Mainline Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MMLl"     -o "RefLib_ARMv8MMLl_build.log"

echo   Building DSP Reference Library for ARMv8-M Mainline with single precision FPU Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MMLlfsp"  -o "RefLib_ARMv8MMLlfsp_build.log"

echo   Building DSP Reference Library for ARMv8-M Mainline with double precision FPU Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MMLlfdp"  -o "RefLib_ARMv8MMLlfdp_build.log"

echo   Building DSP Reference Library for ARMv8-M Mainline with DSP Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MMLld"    -o "RefLib_ARMv8MMLld_build.log"

echo   Building DSP Reference Library for ARMv8-M Mainline with DSP, single precision FPU Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MMLldfsp" -o "RefLib_ARMv8MMLldfsp_build.log"

echo   Building DSP Reference Library for ARMv8-M Mainline with DSP, double precision FPU Little Endian
%UVEXE% -rb -j0 RefLibs.uvprojx -t "ARMv8MMLldfdp" -o "RefLib_ARMv8MMLldfdp_build.log"
REM @echo off

REM big endian libraries are skipped!

REM echo   Building DSP Reference Library for Cortex-M0 Big Endian
REM %UVEXE% -rb -j0  RefLibs.uvprojx -t"cortexM0b"    -o "RefLib_cortexM0b_build.log"

REM echo   Building DSP Reference Library for Cortex-M3 Big Endian
REM %UVEXE% -rb -j0  RefLibs.uvprojx -t"cortexM3b"    -o "RefLib_cortexM3b_build.log"

REM echo   Building DSP Reference Library for Cortex-M4 Big Endian
REM %UVEXE% -rb -j0  RefLibs.uvprojx -t"cortexM4b"    -o "RefLib_cortexM4b_build.log"

REM echo   Building DSP Reference Library for Cortex-M4 with FPU Big Endian
REM %UVEXE% -rb -j0  RefLibs.uvprojx -t"cortexM4bf"   -o "RefLib_cortexM4bf_build.log"

REM echo   Building DSP Reference Library for Cortex-M7 Big Endian
REM %UVEXE% -rb -j0 RefLibs.uvprojx -t "cortexM7b"    -o "RefLib_cortexM7b_build.log"

REM echo   Building DSP Reference Library for Cortex-M7 with single precision FPU Big Endian
REM %UVEXE% -rb -j0 RefLibs.uvprojx -t "cortexM7bfsp" -o "RefLib_cortexM7bfsp_build.log"

REM echo   Building DSP Reference Library for Cortex-M7 with double precision FPU Big Endian
REM %UVEXE% -rb -j0 RefLibs.uvprojx -t "cortexM7bfdp" -o "RefLib_cortexM7bfdp_build.log"

goto deleteIntermediateFiles


:deleteIntermediateFiles
echo.
ECHO   Deleting intermediate files
rmdir /S /Q IntermediateFiles
del /Q *.bak
del /Q *.dep
del /Q *.uvguix.*
del /Q ArInp.*

goto changeDir


:changeDir
cd %CURDIR%
goto end

:err

:help
echo   Syntax: buildRefLibs toolchain
echo.
echo     toolchain:     ARM ^| GCC
echo.
echo   e.g.: buildRefLibs ARM

:end
