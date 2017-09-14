:: Batch file for generating CMSIS pack
:: This batch file uses:
::    7-Zip for packaging
::    Doxygen version 1.8.2 and Mscgen version 0.20 for generating html documentation.
:: The generated pack and pdsc file are placed in folder %RELEASE_PATH% (../../Local_Release)
@ECHO off

SETLOCAL

:: Tool path for zipping tool 7-Zip
SET ZIPPATH=C:\Program Files\7-Zip

:: Tool path for doxygen
SET DOXYGENPATH=C:\Program Files\doxygen\bin

:: Tool path for mscgen utility
SET MSCGENPATH=C:\Program Files (x86)\Mscgen

:: These settings should be passed on to subprocesses as well
SET PATH=%ZIPPATH%;%DOXYGENPATH%;%MSCGENPATH%;%PATH%

:: Pack Path (where generated pack is stored)
SET RELEASE_PATH=..\..\Local_Release

:: !!!!!!!!!!!!!!!!!
:: DO NOT EDIT BELOW
:: !!!!!!!!!!!!!!!!! 

:: Remove previous build
IF EXIST %RELEASE_PATH% (
  ECHO removing %RELEASE_PATH%
  RMDIR /Q /S  %RELEASE_PATH%
)

:: Create build output directory
MKDIR %RELEASE_PATH%


:: Copy PDSC file
COPY ARM.CMSIS-Core_Validation.pdsc %RELEASE_PATH%\ARM.CMSIS-Core_Validation.pdsc

:: Copy LICENSE file
COPY ..\..\LICENSE.txt %RELEASE_PATH%\LICENSE.txt

:: Copy folders
XCOPY /Q /S /Y Examples\*.* %RELEASE_PATH%\Examples\*.*
XCOPY /Q /S /Y Include\*.* %RELEASE_PATH%\Include\*.*
XCOPY /Q /S /Y Source\*.* %RELEASE_PATH%\Source\*.*

:: Checking 
..\Utilities\Win32\PackChk.exe %RELEASE_PATH%\ARM.CMSIS-Core_Validation.pdsc -i ..\..\ARM.CMSIS.pdsc -n %RELEASE_PATH%\PackName.txt -x M353 -x M364

:: --Check if PackChk.exe has completed successfully
IF %errorlevel% neq 0 GOTO ErrPackChk

:: Packing 
PUSHD %RELEASE_PATH%

:: -- Pipe Pack's Name into Variable
SET /P PackName=<PackName.txt
DEL /Q PackName.txt

:: Pack files
ECHO Creating pack file ...
7z.exe a %PackName% -tzip > zip.log
ECHO Packaging complete
POPD
GOTO End

:ErrPackChk
ECHO PackChk.exe has encountered an error!
EXIT /b

:End
ECHO Removing temporary files and folders
RMDIR /Q /S  %RELEASE_PATH%\CMSIS
RMDIR /Q /S  %RELEASE_PATH%\Device
DEL %RELEASE_PATH%\LICENSE.txt
DEL %RELEASE_PATH%\zip.log

ECHO gen_pack.bat completed successfully
