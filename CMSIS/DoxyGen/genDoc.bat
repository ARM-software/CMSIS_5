@echo off

REM ====================================================================================
REM Batch file for generating
REM
REM Author  : 
REM Date    :  17th February 2016
REM Version : 1.0
REM Company : ARM 
REM
REM 
REM Command syntax: genDoc.bat
REM
REM  Version: 1.0 Initial Version.
REM ====================================================================================

SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

REM -- Delete previous generated HTML files ---------------------
  ECHO.
  ECHO Delete previous generated HTML files

IF EXIST DoxyGen\Core\html (
  rmdir /S /Q ..\Documentation\Core\html
)
IF EXIST DoxyGen\Driver\html (
  rmdir /S /Q ..\Documentation\Driver\html
)
IF EXIST DoxyGen\General\html (
  rmdir /S /Q ..\Documentation\General\html
)
IF EXIST DoxyGen\Pack\html (
  rmdir /S /Q ..\Documentation\Pack\html
)
IF EXIST DoxyGen\SVD\html (
  rmdir /S /Q ..\Documentation\SVD\html
)

REM -- Generate New HTML Files ---------------------
  ECHO.
  ECHO Generate New HTML Files

pushd Core
CALL doxygen_core.bat
popd

pushd Driver
CALL doxygen_driver.bat
popd

pushd General
CALL doxygen_general.bat
popd

pushd Pack
CALL doxygen_pack.bat
popd

pushd SVD
CALL doxygen_svd.bat
popd

REM -- Copy search style sheet ---------------------
ECHO Copy search style sheets
copy /Y Doxygen_Templates\search.css ..\Documentation\CORE\html\search\. 
copy /Y Doxygen_Templates\search.css ..\Documentation\Driver\html\search\.
REM copy /Y Doxygen_Templates\search.css ..\Documentation\General\html\search\. 
copy /Y Doxygen_Templates\search.css ..\Documentation\Pack\html\search\.
REM copy /Y Doxygen_Templates\search.css ..\Documentation\SVD\html\search\.
  
:END
  ECHO.
REM done
