# Version: 1.0 
# Date: 2021-06-23
# Author: Junchang Liu (liujunchang97@outlook.com)
# This powershell script generates a CMSIS Software Pack:
#
# Pre-requisites:
# - PackChk in path with execute permission
#   (see CMSIS-Pack: CMSIS/Utilities/<os>/PackChk)
# - powershell 7.0.0 or higher
#   https://github.com/PowerShell/PowerShell/releases

############### EDIT BELOW ###############
# Extend Path environment variable locally

$CMSIS_VERSION = "5.7.0"
$CMSIS_PACK_PATH = "$env:LOCALAPPDATA\Arm\Packs\ARM\CMSIS\$CMSIS_VERSION"
$env:Path += ";$CMSIS_PACK_PATH\CMSIS\Utilities\Win32"

# Specify file names to be added to pack base directory
$PACK_BASE_FILES = "License.txt", "README.md"

# Pack warehouse directory - destination 
$PACK_WAREHOUSE = "output"

# Temporary pack build directory
$PACK_BUILD = "build"

# Specify directories included in pack relative to base directory
# All directories:
$PACK_DIRS = Get-ChildItem -Directory -Exclude $PACK_WAREHOUSE, $PACK_BUILD

############ DO NOT EDIT BELOW ###########
$ErrorActionPreference = "Stop"

Write-Output "Starting CMSIS-Pack Generation: $(Get-Date)"

try {
  Get-Command PackChk > $null
}
catch {
  Write-Output @"
Error: No PackChk Utility found
Action: Add PackChk to your path
<pack_root_dir>/ARM/CMSIS/<version>/CMSIS/Utilities/<os>/
"@
  exit -1
}

$PACK_DESCRIPTION_FILE = Get-ChildItem -Name -Filter *.pdsc
$PDSC_NUM = (Get-ChildItem -Filter *.pdsc|Measure-Object).Count
if ($PDSC_NUM -eq 0) {
  Write-Output "Error: No *.pdsc file found in current directory"
  exit -1
} elseif ($PDSC_NUM -gt 1) {
  Write-Output @"
Error: Only one PDSC file allowed in directory structure:
Found: $PACK_DESCRIPTION_FILE
Action: Delete unused pdsc files
"@
  exit -1
}

$PACK_VENDOR, $PACK_NAME = $PACK_DESCRIPTION_FILE.ToString().Split(".")[0, 1]

#if $PACK_BUILD directory does not exist, create it.
New-Item -ItemType Directory -Force $PACK_BUILD > $null

# Copy files into build base directory: $PACK_BUILD
# pdsc file is mandatory in base directory:
Copy-Item -Force "$PACK_VENDOR.$PACK_NAME.pdsc" -Destination $PACK_BUILD

# directories
Copy-Item -Force $PACK_DIRS -Destination $PACK_BUILD -Recurse

# files for base directory
Copy-Item -Force $PACK_BASE_FILES -Destination $PACK_BUILD -Recurse

# Run Schema Check:
try {
  [xml]$PDSC_XML = Get-Content "$PACK_BUILD\$PACK_VENDOR.$PACK_NAME.pdsc"
  $PDSC_XML.Schemas.Add("", "$CMSIS_PACK_PATH\CMSIS\Utilities\PACK.xsd") > $null
  $PDSC_XML.Validate($null)
}
catch {
  Write-Output "build aborted: Schema check of $PACK_VENDOR.$PACK_NAME.pdsc against PACK.xsd failed"
  exit -1
}


# Run Pack Check and generate PackName file with version
PackChk "$PACK_BUILD\$PACK_VENDOR.$PACK_NAME.pdsc" -n PackName.txt -x M362

[string]$PACKNAME = Get-Content PackName.txt

Write-Output $PACKNAME

# Archiving
New-Item -ItemType Directory -Force $PACK_WAREHOUSE > $null

try {
  $compress = @{
    Path = Get-ChildItem -Path $PACK_BUILD
    CompressionLevel = "Optimal"
    DestinationPath = "$PACK_WAREHOUSE\$PACKNAME"
  }
  Compress-Archive -Force @compress
}
catch {
  Write-Output "build aborted: archiving failed"
  exit -1
}

Write-Output "build of pack succeeded"

# Clean up
Write-Output "cleaning up ..."

Remove-Item -Recurse $PACK_BUILD

Write-Output "Completed CMSIS-Pack Generation: $(Get-Date)"
