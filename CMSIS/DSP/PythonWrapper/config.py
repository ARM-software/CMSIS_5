CMSISDSP = 1

ROOT=".."

config = CMSISDSP

if config == CMSISDSP:
    extensionName = 'internal' 
    setupName = 'CMSISDSP'
    setupDescription = 'CMSIS-DSP Python API'
    cflags="-DCMSISDSP"

