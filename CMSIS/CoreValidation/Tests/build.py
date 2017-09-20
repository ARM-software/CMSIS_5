#! python

import sys
import os.path
from argparse import ArgumentParser
from datetime import datetime
from subprocess import call, Popen

sys.path.append('buildutils') 

from uv4cmd import Uv4Cmd 
from fvpcmd import FvpCmd 
from iarcmd import IarCmd 
from testresult import TestResult

DEVICE_CM0     = 'Cortex-M0'
DEVICE_CM3     = 'Cortex-M3'
DEVICE_CM4     = 'Cortex-M4'
DEVICE_CM4FP   = 'Cortex-M4FP'
DEVICE_CM7     = 'Cortex-M7'
DEVICE_CM7SP   = 'Cortex-M7SP'
DEVICE_CM7DP   = 'Cortex-M7DP'
DEVICE_CM23    = 'Cortex-M23'
DEVICE_CM33    = 'Cortex-M33'
DEVICE_CA5     = 'Cortex-A5'
DEVICE_CA7     = 'Cortex-A7'
DEVICE_CA9     = 'Cortex-A9'
DEVICE_CA5NEON = 'Cortex-A5neon'
DEVICE_CA7NEON = 'Cortex-A7neon'
DEVICE_CA9NEON = 'Cortex-A9neon'

CC_AC6 = 'AC6'
CC_AC5 = 'AC5'
CC_GCC = 'GCC'
CC_IAR = 'IAR'

MDK_ENV = {
  'uVision' : [ DEVICE_CM0, DEVICE_CM3, DEVICE_CM4, DEVICE_CM4FP, DEVICE_CM7, DEVICE_CM7SP, DEVICE_CM7DP, DEVICE_CM23, DEVICE_CM33 ],
  'DS' : [ DEVICE_CA5, DEVICE_CA7, DEVICE_CA9, DEVICE_CA5NEON, DEVICE_CA7NEON, DEVICE_CA9NEON ]
}

TARGET_FVP = 'FVP'

ADEVICES = {
    DEVICE_CM0     : 'CM0',
    DEVICE_CM3     : 'CM3',
    DEVICE_CM4     : 'CM4',
    DEVICE_CM4FP   : 'CM4FP',
    DEVICE_CM7     : 'CM7',
    DEVICE_CM7SP   : 'CM7SP',
    DEVICE_CM7DP   : 'CM7DP',
    DEVICE_CM23    : 'CM23',
    DEVICE_CM33    : 'CM33',
    DEVICE_CA5     : 'CA5',
    DEVICE_CA7     : 'CA7',
    DEVICE_CA9     : 'CA9',
    DEVICE_CA5NEON : 'CA5neon',
    DEVICE_CA7NEON : 'CA7neon',
    DEVICE_CA9NEON : 'CA9neon'
  }

DEVICES = [ DEVICE_CM0, DEVICE_CM3, DEVICE_CM4, DEVICE_CM4FP, DEVICE_CM7, DEVICE_CM7SP, DEVICE_CM7DP, DEVICE_CM23, DEVICE_CM33, DEVICE_CA5, DEVICE_CA7, DEVICE_CA9, DEVICE_CA5NEON, DEVICE_CA7NEON, DEVICE_CA9NEON ]
COMPILERS = [ CC_AC5, CC_AC6, CC_GCC, CC_IAR ]
TARGETS = [ TARGET_FVP ]

SKIP = [ 
    [ DEVICE_CM23, CC_AC5, None ],
    [ DEVICE_CM33, CC_AC5, None ],
  ]
  
FVP_MODELS = { 
    DEVICE_CM0      : { 'cmd': "fvp_mps2_cortex-m0.exe",  'args': { 'limit': "50000000", 'config': "ARMCM0_config.txt" } },
    DEVICE_CM3      : { 'cmd': "fvp_mps2_cortex-m3.exe",  'args': { 'limit': "50000000", 'config': "ARMCM3_config.txt" } },
    DEVICE_CM4      : { 'cmd': "fvp_mps2_cortex-m4.exe",  'args': { 'limit': "50000000", 'config': "ARMCM4_config.txt" } },
    DEVICE_CM4FP    : { 'cmd': "fvp_mps2_cortex-m4.exe",  'args': { 'limit': "50000000", 'config': "ARMCM4FP_config.txt" } },
    DEVICE_CM7      : { 'cmd': "fvp_mps2_cortex-m7.exe",  'args': { 'limit': "50000000", 'config': "ARMCM7_config.txt" } },
    DEVICE_CM7SP    : { 'cmd': "fvp_mps2_cortex-m7.exe",  'args': { 'limit': "50000000", 'config': "ARMCM7SP_config.txt" } },
    DEVICE_CM7DP    : { 'cmd': "fvp_mps2_cortex-m7.exe",  'args': { 'limit': "50000000", 'config': "ARMCM7DP_config.txt" } },
    DEVICE_CM23     : { 'cmd': "fvp_mps2_cortex-m23.exe", 'args': { 'limit': "50000000", 'config': "ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    DEVICE_CM33     : { 'cmd': "fvp_mps2_cortex-m33.exe", 'args': { 'limit': "50000000", 'config': "ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } },
    DEVICE_CA5      : { 'cmd': "fvp_ve_cortex-a5x1.exe",  'args': { 'limit': "70000000", 'config': "ARMCA5_config.txt" } },
    DEVICE_CA7      : { 'cmd': "fvp_ve_cortex-a7x1.exe",  'args': { 'limit': "170000000", 'config': "ARMCA7_config.txt" } },
    DEVICE_CA9      : { 'cmd': "fvp_ve_cortex-a9x1.exe",  'args': { 'limit': "70000000", 'config': "ARMCA9_config.txt" } },
    DEVICE_CA5NEON  : { 'cmd': "fvp_ve_cortex-a5x1.exe",  'args': { 'limit': "70000000", 'config': "ARMCA5neon_config.txt" } },
    DEVICE_CA7NEON  : { 'cmd': "fvp_ve_cortex-a7x1.exe",  'args': { 'limit': "170000000", 'config': "ARMCA7neon_config.txt" } },
    DEVICE_CA9NEON  : { 'cmd': "fvp_ve_cortex-a9x1.exe",  'args': { 'limit': "70000000", 'config': "ARMCA9neon_config.txt" } }
  }

def isSkipped(dev, cc, target):
  for skip in SKIP:
    skipDev = (skip[0] == None or skip[0] == dev)
    skipCc = (skip[1] == None or skip[1] == cc)
    skipTarget = (skip[2] == None or skip[2] == target)
    if skipDev and skipCc and skipTarget:
      return True
  return False
  
def testProject(dev, cc, target):
  if (cc == CC_AC5) or (cc == CC_AC6):
    if dev in MDK_ENV['DS']:
      return [
          "{dev}/{cc}/.project".format(dev = dev, cc = cc),
          "{dev}/{cc}/Debug/CMSIS_CV_{adev}_{cc}.axf".format(dev = dev, adev=ADEVICES[dev], cc = cc)
        ]
    else:
      return [
          "{dev}/{cc}/CMSIS_CV.uvprojx".format(dev = dev, cc = cc),
          "{dev}/{cc}/Objects/CMSIS_CV.axf".format(dev = dev, cc = cc)
        ]
  elif (cc == CC_GCC):
    if dev in MDK_ENV['DS']:
      return [
          "{dev}/{cc}/.project".format(dev = dev, cc = cc),
          "{dev}/{cc}/Debug/CMSIS_CV_{adev}_{cc}.elf".format(dev = dev, adev=ADEVICES[dev], cc = cc)
        ]
    else:
      return [
          "{dev}/{cc}/CMSIS_CV.uvprojx".format(dev = dev, cc = cc),
          "{dev}/{cc}/Objects/CMSIS_CV.elf".format(dev = dev, cc = cc)
        ]
  elif (cc == CC_IAR):
    return [
        "{dev}/{cc}/CMSIS_CV.ewp".format(dev = dev, cc = cc),
        "{dev}/{cc}/{target}/Exe/CMSIS_CV.out".format(dev = dev, cc = cc, target = target)
      ]
  raise "Unknown compiler!"

def bootloaderProject(dev, cc, target):
  if (cc == CC_AC5) or (cc == CC_AC6):
    return [
        "{dev}/{cc}/Bootloader/Bootloader.uvprojx".format(dev = dev, cc = cc),
        "{dev}/{cc}/Bootloader/Objects/Bootloader.axf".format(dev = dev, cc = cc)
      ] 
  elif (cc == CC_GCC):
    return [
        "{dev}/{cc}/Bootloader/Bootloader.uvprojx".format(dev = dev, cc = cc),
        "{dev}/{cc}/Bootloader/Objects/Bootloader.elf".format(dev = dev, cc = cc)
      ] 
  elif (cc == CC_IAR):
    return [
        "{dev}/{cc}/Bootloader/Bootloader.ewp".format(dev = dev, cc = cc),
        "{dev}/{cc}/Bootloader/{target}/Exe/Bootloader.out".format(dev = dev, cc = cc, target = target)
      ]
  raise "Unknown compiler!"
  
def buildStep(dev, cc, target, project):
  if (cc == CC_AC5) or (cc == CC_AC6):
    return Uv4Cmd(project, target)
  elif (cc == CC_GCC):
    return Uv4Cmd(project, target)
  elif (cc == CC_IAR):
    return IarCmd(project, target)
  raise "Unknown compiler!"
  
def prepare(steps, args):
  for dev in args.devices:
    for cc in args.compilers:
      for target in args.targets:
        if not isSkipped(dev, cc, target):
          config = "{dev} ({cc}, {target})".format(dev = dev, cc = cc, target = target)
          prefix = "{dev}_{cc}_{target}".format(dev = dev, cc = cc, target = target)
          
          rv = testProject(dev, cc, target)
          build = [ buildStep(dev, cc, target, rv[0]) ]
          binary = [ rv[1] ]
          
          bl = bootloaderProject(dev, cc, target)
          if os.path.isfile(bl[0]):
            build = [ buildStep(dev, cc, target, bl[0]) ] + build
            binary = [ bl[1] ] + binary

          if target == TARGET_FVP:
            test = FvpCmd(FVP_MODELS[dev]['cmd'], binary, **FVP_MODELS[dev]['args'])
          steps += [ { 'name': config, 'prefix': prefix, 'build': build, 'test': test } ]

def execute(steps, args):
  for step in steps:
    print step['name']
    if step['build'] and not args.execute_only:
      for b in step['build']:
        b.run()
    else:
      print "Skipping build"
      # step['build'].skip()
      
    if step['test'] and not args.build_only:
      step['test'].run()
      step['result'] = TestResult(step['test'].getOutput())
      step['result'].saveXml("result_{0}_{1}.xml".format(step['prefix'], datetime.now().strftime("%Y%m%d%H%M%S")))
    else:
      print "Skipping test"
      step['test'].skip()
      
def printSummary(steps):
  print ""
  print "Test Summary"
  print "============"
  print
  print "Test run                       Total Exec  Pass  Fail  "
  print "-------------------------------------------------------"
  for step in steps:
    try:
      print "{0:30} {1:>4}  {2:>4}  {3:>4}  {4:>4}".format(step['name'], *step['result'].getSummary())
    except:
      print "{0:30} ------ NO RESULTS ------".format(step['name'])

def main(argv):
  parser = ArgumentParser()
  parser.add_argument('--genconfig', action='store_true')
  parser.add_argument('-b', '--build-only', action='store_true')
  parser.add_argument('-e', '--execute-only', action='store_true')
  parser.add_argument('-d', '--devices', nargs='*', choices=DEVICES, default=DEVICES, help = 'Devices to be considered.')
  parser.add_argument('-c', '--compilers', nargs='*', choices=COMPILERS, default=COMPILERS, help = 'Compilers to be considered.')
  parser.add_argument('-t', '--targets', nargs='*', choices=TARGETS, default=TARGETS, help = 'Targets to be considered.')
  args = parser.parse_args()
    
  if args.genconfig:
    for dev in args.devices:
      model = FVP_MODELS[dev]
      cmd = [ model['cmd'], '-l', '-o', model['args']['config'] ]
      print " ".join(cmd)
      call(cmd)
    return 1
    
  steps = []

  prepare(steps, args)
  
  execute(steps, args)
  
  printSummary(steps)
  
if __name__ == "__main__":
  main(sys.argv[1:])