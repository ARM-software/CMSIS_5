#! python

import sys
import os
from argparse import ArgumentParser
from datetime import datetime
from subprocess import call, Popen
from pathlib import Path

sys.path.append('buildutils') 

from rtecmd import RteCmd 
from uv4cmd import Uv4Cmd 
from dscmd import DsCmd 
from fvpcmd import FvpCmd 
from iarcmd import IarCmd 
from testresult import TestResult

try:
  rtebuildhome = os.path.join(os.path.abspath(Path.home()), ".rtebuild")
  sys.path.append(rtebuildhome)
  import buildenv
except ModuleNotFoundError:
  print("No custom buildenv.py found in "+rtebuildhome)
  buildenv = None
  pass

DEVICE_CM0     = 'Cortex-M0'
DEVICE_CM0PLUS = 'Cortex-M0plus'
DEVICE_CM3     = 'Cortex-M3'
DEVICE_CM4     = 'Cortex-M4'
DEVICE_CM4FP   = 'Cortex-M4FP'
DEVICE_CM7     = 'Cortex-M7'
DEVICE_CM7SP   = 'Cortex-M7SP'
DEVICE_CM7DP   = 'Cortex-M7DP'
DEVICE_CM23    = 'Cortex-M23'
DEVICE_CM33    = 'Cortex-M33'
DEVICE_CM23NS  = 'Cortex-M23NS'
DEVICE_CM33NS  = 'Cortex-M33NS'
DEVICE_CM23S   = 'Cortex-M23S'
DEVICE_CM33S   = 'Cortex-M33S'
DEVICE_CA5     = 'Cortex-A5'
DEVICE_CA7     = 'Cortex-A7'
DEVICE_CA9     = 'Cortex-A9'
DEVICE_CA5NEON = 'Cortex-A5neon'
DEVICE_CA7NEON = 'Cortex-A7neon'
DEVICE_CA9NEON = 'Cortex-A9neon'

CC_AC6    = 'AC6'
CC_AC6LTM = 'AC6LTM'
CC_AC5    = 'AC5'
CC_GCC    = 'GCC'
CC_IAR    = 'IAR'

TARGET_FVP = 'FVP'

ADEVICES = {
    DEVICE_CM0     : 'CM0',
    DEVICE_CM0PLUS : 'CM0plus',
    DEVICE_CM3     : 'CM3',
    DEVICE_CM4     : 'CM4',
    DEVICE_CM4FP   : 'CM4FP',
    DEVICE_CM7     : 'CM7',
    DEVICE_CM7SP   : 'CM7SP',
    DEVICE_CM7DP   : 'CM7DP',
    DEVICE_CM23    : 'CM23',
    DEVICE_CM33    : 'CM33',
    DEVICE_CM23S   : 'CM23S',
    DEVICE_CM33S   : 'CM33S',
    DEVICE_CM33NS  : 'CM33NS',
    DEVICE_CM23NS  : 'CM23NS',
    DEVICE_CA5     : 'CA5',
    DEVICE_CA7     : 'CA7',
    DEVICE_CA9     : 'CA9',
    DEVICE_CA5NEON : 'CA5neon',
    DEVICE_CA7NEON : 'CA7neon',
    DEVICE_CA9NEON : 'CA9neon'
  }

DEVICES = [ DEVICE_CM0, DEVICE_CM0PLUS, DEVICE_CM3, DEVICE_CM4, DEVICE_CM4FP, DEVICE_CM7, DEVICE_CM7SP, DEVICE_CM7DP, DEVICE_CM23, DEVICE_CM33, DEVICE_CM23NS, DEVICE_CM33NS, DEVICE_CM23S, DEVICE_CM33S, DEVICE_CA5, DEVICE_CA7, DEVICE_CA9, DEVICE_CA5NEON, DEVICE_CA7NEON, DEVICE_CA9NEON ]
COMPILERS = [ CC_AC5, CC_AC6, CC_AC6LTM, CC_GCC, CC_IAR ]
TARGETS = [ TARGET_FVP ]

SKIP = [ 
    [ DEVICE_CM23,    CC_AC5,    None ],
    [ DEVICE_CM33,    CC_AC5,    None ],
    [ DEVICE_CM23NS,  CC_AC5,    None ],
    [ DEVICE_CM33NS,  CC_AC5,    None ],
    [ DEVICE_CM23S,   CC_AC5,    None ],
    [ DEVICE_CM33S,   CC_AC5,    None ],
    [ DEVICE_CM0,     CC_AC6LTM, None ],
    [ DEVICE_CM0PLUS, CC_AC6LTM, None ],
    [ DEVICE_CM23,    CC_AC6LTM, None ],
    [ DEVICE_CM33,    CC_AC6LTM, None ],
    [ DEVICE_CM23NS,  CC_AC6LTM, None ],
    [ DEVICE_CM33NS,  CC_AC6LTM, None ],
    [ DEVICE_CM23S,   CC_AC6LTM, None ],
    [ DEVICE_CM33S,   CC_AC6LTM, None ],
  ]
  
FVP_MODELS = { 
    DEVICE_CM0      : { 'cmd': "FVP_MPS2_Cortex-M0_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM0_config.txt" } },
    DEVICE_CM0PLUS  : { 'cmd': "FVP_MPS2_Cortex-M0plus_MDK.exe",  'args': { 'limit': "50000000", 'config': "ARMCM0plus_config.txt" } },
    DEVICE_CM3      : { 'cmd': "FVP_MPS2_Cortex-M3_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM3_config.txt" } },
    DEVICE_CM4      : { 'cmd': "FVP_MPS2_Cortex-M4_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM4_config.txt" } },
    DEVICE_CM4FP    : { 'cmd': "FVP_MPS2_Cortex-M4_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM4FP_config.txt" } },
    DEVICE_CM7      : { 'cmd': "FVP_MPS2_Cortex-M7_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM7_config.txt" } },
    DEVICE_CM7SP    : { 'cmd': "FVP_MPS2_Cortex-M7_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM7SP_config.txt" } },
    DEVICE_CM7DP    : { 'cmd': "FVP_MPS2_Cortex-M7_MDK.exe",      'args': { 'limit': "50000000", 'config': "ARMCM7DP_config.txt" } },
    DEVICE_CM23     : { 'cmd': "FVP_MPS2_Cortex-M23_MDK.exe",     'args': { 'limit': "50000000", 'config': "ARMCM23_config.txt",           'target': "cpu0" } },
    DEVICE_CM33     : { 'cmd': "FVP_MPS2_Cortex-M33_MDK.exe",     'args': { 'limit': "50000000", 'config': "ARMCM33_config.txt",           'target': "cpu0" } },
    DEVICE_CM23NS   : { 'cmd': "FVP_MPS2_Cortex-M23_MDK.exe",     'args': { 'limit': "50000000", 'config': "ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    DEVICE_CM33NS   : { 'cmd': "FVP_MPS2_Cortex-M33_MDK.exe",     'args': { 'limit': "50000000", 'config': "ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } },
    DEVICE_CM23S    : { 'cmd': "FVP_MPS2_Cortex-M23_MDK.exe",     'args': { 'limit': "50000000", 'config': "ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    DEVICE_CM33S    : { 'cmd': "FVP_MPS2_Cortex-M33_MDK.exe",     'args': { 'limit': "50000000", 'config': "ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } },
    DEVICE_CA5      : { 'cmd': "fvp_ve_cortex-a5x1.exe",          'args': { 'limit': "70000000", 'config': "ARMCA5_config.txt" } },
    DEVICE_CA7      : { 'cmd': "fvp_ve_cortex-a7x1.exe",          'args': { 'limit': "170000000", 'config': "ARMCA7_config.txt" } },
    DEVICE_CA9      : { 'cmd': "fvp_ve_cortex-a9x1.exe",          'args': { 'limit': "70000000", 'config': "ARMCA9_config.txt" } },
    DEVICE_CA5NEON  : { 'cmd': "fvp_ve_cortex-a5x1.exe",          'args': { 'limit': "70000000", 'config': "ARMCA5neon_config.txt" } },
    DEVICE_CA7NEON  : { 'cmd': "fvp_ve_cortex-a7x1.exe",          'args': { 'limit': "170000000", 'config': "ARMCA7neon_config.txt" } },
    DEVICE_CA9NEON  : { 'cmd': "fvp_ve_cortex-a9x1.exe",          'args': { 'limit': "70000000", 'config': "ARMCA9neon_config.txt" } }
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
  rtebuild = "{dev}/{cc}/default.rtebuild".format(dev = dev, cc = cc, target=target)
  if os.path.exists(rtebuild):
    return [
        rtebuild,
        "{dev}/{cc}/build/{target}/{target}.elf".format(dev = dev, cc = cc, target=target)
      ]
  elif (cc == CC_AC5) or (cc == CC_AC6):
    return [
        "{dev}/{cc}/CMSIS_CV.uvprojx".format(dev = dev, cc = cc),
        "{dev}/{cc}/Objects/CMSIS_CV.axf".format(dev = dev, cc = cc)
      ]
  elif (cc == CC_AC6LTM):
    return [
        "{dev}/{cc}/CMSIS_CV.uvprojx".format(dev = dev, cc = CC_AC6),
        "{dev}/{cc}/Objects/CMSIS_CV.axf".format(dev = dev, cc = CC_AC6)
      ]
  elif (cc == CC_GCC):
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
  rtebuild = "{dev}/{cc}/Bootloader/default.rtebuild".format(dev = dev, cc = cc, target=target)
  if os.path.exists(rtebuild):
    return [
        rtebuild,
        "{dev}/{cc}/Bootloader/build/{target}/{target}.elf".format(dev = dev, cc = cc, target=target)
      ]
  elif (cc == CC_AC5) or (cc == CC_AC6):
    return [
        "{dev}/{cc}/Bootloader/Bootloader.uvprojx".format(dev = dev, cc = cc),
        "{dev}/{cc}/Bootloader/Objects/Bootloader.axf".format(dev = dev, cc = cc)
      ] 
  elif (cc == CC_AC6LTM):
    return [
        "{dev}/{cc}/Bootloader/Bootloader.uvprojx".format(dev = dev, cc = CC_AC6),
        "{dev}/{cc}/Bootloader/Objects/Bootloader.axf".format(dev = dev, cc = CC_AC6)
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
  
def buildStep(dev, cc, target, project, env=os.environ):
  STEP_TYPES = {
    ".uvprojx"  : Uv4Cmd,
    ".ewp"      : IarCmd,
    ".rtebuild" : RteCmd
  }
  
  projectfile, projectext = os.path.splitext(project)
  
  if not projectext in STEP_TYPES:
    raise "Unknown project type '"+projectext+"'!"
    
  return STEP_TYPES[projectext](project, target, env)
  
def prepare(steps, args):
  for dev in args.devices:
    for cc in args.compilers:
      for target in args.targets:
        if not isSkipped(dev, cc, target):
          config = "{dev} ({cc}, {target})".format(dev = dev, cc = cc, target = target)
          prefix = "{dev}_{cc}_{target}".format(dev = dev, cc = cc, target = target)
          
          env = os.environ
          if hasattr(buildenv, "env_"+cc):
            env = getattr(buildenv, "env_"+cc)()

          rv = testProject(dev, cc, target)
          build = [ buildStep(dev, cc, target, rv[0], env=env) ]
          binary = [ rv[1] ]
          
          bl = bootloaderProject(dev, cc, target)
          if os.path.isfile(bl[0]):
            build = [ buildStep(dev, cc, target, bl[0], env=env) ] + build
            binary = [ bl[1] ] + binary

          if target == TARGET_FVP:
            test = FvpCmd(FVP_MODELS[dev]['cmd'], binary, **FVP_MODELS[dev]['args'])
          steps += [ { 'name': config, 'prefix': prefix, 'build': build, 'test': test } ]

def execute(steps, args):
  for step in steps:
    print(step['name'])
    if step['build'] and not args.execute_only:
      for b in step['build']:
        b.run()
    else:
      print("Skipping build")
      # step['build'].skip()
      
    if step['test'] and not args.build_only:
      step['test'].run()
      step['result'] = TestResult(step['test'].getOutput())
      step['result'].saveXml("result_{0}_{1}.xml".format(step['prefix'], datetime.now().strftime("%Y%m%d%H%M%S")))
    else:
      print("Skipping test")
      step['test'].skip()
      
def printSummary(steps):
  print("")
  print("Test Summary")
  print("============")
  print()
  print("Test run                       Total Exec  Pass  Fail  ")
  print("-------------------------------------------------------")
  for step in steps:
    try:
      print("{0:30} {1:>4}  {2:>4}  {3:>4}  {4:>4}".format(step['name'], *step['result'].getSummary()))
    except:
      print("{0:30} ------ NO RESULTS ------".format(step['name']))

def main(argv):
  parser = ArgumentParser()
  parser.add_argument('--genconfig', action='store_true')
  parser.add_argument('-b', '--build-only', action='store_true')
  parser.add_argument('-e', '--execute-only', action='store_true')
  parser.add_argument('-d', '--devices', nargs='*', choices=(DEVICES+list(ADEVICES.values())), default=DEVICES, help = 'Devices to be considered.')
  parser.add_argument('-c', '--compilers', nargs='*', choices=COMPILERS, default=COMPILERS, help = 'Compilers to be considered.')
  parser.add_argument('-t', '--targets', nargs='*', choices=TARGETS, default=TARGETS, help = 'Targets to be considered.')
  args = parser.parse_args()
  
  devices = set()
  for d in args.devices:
    if d in ADEVICES.values():
      d = list(ADEVICES.keys())[list(ADEVICES.values()).index(d)]
    devices.add(d)
  
  args.devices = list(devices)
    
  if args.genconfig:
    for dev in args.devices:
      model = FVP_MODELS[dev]
      cmd = [ model['cmd'], '-l', '-o', model['args']['config'] ]
      print(" ".join(cmd))
      call(cmd)
    return 1
    
  steps = []

  prepare(steps, args)
  
  execute(steps, args)
  
  printSummary(steps)
  
if __name__ == "__main__":
  main(sys.argv[1:])