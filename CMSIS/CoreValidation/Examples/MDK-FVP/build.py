#! python

import sys
from argparse import ArgumentParser
from datetime import datetime

sys.path.append('../../../Utilities/buildutils') 

from uv4cmd import Uv4Cmd 
from fvpcmd import FvpCmd 
from testresult import TestResult

DEVICE_CM0  = 'Cortex-M0'
DEVICE_CM3  = 'Cortex-M3'
DEVICE_CM4f = 'Cortex-M4f'
DEVICE_CM7  = 'Cortex-M7'
DEVICE_CM23 = 'Cortex-M23'
DEVICE_CM33 = 'Cortex-M33'

CC_AC6 = 'AC6'
CC_AC5 = 'AC5'
CC_GCC = 'GCC'

TARGET_FVP = 'FVP'

DEVICES = [ DEVICE_CM0, DEVICE_CM3, DEVICE_CM4f, DEVICE_CM7 ]
COMPILERS = [ CC_AC5, CC_AC6, CC_GCC ]
TARGETS = [ TARGET_FVP ]

SKIP = [ 
    [ DEVICE_CM23, CC_GCC, None ],
    [ DEVICE_CM33, CC_GCC, None ]
  ]

APP_FORMAT = {
  CC_AC6: "axf",
  CC_AC5: "axf",
  CC_GCC: "elf"
}
  
FVP_MODELS = { 
    DEVICE_CM0  : { 'cmd': "fvp_mps2_cortex-m0.exe",  'args': { 'limit': "2000000" } },
    DEVICE_CM3  : { 'cmd': "fvp_mps2_cortex-m3.exe",  'args': { 'limit': "2000000" } },
    DEVICE_CM4f : { 'cmd': "fvp_mps2_cortex-m4.exe",  'args': { 'limit': "5000000" } },
    DEVICE_CM7  : { 'cmd': "fvp_mps2_cortex-m7.exe",  'args': { 'limit': "5000000" } },
    DEVICE_CM23 : { 'cmd': "fvp_mps2_cortex-m23.exe", 'args': { 'limit': "5000000", 'config': "ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    DEVICE_CM33 : { 'cmd': "fvp_mps2_cortex-m33.exe", 'args': { 'limit': "5000000", 'config': "ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } }
  }

def isSkipped(dev, cc, target):
  for skip in SKIP:
    skipDev = (skip[0] == None or skip[0] == dev)
    skipCc = (skip[1] == None or skip[1] == cc)
    skipTarget = (skip[2] == None or skip[2] == target)
    if skipDev and skipCc and skipTarget:
      return True
  return False

def prepare(steps, args):
  for dev in args.devices:
    for cc in args.compilers:
      for target in args.targets:
        if not isSkipped(dev, cc, target):
          config = "{dev} ({cc}, {target})".format(dev = dev, cc = cc, target = target)
          prefix = "{dev}_{cc}_{target}".format(dev = dev, cc = cc, target = target)
          if args.execute_only:
            build = None
          else:
            build = Uv4Cmd("CMSIS_CV.uvprojx", config)
          if args.build_only:
            test = None
          else:
            test = FvpCmd(FVP_MODELS[dev]['cmd'], "Objects\CMSIS_CV."+APP_FORMAT[cc], **FVP_MODELS[dev]['args'])
          steps += [ { 'name': config, 'prefix': prefix, 'build': build, 'test': test } ]

def execute(steps):
  for step in steps:
    print step['name']
    if step['build']:
      step['build'].run()
    else:
      print "Skipping build"
      
    if (not step['build']) or step['build'].isSuccess():
      step['test'].run()
      step['result'] = TestResult(step['test'].getOutput())
      step['result'].saveXml("result_{0}_{1}.xml".format(step['prefix'], datetime.now().strftime("%Y%m%d%H%M%S")))
    else:
      print "Skipping test"
      
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
  parser.add_argument('-b', '--build-only', action='store_true')
  parser.add_argument('-e', '--execute-only', action='store_true')
  parser.add_argument('-d', '--devices', nargs='*', choices=DEVICES, default=DEVICES, help = 'Devices to be considered.')
  parser.add_argument('-c', '--compilers', nargs='*', choices=COMPILERS, default=COMPILERS, help = 'Compilers to be considered.')
  parser.add_argument('-t', '--targets', nargs='*', choices=TARGETS, default=TARGETS, help = 'Targets to be considered.')
  args = parser.parse_args()
    
  steps = []

  prepare(steps, args)
  
  execute(steps)
  
  printSummary(steps)
  
if __name__ == "__main__":
  main(sys.argv[1:])