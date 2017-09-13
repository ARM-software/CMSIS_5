#! python

import sys
from argparse import ArgumentParser
from datetime import datetime

sys.path.append('../../../Utilities/buildutils') 

from fvpcmd import FvpCmd 
from testresult import TestResult

DEVICE_A5  = 'Cortex-A5'
DEVICE_A7  = 'Cortex-A7'
DEVICE_A9  = 'Cortex-A9'

CC_AC6 = 'AC6'
CC_AC5 = 'AC5'
CC_GCC = 'GCC'

TARGET_FVP = 'FVP'

DEVICES = [ DEVICE_A5, DEVICE_A7, DEVICE_A9 ]
COMPILERS = [ CC_AC5, CC_AC6, CC_GCC ]
TARGETS = [ TARGET_FVP ]

SKIP = [ 
  ]

DEVICE_ABREV = {
  DEVICE_A5 : 'CA5',
  DEVICE_A7 : 'CA7',
  DEVICE_A9 : 'CA9'
}
  
APP_FORMAT = {
  CC_AC6: "axf",
  CC_AC5: "axf",
  CC_GCC: "elf"
}
  
FVP_MODELS = { 
    DEVICE_A5   : { 'cmd': "fvp_ve_cortex-a5x1.exe",  'args': { 'limit': "5000000", 'config': "ARMCA5_config.txt" } },
    DEVICE_A7   : { 'cmd': "fvp_ve_cortex-a7x1.exe",  'args': { 'limit': "5000000", 'config': "ARMCA7_config.txt" } },
    DEVICE_A9   : { 'cmd': "fvp_ve_cortex-a9x1.exe",  'args': { 'limit': "5000000", 'config': "ARMCA9_config.txt" } }
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
          build = None
          binary = "{dev}/{cc}/Debug/CMSIS_CV_{abrev}_{cc}.{format}".format(dev = dev, abrev = DEVICE_ABREV[dev], cc = cc, format = APP_FORMAT[cc])
          test = FvpCmd(FVP_MODELS[dev]['cmd'], binary, **FVP_MODELS[dev]['args'])
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