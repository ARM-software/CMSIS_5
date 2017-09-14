#! python

import sys
from argparse import ArgumentParser
from datetime import datetime

sys.path.append('../../../Utilities/buildutils') 

from iarcmd import IarCmd 
from fvpcmd import FvpCmd 
from testresult import TestResult

DEVICE_CM0  = 'Cortex-M0'
DEVICE_CM3  = 'Cortex-M3'
DEVICE_CM4  = 'Cortex-M4'
DEVICE_CM7  = 'Cortex-M7'
DEVICE_CM23 = 'Cortex-M23'
DEVICE_CM33 = 'Cortex-M33'

DEVICES = [ DEVICE_CM0, DEVICE_CM3, DEVICE_CM4, DEVICE_CM7, DEVICE_CM23, DEVICE_CM33 ]

SKIP = [ 
  ]
  
FVP_MODELS = { 
    DEVICE_CM0  : { 'cmd': "fvp_mps2_cortex-m0.exe",  'args': { 'limit': "2000000" } },
    DEVICE_CM3  : { 'cmd': "fvp_mps2_cortex-m3.exe",  'args': { 'limit': "2000000" } },
    DEVICE_CM4  : { 'cmd': "fvp_mps2_cortex-m4.exe",  'args': { 'limit': "5000000" } },
    DEVICE_CM7  : { 'cmd': "fvp_mps2_cortex-m7.exe",  'args': { 'limit': "5000000" } },
    DEVICE_CM23 : { 'cmd': "fvp_mps2_cortex-m23.exe", 'args': { 'limit': "5000000", 'config': "ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    DEVICE_CM33 : { 'cmd': "fvp_mps2_cortex-m33.exe", 'args': { 'limit': "5000000", 'config': "ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } }
  }

def isSkipped(dev):
  for skip in SKIP:
    skipDev = (skip[0] == None or skip[0] == dev)
    if skipDev and skipCc and skipTarget:
      return True
  return False

def prepare(steps, args):
  for dev in args.devices:
    if not isSkipped(dev):
      if args.execute_only:
        build = None
      else:
        build = IarCmd(dev+"/CMSIS_CV.ewp", "Debug")
      if args.build_only:
        test = None
      else:
        test = FvpCmd(FVP_MODELS[dev]['cmd'], dev+"/Debug/Exe/CMSIS_CV.out", **FVP_MODELS[dev]['args'])
      steps += [ { 'name': dev, 'prefix': dev, 'build': build, 'test': test } ]

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
  args = parser.parse_args()
    
  steps = []

  prepare(steps, args)
  
  execute(steps)
  
  printSummary(steps)
  
if __name__ == "__main__":
  main(sys.argv[1:])