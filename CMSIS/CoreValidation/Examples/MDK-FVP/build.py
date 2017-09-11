#! python

from subprocess import call
from xml.etree import ElementTree
import os.path

print "Build CMSIS-Core Validation using MDK"

TARGET_FVP = 'FVP'
CC_AC6 = 'AC6'
CC_AC5 = 'AC5'
CC_GCC = 'GCC'

UV4 = "UV4.exe"
PRJ = "CMSIS_CV.uvprojx"
# DEVICES = [ 'Cortex-M0', 'Cortex-M3', 'Cortex-M4f', 'Cortex-M7', 'Cortex-M23', 'Cortex-M33' ]
# DEVICES = [ 'Cortex-M0', 'Cortex-M3', 'Cortex-M4f', 'Cortex-M7' ]
DEVICES = [ 'Cortex-M4f' ]
COMPILERS = [ CC_AC5, CC_AC6, CC_GCC ]
TARGETS = [ TARGET_FVP ]

FVP_MODELS = { 
    'Cortex-M0'  : [ "fvp_mps2_cortex-m0.exe", "--cyclelimit",   "2000000", "" ],
    'Cortex-M3'  : [ "fvp_mps2_cortex-m3.exe", "--cyclelimit",   "2000000", "" ],
    'Cortex-M4f' : [ "fvp_mps2_cortex-m4.exe", "--cyclelimit",   "5000000", "" ],
    'Cortex-M7'  : [ "fvp_mps2_cortex-m7.exe", "--cyclelimit",   "5000000", "" ],
    'Cortex-M23' : [ "fvp_mps2_cortex-m23.exe", "--cyclelimit", "10000000", "-f", "ARMCM23_TZ_config.txt", "-a", "cpu0=" ],
    'Cortex-M33' : [ "fvp_mps2_cortex-m33.exe", "--cyclelimit", "10000000", "-f", "ARMCM33_DSP_FP_TZ_config.txt", "-a", "cpu0=" ]
  } 

SKIP = [ 
    ['Cortex-M23', CC_GCC, None ], 
    ['Cortex-M33', CC_GCC, None ] 
  ]

def isSkipped(dev, cc, target):
  for skip in SKIP:
    skipDev = (skip[0] == None or skip[0] == dev)
    skipCc = (skip[1] == None or skip[1] == cc)
    skipTarget = (skip[2] == None or skip[2] == target)
    if skipDev and skipCc and skipTarget:
      return True
  return False

def ret2result(ret):
  if ret == 0:
    return "successfull"
  elif ret == 1:
    return "successfull with warnings"
  else:
    return "failed!"

def binary(cc):
  if cc == CC_GCC:
    return 'Objects/CMSIS_CV.elf'
  else:
    return 'Objects/CMSIS_CV.axf'
    
def build(dev, cc, target):
  print "Building..."
  config = "{dev} ({cc}, {target})".format(dev = dev, cc = cc, target = target)
  log = "build_{dev}_{cc}_{target}.log".format(dev = dev, cc = cc, target = target)
  print "{cmd} -t {config} -r {prj} -j0 -o {log}".format(cmd = UV4, config = config, prj = PRJ, log = log)
  try:
    ret = call([UV4, "-t", config, "-r", PRJ, "-j0", "-o", log])
    print open(log, "r").read()
    print "Build " + ret2result(ret)
    return (ret <= 1)
  except:
    print "Build failed!"
    return False

def run(dev, cc, target):
  print "Running..."
  config = "{dev} ({cc}, {target})".format(dev = dev, cc = cc, target = target)
  log = "run_{dev}_{cc}_{target}.log".format(dev = dev, cc = cc, target = target)
  xml = "result_{dev}_{cc}_{target}.xml".format(dev = dev, cc = cc, target = target)
  
  if target == TARGET_FVP:
    model = FVP_MODELS[dev][:]
    app = binary(cc)
    model[-1] = model[-1] + app
    print ' '.join(model)
    logfile = open(log, "w")
    call(model, stdout=logfile)
    
  logfile = open(log, "r")
  xmlfile = open(xml, "w")
  dump = False
  for line in logfile:
    if dump:
      xmlfile.write(line)
      if line.strip() == "</report>":
        dump = False
    else:
      if line.strip() == "Simulation is started":
        dump = True
      print line,
  
  return True

for dev in DEVICES:
  for cc in COMPILERS:
    for target in TARGETS:
      if not isSkipped(dev, cc, target):
        print ""
        print "{dev} with {cc} on {target}".format(dev = dev, cc = cc, target = target)
        success = build(dev, cc, target)
        if success:
          run(dev, cc, target)

# Test Summary 
print ""
print "Test Summary"
print "============"
print
print "Test run                       Total Exec  Pass  Fail  "
print "-------------------------------------------------------"
for dev in DEVICES:
  for cc in COMPILERS:
    for target in TARGETS:
      name = "{dev} ({cc}, {target})".format(dev = dev, cc = cc, target = target)
      if isSkipped(dev, cc, target):
        print "{0:30} ------- skipped --------".format(name)
      else:
        try:
          xml = "result_{dev}_{cc}_{target}.xml".format(dev = dev, cc = cc, target = target)
          report = ElementTree.parse(xml).getroot()
          summary = report[0].findall('summary')[0]
          tests = summary.find('tcnt').text
          executed = summary.find('exec').text
          passed = summary.find('pass').text
          failed = summary.find('fail').text
          print "{0:30} {1:>4}  {2:>4}  {3:>4}  {4:>4}".format(name, tests, executed, passed, failed)
        except:
          print "{0:30} ------ NO RESULTS ------".format(name)
