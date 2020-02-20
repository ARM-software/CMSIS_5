import os
import os.path
import subprocess 
import colorama
from colorama import init,Fore, Back, Style
import argparse
from TestScripts.Regression.Commands import *
import yaml
import sys
import itertools
from pathlib import Path


# Small state machine
def updateTestStatus(testStatusForThisBuild,newTestStatus):
    if testStatusForThisBuild == NOTESTFAILED:
        if newTestStatus == NOTESTFAILED:
           return(NOTESTFAILED)
        if newTestStatus == MAKEFAILED:
           return(MAKEFAILED)
        if newTestStatus == TESTFAILED:
          return(TESTFAILED)
    if testStatusForThisBuild == MAKEFAILED:
        if newTestStatus == NOTESTFAILED:
           return(MAKEFAILED)
        if newTestStatus == MAKEFAILED:
           return(MAKEFAILED)
        if newTestStatus == TESTFAILED:
          return(TESTFAILED)
    if testStatusForThisBuild == TESTFAILED:
        if newTestStatus == NOTESTFAILED:
           return(TESTFAILED)
        if newTestStatus == MAKEFAILED:
           return(TESTFAILED)
        if newTestStatus == TESTFAILED:
          return(TESTFAILED)
    if testStatusForThisBuild == FLOWFAILURE:
       return(testStatusForThisBuild)
    if testStatusForThisBuild == CALLFAILURE:
       return(testStatusForThisBuild)

root = Path(os.getcwd()).parent.parent.parent


def cartesian(*somelists):
   r=[]
   for element in itertools.product(*somelists):
       r.append(list(element))
   return(r)

testFailed = 0

init()

parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-i', nargs='?',type = str, default="testrunConfig.yaml",help="Config file")
parser.add_argument('-r', nargs='?',type = str, default=root, help="Root folder")
parser.add_argument('-n', nargs='?',type = int, default=0, help="ID value when launchign in parallel")

args = parser.parse_args()


with open(args.i,"r") as f:
     config=yaml.safe_load(f)

#print(config)

#print(config["IMPLIEDFLAGS"])

flags = config["FLAGS"]
onoffFlags = []
for f in flags:
  onoffFlags.append(["-D" + f +"=ON","-D" + f +"=OFF"])

allConfigs=cartesian(*onoffFlags)

if DEBUGMODE:
   allConfigs=[allConfigs[0]]
   

failedBuild = {}
# Test all builds

folderCreated=False

def logFailedBuild(root,f):
  with open(os.path.join(fullTestFolder(root),"buildStatus_%d.txt" % args.n),"w") as status:
       for build in f:
            s = f[build]
            if s == MAKEFAILED:
               status.write("%s : Make failure\n" % build)
            if s == TESTFAILED:
              status.write("%s : Test failure\n" % build)
            if s == FLOWFAILURE:
              status.write("%s : Flow failure\n" % build)
            if s == CALLFAILURE:
              status.write("%s : Subprocess failure\n" % build)


def buildAndTest(compiler):
    # Run all tests for AC6
    try:
       for core in config['CORES']:
         configNb = 0
         if compiler in config['CORES'][core]:
            for flagConfig in allConfigs:
               folderCreated = False
               configNb = configNb + 1
               buildStr = "build_%s_%s_%d" % (compiler,core,configNb)
               toUnset = None
               toSet = None

               if compiler in config['UNSET']:
                  if core in config['UNSET'][compiler]:
                     toUnset = config['UNSET'][compiler][core]

               if compiler in config['SET']:
                  if core in config['SET'][compiler]:
                     toSet = config['SET'][compiler][core]

               build = BuildConfig(toUnset,toSet,args.r,
                  buildStr,
                  config['COMPILERS'][core][compiler],
                  config['TOOLCHAINS'][compiler],
                  config['CORES'][core][compiler],
                  config["CMAKE"]
                  )
               
               flags = []
               if core in config["IMPLIEDFLAGS"]:
                  flags += config["IMPLIEDFLAGS"][core]
               flags += flagConfig
   
               if compiler in config["IMPLIEDFLAGS"]:
                  flags += config["IMPLIEDFLAGS"][compiler]
       
               build.createFolder()
               # Run all tests for the build
               testStatusForThisBuild = NOTESTFAILED
               try:
                  # This is saving the flag configuration
                  build.createArchive(flags)
   
                  build.createCMake(flags)
                  for test in config["TESTS"]:
                      msg(test["testName"]+"\n")
                      testClass=test["testClass"]
                      test = build.getTest(testClass)
                      fvp = None 
                      if core in config['FVP']:
                        fvp = config['FVP'][core] 
                      newTestStatus = test.runAndProcess(compiler,fvp)
                      testStatusForThisBuild = updateTestStatus(testStatusForThisBuild,newTestStatus)
                      if testStatusForThisBuild != NOTESTFAILED:
                         failedBuild[buildStr] = testStatusForThisBuild
                         # Final script status
                         testFailed = 1
                  build.archiveResults()
               finally:
                   build.cleanFolder()
         else:
           msg("No toolchain %s for core %s" % (compiler,core))
   
    except TestFlowFailure as flow:
         errorMsg("Error flow id %d\n" % flow.errorCode())
         failedBuild[buildStr] = FLOWFAILURE
         logFailedBuild(args.r,failedBuild)
         sys.exit(1)
    except CallFailure: 
         errorMsg("Call failure\n")
         failedBuild[buildStr] = CALLFAILURE
         logFailedBuild(args.r,failedBuild)
         sys.exit(1)

############## Builds for all toolchains

if not DEBUGMODE:
   preprocess()
   generateAllCCode()

for t in config["TOOLCHAINS"]:
    msg("Testing toolchain %s\n" % t)
    buildAndTest(t)

logFailedBuild(args.r,failedBuild)
sys.exit(testFailed)
  