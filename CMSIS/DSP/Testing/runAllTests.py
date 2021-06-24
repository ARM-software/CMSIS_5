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
import sqlite3

# Command to get last runid 
lastID="""SELECT runid FROM RUN ORDER BY runid DESC LIMIT 1
"""

addNewIDCmd="""INSERT INTO RUN VALUES(?,date('now'))
"""

benchID = 0
regID = 0

def getLastRunID(c):
  r=c.execute(lastID)
  result=r.fetchone()
  if result is None:
     return(0)
  else:
     return(int(result[0]))

def addNewID(c,newid):
  print("  New run ID = %d" % newid)
  c.execute(addNewIDCmd,(newid,))
  c.commit()

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

# Analyze the configuration flags (like loopunroll etc ...)
def analyzeFlags(flags):
  
  onoffFlags = []
  for f in flags:
    if type(f) is dict:
      for var in f:
         if type(f[var]) is bool:
            if f[var]:
              onoffFlags.append(["-D%s=ON" % (var)])
            else:
              onoffFlags.append(["-D%s=OFF" % (var)])
         else:   
           onoffFlags.append(["-D%s=%s" % (var,f[var])])
    else:
      onoffFlags.append(["-D" + f +"=ON","-D" + f +"=OFF"])
  
  allConfigs=cartesian(*onoffFlags)
  return(allConfigs)

# Extract the cmake for a specific compiler
# and the flag configuration to use for this compiler.
# This flags configuration will override the global one
def analyzeToolchain(toolchain, globalConfig):
    config=globalConfig
    cmake=""
    sim=True
    if type(toolchain) is str:
       cmake=toolchain 
    else:
       for t in toolchain:
         if type(t) is dict:
            if "FLAGS" in t:
               hasConfig=True 
               config = analyzeFlags(t["FLAGS"])
            if "SIM" in t:
               sim = t["SIM"]
         if type(t) is str:
           cmake=t 
    return(cmake,config,sim)



def cartesian(*somelists):
   r=[]
   for element in itertools.product(*somelists):
       r.append(list(element))
   return(r)

root = Path(os.getcwd()).parent.parent.parent


testFailed = 0

init()

parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-i', nargs='?',type = str, default="testrunConfig.yaml",help="Config file")
parser.add_argument('-r', nargs='?',type = str, default=root, help="Root folder")
parser.add_argument('-n', nargs='?',type = int, default=0, help="ID value when launching in parallel")
parser.add_argument('-b', action='store_true', help="Benchmark mode")
parser.add_argument('-f', nargs='?',type = str, default="desc.txt",help="Test description file")
parser.add_argument('-p', nargs='?',type = str, default="FVP",help="Platform for running")

parser.add_argument('-db', nargs='?',type = str,help="Benchmark database")
parser.add_argument('-regdb', nargs='?',type = str,help="Regression database")
parser.add_argument('-sqlite', nargs='?',default="/usr/bin/sqlite3",type = str,help="sqlite executable")

parser.add_argument('-debug', action='store_true', help="Debug mode")
parser.add_argument('-nokeep', action='store_true', help="Do not keep build folder")

args = parser.parse_args()

if args.debug:
   setDebugMode()

if args.nokeep:
   setNokeepBuildFolder()

# Create missing database files
# if the db arguments are specified
if args.db is not None:
   if not os.path.exists(args.db):
      createDb(args.sqlite,args.db)

   conn = sqlite3.connect(args.db)
   try:
      currentID = getLastRunID(conn)
      benchID = currentID + 1 
      print("Bench db")
      addNewID(conn,benchID)
   finally:
     conn.close()

if args.regdb is not None:
   if not os.path.exists(args.regdb):
      createDb(args.sqlite,args.regdb)

   conn = sqlite3.connect(args.regdb)
   try:
      currentID = getLastRunID(conn)
      regID = currentID + 1 
      print("Regression db")
      addNewID(conn,regID)
   finally:
     conn.close()


with open(args.i,"r") as f:
     config=yaml.safe_load(f)

#print(config)

#print(config["IMPLIEDFLAGS"])


patternConfig={}



if "PATTERNS" in config:
   patternConfig={
    "patterns":os.path.join(config["PATTERNS"],"Patterns")
   ,"parameters":os.path.join(config["PATTERNS"],"Parameters")
   }



flags = config["FLAGS"]
allConfigs = analyzeFlags(flags)

if isDebugMode():
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


def buildAndTest(compiler,theConfig,cmake,sim):
    # Run all tests for AC6
    try:
       for core in config['CORES']:
         configNb = 0
         if compiler in config['CORES'][core]:
            msg("Testing Core %s\n" % core)
            for flagConfig in theConfig:
               folderCreated = False
               configNb = configNb + 1
               buildStr = "build_%s_%s_%d" % (compiler,core,configNb)
               toUnset = None
               toSet = None

               if 'UNSET' in config:
                  if compiler in config['UNSET']:
                     if core in config['UNSET'][compiler]:
                        toUnset = config['UNSET'][compiler][core]

               if 'SET' in config:
                  if compiler in config['SET']:
                     if core in config['SET'][compiler]:
                        toSet = config['SET'][compiler][core]

               build = BuildConfig(toUnset,toSet,args.r,
                  buildStr,
                  config['COMPILERS'][core][compiler],
                  cmake,
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
                  msg("Config " + str(flagConfig) + "\n")

                  build.createCMake(core,flags,args.b,args.p)
                  for test in config["TESTS"]:
                      msg(test["testName"]+"\n")
                      testClass=test["testClass"]
                      test = build.getTest(testClass)
                      fvp = None 
                      if 'FVP' in config:
                        if core in config['FVP']:
                           fvp = config['FVP'][core] 
                      if 'SIM' in config:
                        if core in config['SIM']:
                           fvp = config['SIM'][core] 
                      newTestStatus = test.runAndProcess(patternConfig,compiler,fvp,sim,args.b,args.db,args.regdb,benchID,regID)
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

if not isDebugMode():
   preprocess(args.f)
   generateAllCCode(patternConfig)
else:
   msg("Debug Mode\n")

for t in config["TOOLCHAINS"]:
    cmake,localConfig,sim = analyzeToolchain(config["TOOLCHAINS"][t],allConfigs)
    msg("Testing toolchain %s\n" % t)
    buildAndTest(t,localConfig,cmake,sim)


logFailedBuild(args.r,failedBuild)
sys.exit(testFailed)
  