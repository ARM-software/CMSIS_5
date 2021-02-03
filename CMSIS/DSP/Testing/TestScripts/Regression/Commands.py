import subprocess 
import colorama
from colorama import init,Fore, Back, Style
import argparse
import os
import os.path
from contextlib import contextmanager
import shutil
import glob
from pathlib import Path
import sys

DEBUGMODE = False
KEEPBUILDFOLDER = True

DEBUGLIST=[
"-DBASICMATH=ON",
"-DCOMPLEXMATH=OFF",
"-DCONTROLLER=OFF",
"-DFASTMATH=OFF",
"-DFILTERING=OFF",
"-DMATRIX=OFF",
"-DSTATISTICS=OFF",
"-DSUPPORT=OFF",
"-DTRANSFORM=OFF",
"-DSVM=OFF",
"-DBAYES=OFF",
"-DDISTANCE=OFF"]

NOTESTFAILED = 0
MAKEFAILED = 1 
TESTFAILED = 2
FLOWFAILURE = 3 
CALLFAILURE = 4

def setDebugMode():
  global DEBUGMODE
  DEBUGMODE=True

def isDebugMode():
  global DEBUGMODE
  return(DEBUGMODE)

def setNokeepBuildFolder():
  global KEEPBUILDFOLDER
  KEEPBUILDFOLDER=False

def isKeepMode():
  global KEEPBUILDFOLDER
  return(KEEPBUILDFOLDER)


def joinit(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

def addToDb(db,desc,runid):
    msg("Add %s to summary database\n" % desc)
    completed = subprocess.run([sys.executable, "addToDB.py","-o",db,"-r",str(runid),desc], timeout=3600)
    check(completed)

def addToRegDb(db,desc,runid):
    msg("Add %s to regression database\n" % desc)
    completed = subprocess.run([sys.executable, "addToRegDB.py","-o",db,"-r",str(runid),desc], timeout=3600)
    check(completed)


class TestFlowFailure(Exception):
    def __init__(self,completed):
        self._errorcode = completed.returncode 

    def errorCode(self):
      return(self._errorcode)

class CallFailure(Exception):
    pass

def check(n):
  #print(n)
  if n is not None:
    if n.returncode != 0:
       raise TestFlowFailure(n)
  else: 
    raise CallFailure()

def msg(t):
    print(Fore.CYAN + t + Style.RESET_ALL)

def errorMsg(t):
    print(Fore.RED + t + Style.RESET_ALL)

def fullTestFolder(rootFolder):
    return(os.path.join(rootFolder,"CMSIS","DSP","Testing","fulltests"))

class BuildConfig:
    def __init__(self,toUnset,toSet,rootFolder,buildFolder,compiler,toolchain,core,cmake):
        self._toUnset = toUnset
        self._toSet = toSet
        self._buildFolder = buildFolder
        self._rootFolder = os.path.abspath(rootFolder)
        self._dspFolder = os.path.join(self._rootFolder,"CMSIS","DSP")
        self._testingFolder = os.path.join(self._dspFolder,"Testing")
        self._fullTests = os.path.join(self._testingFolder,"fulltests")
        self._compiler = compiler
        self._toolchain = toolchain
        self._core = core
        self._cmake = cmake
        self._savedEnv = {}

    def compiler(self):
        return(self._compiler)

    def toolChainFile(self):
        return(self._toolchain)

    def core(self):
        return(self._core)

    def path(self):
        return(os.path.join(self._fullTests,self._buildFolder))

    def archivePath(self):
        return(os.path.join(self._fullTests,"archive",self._buildFolder))

    def archiveResultPath(self):
        return(os.path.join(self._fullTests,"archive",self._buildFolder,"results"))

    def archiveLogPath(self):
        return(os.path.join(self._fullTests,"archive",self._buildFolder,"logs"))

    def archiveErrorPath(self):
        return(os.path.join(self._fullTests,"archive",self._buildFolder,"errors"))

    def toolChainPath(self):
        return(self._dspFolder)

    def cmakeFilePath(self):
        return(self._testingFolder)

    def buildFolderName(self):
        return(self._buildFolder)

    def saveEnv(self):
      if self._toUnset is not None:
         for v in self._toUnset:
            if v in os.environ:
               self._savedEnv[v] = os.environ[v]
            else:
               self._savedEnv[v] = None
            del os.environ[v]

      if self._toSet is not None:
         for v in self._toSet:
            for varName in v:
                if varName in os.environ:
                   self._savedEnv[varName] = os.environ[varName]
                else:
                   self._savedEnv[varName] = None
                os.environ[varName] = v[varName]



    def restoreEnv(self):
          for v in self._savedEnv:
            if self._savedEnv[v] is not None:
              os.environ[v] = self._savedEnv[v]
            else:
              if v in os.environ:
                del os.environ[v]
          self._savedEnv = {}

    # Build for a folder
    # We need to be able to detect failed build
    def build(self,test):
        completed=None
        # Save and unset some environment variables
        self.saveEnv()
        with self.buildFolder() as b:
           msg("  Build %s\n" % self.buildFolderName())
           with open(os.path.join(self.archiveLogPath(),"makelog_%s.txt" % test),"w") as makelog:
               with open(os.path.join(self.archiveErrorPath(),"makeerror_%s.txt" % test),"w") as makeerr:
                    if DEBUGMODE:
                       completed=subprocess.run(["make","-j4","VERBOSE=1"],timeout=3600)
                    else:
                       completed=subprocess.run(["make","-j4","VERBOSE=1"],stdout=makelog,stderr=makeerr,timeout=3600)
        # Restore environment variables
        self.restoreEnv()
        check(completed)

    def getTest(self,test):
        return(Test(self,test))

    

    # Launch cmake command.
    def createCMake(self,configid,flags,benchMode,platform):
        with self.buildFolder() as b:
            self.saveEnv()
            if benchMode:
              msg("Benchmark mode\n")
            msg("Create cmake for %s\n" % self.buildFolderName())
            toolchainCmake = os.path.join(self.toolChainPath(),self.toolChainFile())
            cmd = [self._cmake]
            cmd += ["-DCMAKE_PREFIX_PATH=%s" % self.compiler(),
                             "-DCMAKE_TOOLCHAIN_FILE=%s" % toolchainCmake,
                             "-DARM_CPU=%s" % self.core(),
                             "-DPLATFORM=%s" % platform,
                             "-DCONFIGID=%s" % configid
                    ]
            cmd += flags 

            if DEBUGMODE:
              cmd += DEBUGLIST

            if benchMode:
              cmd += ["-DBENCHMARK=ON"]
              cmd += ["-DWRAPPER=ON"]
            else: 
              cmd += ["-DBENCHMARK=OFF"]
              cmd += ["-DWRAPPER=OFF"]

            cmd += ["-DROOT=%s" % self._rootFolder,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-G", "Unix Makefiles" ,"%s" % self.cmakeFilePath()]

            if DEBUGMODE:
               print(cmd)

            with open(os.path.join(self.archiveLogPath(),"cmakecmd.txt"),"w") as cmakecmd:
                 cmakecmd.write("".join(joinit(cmd," ")))

            with open(os.path.join(self.archiveLogPath(),"cmakelog.txt"),"w") as cmakelog:
               with open(os.path.join(self.archiveErrorPath(),"cmakeerror.txt"),"w") as cmakeerr:
                    completed=subprocess.run(cmd, stdout=cmakelog,stderr=cmakeerr, timeout=3600)
            self.restoreEnv()
            check(completed)


    # Create the build folder if missing
    def createFolder(self):
        os.makedirs(self.path(),exist_ok=True)

    def createArchive(self, flags):
        os.makedirs(self.archivePath(),exist_ok=True)
        os.makedirs(self.archiveResultPath(),exist_ok=True)
        os.makedirs(self.archiveErrorPath(),exist_ok=True)
        os.makedirs(self.archiveLogPath(),exist_ok=True)
        with open(os.path.join(self.archivePath(),"flags.txt"),"w") as f:
            for flag in flags:
               f.write(flag)
               f.write("\n")


    # Delete the build folder
    def cleanFolder(self):
        print("Delete %s\n" % self.path())
        #DEBUG
        if not isDebugMode() and not isKeepMode():
           shutil.rmtree(self.path())

    # Archive results and currentConfig.csv to another folder
    def archiveResults(self):
        results=glob.glob(os.path.join(self.path(),"results_*"))
        for result in results:
            dst=os.path.join(self.archiveResultPath(),os.path.basename(result))
            shutil.copy(result,dst)

        src = os.path.join(self.path(),"currentConfig.csv")
        dst = os.path.join(self.archiveResultPath(),os.path.basename(src))
        shutil.copy(src,dst)
       

    @contextmanager
    def buildFolder(self):
       current=os.getcwd()
       try:
           os.chdir(self.path() )
           yield self.path()
       finally:
           os.chdir(current)

    @contextmanager
    def archiveFolder(self):
       current=os.getcwd()
       try:
           os.chdir(self.archivePath() )
           yield self.archivePath()
       finally:
           os.chdir(current)

    @contextmanager
    def resultFolder(self):
       current=os.getcwd()
       try:
           os.chdir(self.archiveResultPath())
           yield self.archiveResultPath()
       finally:
           os.chdir(current)

    @contextmanager
    def logFolder(self):
       current=os.getcwd()
       try:
           os.chdir(self.archiveLogPath())
           yield self.archiveLogPath()
       finally:
           os.chdir(current)

    @contextmanager
    def errorFolder(self):
       current=os.getcwd()
       try:
           os.chdir(self.archiveErrorPath())
           yield self.archiveErrorPath()
       finally:
           os.chdir(current)

class Test:
    def __init__(self,build,test):
        self._test = test
        self._buildConfig = build

    def buildConfig(self):
        return(self._buildConfig)

    def testName(self):
        return(self._test)

    # Process a test from the test description file
    def processTest(self,patternConfig):
      if isDebugMode():
        if patternConfig:
           completed=subprocess.run([sys.executable,"processTests.py","-p",patternConfig["patterns"],"-d",patternConfig["parameters"],"-e",self.testName(),"1"],timeout=3600)
        else:
           completed=subprocess.run([sys.executable,"processTests.py","-e",self.testName(),"1"],timeout=3600)
        check(completed)
      else:
        if patternConfig:
           completed=subprocess.run([sys.executable,"processTests.py","-p",patternConfig["patterns"],"-d",patternConfig["parameters"],"-e",self.testName()],timeout=3600)
        else:
           completed=subprocess.run([sys.executable,"processTests.py","-e",self.testName()],timeout=3600)
        check(completed)

    def getResultPath(self):
        return(os.path.join(self.buildConfig().path() ,self.resultName()))

    def resultName(self):
        return("results_%s.txt" % self.testName())

    # Run a specific test in the current folder
    # A specific results.txt file is created in
    # the build folder for this test
    #
    # We need a timeout and detect failed run
    def run(self,fvp,benchmode):
        timeoutVal=3600
        if benchmode:
          timeoutVal = 3600 * 4
        completed = None
        with self.buildConfig().buildFolder() as b:
           msg("  Run %s\n" % self.testName() )
           with open(self.resultName(),"w") as results:
              if isDebugMode():
                 print(os.getcwd())
                 print(fvp.split())
                 completed=subprocess.run(fvp.split(),timeout=timeoutVal)
              else:
                 completed=subprocess.run(fvp.split(),stdout=results,timeout=timeoutVal)
        check(completed)

    # Process results of the given tests
    # in given build folder
    # We need to detect failed tests
    def processResult(self):
        msg("  Parse result for %s\n" % self.testName())
        with open(os.path.join(self.buildConfig().archiveResultPath(),"processedResult_%s.txt" % self.testName()),"w") as presult:
             completed=subprocess.run([sys.executable,"processResult.py","-e","-r",self.getResultPath()],stdout=presult,timeout=3600)
        # When a test fail, the regression is continuing but we
        # track that a test has failed
        if completed.returncode==0:
           return(NOTESTFAILED)
        else:
           return(TESTFAILED)

    # Compute the regression data
    def computeSummaryStat(self):
        msg("  Compute regressions for %s\n" % self.testName())
        completed=subprocess.run([sys.executable,"summaryBench.py","-r",self.getResultPath(),self.testName()],timeout=3600)
        # When a test fail, the regression is continuing but we
        # track that a test has failed
        if completed.returncode==0:
           return(NOTESTFAILED)
        else:
           return(TESTFAILED)

    def runAndProcess(self,patternConfig,compiler,fvp,sim,benchmode,db,regdb,benchid,regid):
        # If we can't parse test description we fail all tests
        self.processTest(patternConfig)
        # Otherwise if only building or those tests are failing, we continue
        # with other tests
        try:
           self.buildConfig().build(self.testName())
        except:
            return(MAKEFAILED)
        # We run tests only for AC6
        # For other compilers only build is tests
        # Since full build is no more possible because of huge pattersn,
        # build is done per test suite.
        if sim:
           if fvp is not None:
              if isDebugMode():
                 print(fvp)
              self.run(fvp,benchmode)
              error=self.processResult()
              if benchmode and (error == NOTESTFAILED):
                 error = self.computeSummaryStat()
                 if db is not None:
                    addToDb(db,self.testName(),benchid)
                 if regdb is not None:
                    addToRegDb(regdb,self.testName(),regid)
              return(error)
           else:
              msg("No FVP available")
              return(NOTESTFAILED)
        else:
           return(NOTESTFAILED)
        


# Preprocess the test description
def preprocess(desc):
    msg("Process test description file %s\n" % desc)
    completed = subprocess.run([sys.executable, "preprocess.py","-f",desc],timeout=3600)
    check(completed)

# Generate all missing C code by using all classes in the
# test description file
def generateAllCCode(patternConfig):
    msg("Generate all missing C files\n")
    if patternConfig:
       completed = subprocess.run([sys.executable,"processTests.py",
        "-p",patternConfig["patterns"],"-d",patternConfig["parameters"],"-e"],timeout=3600)
    else:
       completed = subprocess.run([sys.executable,"processTests.py", "-e"],timeout=3600)
    check(completed)

# Create db
def createDb(sqlite,desc):
    msg("Create database %s\n" % desc)
    with open("createDb.sql") as db:
        completed = subprocess.run([sqlite, desc],stdin=db, timeout=3600)
    check(completed)







