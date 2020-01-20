import os
import os.path
import subprocess 
import colorama
from colorama import init,Fore, Back, Style
import argparse

init()

def msg(t):
    print(Fore.CYAN + t + Style.RESET_ALL)

def processTest(test):
    subprocess.call(["python","processTests.py","-e",test])

def build(build,fvp,test):
    result = "results_%s.txt" % test
    resultPath = os.path.join(build,result)

    current=os.getcwd()
    try:
       msg("Build %s" % test)
       os.chdir(build)
       subprocess.call(["make"])
       msg("Run %s" % test)
       with open(result,"w") as results:
          subprocess.call([fvp,"-a","Testing"],stdout=results)
    finally:
       os.chdir(current)

    msg("Parse result for %s" % test)
    subprocess.call(["python","processResult.py","-e","-r",resultPath])

def processAndRun(buildfolder,fvp,test):
    processTest(test)
    build(buildfolder,fvp,test)

parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-f', nargs='?',type = str, default="build_m7", help="Build folder")
parser.add_argument('-v', nargs='?',type = str, default="C:\\Program Files\\ARM\\Development Studio 2019.0\\sw\\models\\bin\\FVP_MPS2_Cortex-M7.exe", help="Fast Model")

args = parser.parse_args()

if args.f is not None:
   BUILDFOLDER=args.f
else:
   BUILDFOLDER="build_m7"

if args.v is not None:
   FVP=args.v
else:
   FVP="C:\\Program Files\\ARM\\Development Studio 2019.0\\sw\\models\\bin\\FVP_MPS2_Cortex-M7.exe"

msg("Process test description file")
subprocess.call(["python", "preprocess.py","-f","desc.txt"])

msg("Generate all missing C files")
subprocess.call(["python","processTests.py", "-e"])

msg("Statistics Tests")
processAndRun(BUILDFOLDER,FVP,"StatsTests")

msg("Support Tests")
processAndRun(BUILDFOLDER,FVP,"SupportTests")

msg("Support Bar Tests F32")
processAndRun(BUILDFOLDER,FVP,"SupportBarTestsF32")

msg("Basic Tests")
processAndRun(BUILDFOLDER,FVP,"BasicTests")

msg("Complex Tests")
processAndRun(BUILDFOLDER,FVP,"ComplexTests")

msg("Fast Maths Tests")
processAndRun(BUILDFOLDER,FVP,"FastMath")

msg("SVM Tests")
processAndRun(BUILDFOLDER,FVP,"SVMTests")

msg("Bayes Tests")
processAndRun(BUILDFOLDER,FVP,"BayesTests")

msg("Distance Tests")
processAndRun(BUILDFOLDER,FVP,"DistanceTests")

msg("Filtering Tests")
processAndRun(BUILDFOLDER,FVP,"FilteringTests")

msg("Matrix Tests")
processAndRun(BUILDFOLDER,FVP,"MatrixTests")

msg("Transform Tests")
processAndRun(BUILDFOLDER,FVP,"TransformTests")


