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

def build(build,fvp,test,custom=None):
    result = "results_%s.txt" % test
    resultPath = os.path.join(build,result)

    current=os.getcwd()
    try:
       msg("Build %s" % test)
       os.chdir(build)
       subprocess.call(["make"])
       msg("Run %s" % test)
       with open(result,"w") as results:
          if custom:
            subprocess.call([fvp] + custom,stdout=results)
          else:
             subprocess.call([fvp,"-a","Testing"],stdout=results)
    finally:
       os.chdir(current)

    msg("Parse result for %s" % test)
    subprocess.call(["python","processResult.py","-e","-r",resultPath])

def processAndRun(buildfolder,fvp,test,custom=None):
    processTest(test)
    build(buildfolder,fvp,test,custom=custom)

parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-f', nargs='?',type = str, default="build_m7", help="Build folder")
parser.add_argument('-v', nargs='?',type = str, default="C:\\Program Files\\ARM\\Development Studio 2019.0\\sw\\models\\bin\\FVP_MPS2_Cortex-M7.exe", help="Fast Model")
parser.add_argument('-c', nargs='?',type = str, help="Custom args")

args = parser.parse_args()

if args.f is not None:
   BUILDFOLDER=args.f
else:
   BUILDFOLDER="build_m7"

if args.v is not None:
   FVP=args.v
else:
   FVP="C:\\Program Files\\ARM\\Development Studio 2019.0\\sw\\models\\bin\\FVP_MPS2_Cortex-M7.exe"


if args.c:
    custom = args.c.split()
else:
    custom = None

msg("Process test description file")
subprocess.call(["python", "preprocess.py","-f","desc.txt"])

msg("Generate all missing C files")
subprocess.call(["python","processTests.py", "-e"])

msg("Statistics Tests")
processAndRun(BUILDFOLDER,FVP,"StatsTests",custom=custom)

msg("Support Tests")
processAndRun(BUILDFOLDER,FVP,"SupportTests",custom=custom)

msg("Support Bar Tests F32")
processAndRun(BUILDFOLDER,FVP,"SupportBarTestsF32",custom=custom)

msg("Basic Tests")
processAndRun(BUILDFOLDER,FVP,"BasicTests",custom=custom)

msg("Interpolation Tests")
processAndRun(BUILDFOLDER,FVP,"InterpolationTests",custom=custom)

msg("Complex Tests")
processAndRun(BUILDFOLDER,FVP,"ComplexTests",custom=custom)

msg("Fast Maths Tests")
processAndRun(BUILDFOLDER,FVP,"FastMath",custom=custom)

msg("SVM Tests")
processAndRun(BUILDFOLDER,FVP,"SVMTests",custom=custom)

msg("Bayes Tests")
processAndRun(BUILDFOLDER,FVP,"BayesTests",custom=custom)

msg("Distance Tests")
processAndRun(BUILDFOLDER,FVP,"DistanceTests",custom=custom)

msg("Filtering Tests")
processAndRun(BUILDFOLDER,FVP,"FilteringTests",custom=custom)

msg("Matrix Tests")
processAndRun(BUILDFOLDER,FVP,"MatrixTests",custom=custom)

# Too many patterns to run the full transform directly
msg("Transform Tests CF64")
processAndRun(BUILDFOLDER,FVP,"TransformCF64",custom=custom)

msg("Transform Tests RF64")
processAndRun(BUILDFOLDER,FVP,"TransformRF64",custom=custom)

msg("Transform Tests CF32")
processAndRun(BUILDFOLDER,FVP,"TransformCF32",custom=custom)

msg("Transform Tests RF32")
processAndRun(BUILDFOLDER,FVP,"TransformRF32",custom=custom)

msg("Transform Tests CQ31")
processAndRun(BUILDFOLDER,FVP,"TransformCQ31",custom=custom)

msg("Transform Tests RQ31")
processAndRun(BUILDFOLDER,FVP,"TransformRQ31",custom=custom)

msg("Transform Tests CQ15")
processAndRun(BUILDFOLDER,FVP,"TransformCQ15",custom=custom)

msg("Transform Tests RQ15")
processAndRun(BUILDFOLDER,FVP,"TransformRQ15",custom=custom)