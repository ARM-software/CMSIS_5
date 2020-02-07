import os
import os.path
import subprocess 
import colorama
from colorama import init,Fore, Back, Style
import argparse

GROUPS = [
"BasicBenchmarks",
"ComplexBenchmarks",
"FIR",
"MISC",
"DECIM",
"BIQUAD",
"Controller",
"FastMath",
"SupportBarF32",
"Support",
"Unary",
"Binary",
"Transform"
]

init()

def msg(t):
    print(Fore.CYAN + t + Style.RESET_ALL)

def processTest(test):
    subprocess.call(["python","processTests.py","-e",test])

def addToDB(cmd):
    for g in GROUPS:
        msg("Add group %s" % g)
        subprocess.call(["python",cmd,g])

def run(build,fvp,custom=None):
    result = "results.txt"
    resultPath = os.path.join(build,result)

    current=os.getcwd()
    try:
       msg("Build" )
       os.chdir(build)
       subprocess.call(["make"])
       msg("Run")
       with open(result,"w") as results:
          if custom:
            subprocess.call([fvp] + custom,stdout=results)
          else:
             subprocess.call([fvp,"-a","Testing"],stdout=results)
    finally:
       os.chdir(current)

    msg("Parse result")
    subprocess.call(["python","processResult.py","-e","-r",resultPath])

    msg("Regression computations")
    subprocess.call(["python","summaryBench.py","-r",resultPath])

    msg("Add results to benchmark database")
    addToDB("addToDB.py")

    msg("Add results to regression database")
    addToDB("addToRegDB.py")



def processAndRun(buildfolder,fvp,custom=None):
    processTest("DSPBenchmarks")
    run(buildfolder,fvp,custom=custom)

parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-f', nargs='?',type = str, default="build_benchmark_m7", help="Build folder")
parser.add_argument('-v', nargs='?',type = str, default="C:\\Program Files\\ARM\\Development Studio 2019.0\\sw\\models\\bin\\FVP_MPS2_Cortex-M7.exe", help="Fast Model")
parser.add_argument('-c', nargs='?',type = str, help="Custom args")

args = parser.parse_args()

if args.f is not None:
   BUILDFOLDER=args.f
else:
   BUILDFOLDER="build_benchmark_m7"

if args.v is not None:
   FVP=args.v
else:
   FVP="C:\\Program Files\\ARM\\Development Studio 2019.0\\sw\\models\\bin\\FVP_MPS2_Cortex-M7.exe"


if args.c:
    custom = args.c.split()
else:
    custom = None

print(Fore.RED + "bench.db and reg.db databases must exist before running this script" + Style.RESET_ALL)

msg("Process benchmark description file")
subprocess.call(["python", "preprocess.py","-f","bench.txt"])

msg("Generate all missing C files")
subprocess.call(["python","processTests.py", "-e"])


processAndRun(BUILDFOLDER,FVP,custom=custom)
