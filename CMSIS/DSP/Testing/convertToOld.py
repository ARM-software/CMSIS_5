# Process the test results
# Test status (like passed, or failed with error code)

import argparse
import re 
import TestScripts.NewParser as parse
import TestScripts.CodeGen
from collections import deque
import os.path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import csv
import TestScripts.Deprecate as d

result = []
commonParams = []

def findItem(root,path):
        """ Find a node in a tree
      
        Args:
          path (list) : A list of node ID
            This list is describing a path in the tree.
            By starting from the root and following this path,
            we can find the node in the tree.
        Raises:
          Nothing 
        Returns:
          TreeItem : A node
        """
        # The list is converted into a queue.
        q = deque(path) 
        q.popleft()
        c = root
        while q:
            n = q.popleft() 
            # We get the children based on its ID and continue
            c = c[n-1]
        return(c)



NORMAL = 1 
INTEST = 2
TESTPARAM = 3

def joinit(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

def formatProd(a,b):
  if a == "Intercept":
     return(str(b))
  return("%s * %s" % (a,b))

def convert(elem,fullPath):
   global commonParams
   global result
   regressionPath=os.path.join(os.path.dirname(fullPath),"regression.csv")
   full=pd.read_csv(fullPath,dtype={'OLDID': str} ,keep_default_na = False)
   reg=pd.read_csv(regressionPath,dtype={'OLDID': str} ,keep_default_na = False)
   commonParams = list(joinit(elem.params.full,","))
   header = ["OLDID"] + commonParams + ["CYCLES"]

   r=full[header].rename(columns = {"OLDID":"TESTNB"})
   r["TESTNB"] = pd.to_numeric(r["TESTNB"])
   r["PASSED"]=1 
   result.append(r)


def extractBenchmarks(benchmark,elem):
  if not elem.data["deprecated"]:
     if elem.params:
         benchPath = os.path.join(benchmark,elem.fullPath(),"fullBenchmark.csv")
         print("Processing %s" % benchPath)
         convert(elem,benchPath)
         
     for c in elem.children:
       extractBenchmarks(benchmark,c)



parser = argparse.ArgumentParser(description='Generate summary benchmarks')

parser.add_argument('-f', nargs='?',type = str, default="Output.pickle", help="Test description file path")
parser.add_argument('-b', nargs='?',type = str, default="FullBenchmark", help="Full Benchmark dir path")
parser.add_argument('-e', action='store_true', help="Embedded test")
parser.add_argument('-o', nargs='?',type = str, default="bench.csv", help="Output csv file using old format")

parser.add_argument('others', nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.f is not None:
    #p = parse.Parser()
    # Parse the test description file
    #root = p.parse(args.f)
    root=parse.loadRoot(args.f)
    d.deprecate(root,args.others)
    extractBenchmarks(args.b,root)
    finalResult = pd.concat(result)
    cols = ['TESTNB'] + commonParams
    finalResult=finalResult.sort_values(by=cols)
    finalResult.to_csv(args.o,index=False,quoting=csv.QUOTE_NONNUMERIC)
    
else:
    parser.print_help()