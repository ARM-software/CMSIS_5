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
     return("%.3f" % b)
  return("%s * %.3f" % (a,b))

def log2(x):
  return(np.log2(x))

def log(x):
  return(np.log(x))

def summaryBenchmark(resultPath,elem,path):
   regressionPath=os.path.join(os.path.dirname(path),"regression.csv")

   if os.path.isfile(path):
      print("  Generating %s" % regressionPath)
      full=pd.read_csv(path,dtype={'OLDID': str} ,keep_default_na = False)
      #print(full)
      
      csvheaders = []
      with open(os.path.join(resultPath,'currentConfig.csv'), 'r') as f:
           reader = csv.reader(f)
           csvheaders = next(reader, None)
   
      groupList = list(set(elem.params.full) - set(elem.params.summary))
      #grouped=full.groupby(list(elem.params.summary) + ['ID','CATEGORY']).max()
      #grouped.reset_index(level=grouped.index.names, inplace=True)
      #print(grouped)
      #print(grouped.columns)

  
      def reg(d):
       m=d["CYCLES"].max()
       #print( elem.params.formula)
       
       results = smf.ols('CYCLES ~ ' + elem.params.formula, data=d).fit()

       f=joinit([formatProd(a,b) for (a,b) in zip(results.params.index,results.params.values)]," + ")
       f="".join(f)
       f = re.sub(r':','*',f)
       #print(results.summary())
       return(pd.Series({'Regression':"%s" % f,'MAX' : m,'MAXREGCOEF' : results.params.values[-1]}))
   
      regList = ['ID','OLDID','CATEGORY','TESTNAME','NAME'] + csvheaders + groupList 
      
      regression=full.groupby(regList).apply(reg)
      regression.reset_index(level=regression.index.names, inplace=True)
      renamingDict = { a : b for (a,b) in zip(elem.params.full,elem.params.paramNames)}
      regression = regression.rename(columns=renamingDict)
      regression.to_csv(regressionPath,index=False,quoting=csv.QUOTE_NONNUMERIC)


def extractBenchmarks(resultPath,benchmark,elem):
  if not elem.data["deprecated"]:
     if elem.params:
         benchPath = os.path.join(benchmark,elem.fullPath(),"fullBenchmark.csv")
         print("Processing %s" % benchPath)
         summaryBenchmark(resultPath,elem,benchPath)
         
     for c in elem.children:
       extractBenchmarks(resultPath,benchmark,c)



parser = argparse.ArgumentParser(description='Generate summary benchmarks')

parser.add_argument('-f', nargs='?',type = str, default="Output.pickle", help="Test description cache")
parser.add_argument('-b', nargs='?',type = str, default="FullBenchmark", help="Full Benchmark dir path")
# Needed to find the currentConfig.csv and know the headers
parser.add_argument('-r', nargs='?',type = str, default=None, help="Result file path")

parser.add_argument('others', nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.f is not None:
    #p = parse.Parser()
    # Parse the test description file
    #root = p.parse(args.f)
    root=parse.loadRoot(args.f)
    d.deprecate(root,args.others)
    resultPath=os.path.dirname(args.r)
    extractBenchmarks(resultPath,args.b,root)
    
else:
    parser.print_help()