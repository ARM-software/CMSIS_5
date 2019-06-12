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

def summaryBenchmark(elem,path):
   regressionPath=os.path.join(os.path.dirname(path),"regression.csv")
   full=pd.read_csv(path)
   
   csvheaders = []
   with open('currentConfig.csv', 'r') as f:
        reader = csv.reader(f)
        csvheaders = next(reader, None)

   groupList = list(set(elem.params.full) - set(elem.params.summary))
   #grouped=full.groupby(list(elem.params.summary) + ['ID','CATEGORY']).max()
   #grouped.reset_index(level=grouped.index.names, inplace=True)
   #print(grouped)
   #print(grouped.columns)

  
   def reg(d):
    m=d["CYCLES"].max()
    results = smf.ols('CYCLES ~ ' + elem.params.formula, data=d).fit()
    f=joinit([formatProd(a,b) for (a,b) in zip(results.params.index,results.params.values)]," + ")
    f="".join(f)
    f = re.sub(r':','*',f)
    #print(results.summary())
    return(pd.Series({'Regression':"%s" % f,'MAX' : m}))

   regList = ['ID','CATEGORY','NAME'] + csvheaders + groupList 
   
   regression=full.groupby(regList).apply(reg)
   regression.reset_index(level=regression.index.names, inplace=True)
   renamingDict = { a : b for (a,b) in zip(elem.params.full,elem.params.paramNames)}
   regression = regression.rename(columns=renamingDict)
   regression.to_csv(regressionPath,index=False,quoting=csv.QUOTE_NONNUMERIC)



def analyseResult(root,results,embedded,benchmark):
    path = []
    state = NORMAL
    prefix=""
    elem=None
    theId=None
    theError=None
    theLine=None
    passed=0
    cycles=None
    benchFile = None
    if embedded:
       prefix = ".*S:[ ]"

    # Parse the result file.
    # NORMAL mode is when we are parsing suite or group.
    # Otherwise we are parsing a test and we need to analyse the
    # test result.
    # TESTPARAM is used to read parameters of the test.
    # Format of output is:
    #node ident : s id or g id or t or u
    #test status : id error linenb status Y or N (Y when passing)
    #param for this test b x,x,x,x or b alone if not param
    #node end : p
    # In FPGA mode:
    #Prefix S:[ ] before driver dump
    # D:[ ] before data dump (output patterns)

    for l in results:
        l = l.strip() 
        if not re.match(r'^.*D:[ ].*$',l):
           if state == NORMAL:
              if len(l) > 0:
                 # Line starting with g or s is a suite or group.
                 # In FPGA mode, those line are prefixed with 'S: '
                 # and data file with 'D: '
                 if re.match(r'^%s[gs][ ]+[0-9]+.*$' % prefix,l):
                    # Extract the test id
                    theId=re.sub(r'^%s[gs][ ]+([0-9]+).*$' % prefix,r'\1',l)
                    theId=int(theId)
                    path.append(theId)
                    # From a list of id, find the TreeElem in the Parsed tree
                    # to know what is the node.
                    elem = findItem(root,path)
                    # Display formatted output for this node
                    if elem.params:
                       #print(elem.params.full)
                       benchPath = os.path.join(benchmark,elem.fullPath(),"fullBenchmark.csv")
                       summaryBenchmark(elem,benchPath)
                       
   
                 # If we have detected a test, we switch to test mode
                 if re.match(r'^%s[t][ ]*$' % prefix,l):
                    state = INTEST
                 
      
                 # Pop
                 # End of suite or group
                 if re.match(r'^%sp.*$' % prefix,l):
                   path.pop()
           elif state == INTEST:
             if len(l) > 0:
               # In test mode, we are looking for test status.
               # A line starting with S
               # (There may be empty lines or line for data files)
               passRe = r'^%s([0-9]+)[ ]+([0-9]+)[ ]+([0-9]+)[ ]+([0-9]+)[ ]+([YN]).*$'  % prefix
               if re.match(passRe,l):
                    # If we have found a test status then we will start again
                    # in normal mode after this.
                    
                    m = re.match(passRe,l)
                    
                    # Extract test ID, test error code, line number and status
                    theId=m.group(1)
                    theId=int(theId)
      
                    status=m.group(5)
                    passed=0
      
                    # Convert status to number as used by formatter.
                    if status=="Y":
                       passed = 1
                    if status=="N":
                       passed = 0
                    # Compute path to this node
                    newPath=path.copy()
                    newPath.append(theId)
                    # Find the node in the Tree
                    elem = findItem(root,newPath)
   
                    
                    state = TESTPARAM
               else:
                 if re.match(r'^%sp.*$' % prefix,l):
                   path.pop()
                 if re.match(r'^%s[t][ ]*$' % prefix,l):
                    state = INTEST
                 else:
                    state = NORMAL
           else:
             if len(l) > 0:
                state = INTEST 
                params=""


parser = argparse.ArgumentParser(description='Generate summary benchmarks')

parser.add_argument('-f', nargs='?',type = str, default=None, help="Test description file path")
# Where the result file can be found
parser.add_argument('-r', nargs='?',type = str, default=None, help="Result file path")

parser.add_argument('-b', nargs='?',type = str, default="FullBenchmark", help="Full Benchmark dir path")
parser.add_argument('-e', action='store_true', help="Embedded test")

args = parser.parse_args()

if args.f is not None:
    p = parse.Parser()
    # Parse the test description file
    root = p.parse(args.f)
    with open(args.r,"r") as results:
        analyseResult(root,results,args.e,args.b)
    
else:
    parser.print_help()