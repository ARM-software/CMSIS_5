# Process the test results
# Test status (like passed, or failed with error code)

import argparse
import re 
import TestScripts.NewParser as parse
import TestScripts.CodeGen
from collections import deque
import os.path
import csv
import TestScripts.ParseTrace
import colorama
from colorama import init,Fore, Back, Style
import sys 

resultStatus=0

init()


def errorStr(id):
  if id == 1:
     return("UNKNOWN_ERROR")
  if id == 2:
     return("Equality error")
  if id == 3:
     return("Absolute difference error")
  if id == 4:
     return("Relative difference error")
  if id == 5:
     return("SNR error")
  if id == 6:
     return("Different length error")
  if id == 7:
     return("Assertion error")
  if id == 8:
     return("Memory allocation error")
  if id == 9:
     return("Empty pattern error")
  if id == 10:
     return("Buffer tail corrupted")
  if id == 11:
     return("Close float error")

  return("Unknown error %d" % id)


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

def joinit(iterable, delimiter):
    # Intersperse a delimiter between element of a list
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

# Return test result as a text tree
class TextFormatter:
      def start(self):
          None 

      def printGroup(self,elem,theId):
        if elem is None:
           elem = root
        message=elem.data["message"]
        if not elem.data["deprecated"]:
           kind = "Suite"
           ident = " " * elem.ident
           if elem.kind == TestScripts.Parser.TreeElem.GROUP:
              kind = "Group"
           #print(elem.path)
           print(Style.BRIGHT + ("%s%s : %s (%d)" % (ident,kind,message,theId)) + Style.RESET_ALL)

      def printTest(self,elem, theId, theError,errorDetail,theLine,passed,cycles,params):
          message=elem.data["message"]
          if not elem.data["deprecated"]:
             kind = "Test"
             ident = " " * elem.ident
             p=Fore.RED + "FAILED" + Style.RESET_ALL
             if passed == 1:
                p= Fore.GREEN + "PASSED" + Style.RESET_ALL
             print("%s%s %s(%d)%s : %s (cycles = %d)" % (ident,message,Style.BRIGHT,theId,Style.RESET_ALL,p,cycles))
             if params:
                print("%s %s" % (ident,params))
             if passed != 1:
                print(Fore.RED + ("%s %s at line %d" % (ident, errorStr(theError), theLine)) + Style.RESET_ALL)
                if (len(errorDetail)>0):
                   print(Fore.RED + ident + " " + errorDetail + Style.RESET_ALL)

      def pop(self):
          None

      def end(self):
        None

# Return test result as a text tree
class HTMLFormatter:
      def __init__(self):
        self.nb=1
        self.suite=False

      def start(self):
          print("<html><head><title>Test Results</title></head><body>") 

      def printGroup(self,elem,theId):
        if elem is None:
           elem = root
        message=elem.data["message"]
        if not elem.data["deprecated"]:
           kind = "Suite"
           ident = " " * elem.ident
           if elem.kind == TestScripts.Parser.TreeElem.GROUP:
              kind = "Group"
           if kind == "Group":
              print("<h%d> %s (%d) </h%d>" % (self.nb,message,theId,self.nb)) 
           else:
              print("<h%d> %s (%d) </h%d>" % (self.nb,message,theId,self.nb)) 
              self.suite=True
              print("<table style=\"width:100%\">")
              print("<tr>")
              print("<td>Name</td>")
              print("<td>ID</td>")
              print("<td>Status</td>")
              print("<td>Params</td>")
              print("<td>Cycles</td>")
              print("</tr>")
           self.nb = self.nb + 1

      def printTest(self,elem, theId, theError,errorDetail,theLine,passed,cycles,params):
          message=elem.data["message"]
          if not elem.data["deprecated"]:
             kind = "Test"
             ident = " " * elem.ident
             p="<font color=\"red\">FAILED</font>"
             if passed == 1:
                p= "<font color=\"green\">PASSED</font>"
             print("<tr>")
             print("<td><pre>%s</pre></td>" % message)
             print("<td>%d</td>" % theId)
             print("<td>%s</td>" % p)
             if params:
                print("<td>%s</td>\n" % (params))
             else:
                print("<td></td>\n")
             print("<td>%d</td>" % cycles)
             print("</tr>")

             if passed != 1:

                print("<tr><td colspan=4><font color=\"red\">%s at line %d</font></td></tr>" % (errorStr(theError), theLine))
                if (len(errorDetail)>0):
                   print("<tr><td colspan=4><font color=\"red\">" + errorDetail + "</font></td></tr>")

      def pop(self):
          if self.suite:
            print("</table>")
          self.nb = self.nb - 1
          self.suite=False

      def end(self):
        print("</body></html>")

# Return test result as a CSV
class CSVFormatter:

      def __init__(self):
        self.name=[]
        self._start=True

      def start(self):
          print("CATEGORY,NAME,ID,STATUS,CYCLES,PARAMS") 
          
      def printGroup(self,elem,theId):
        if elem is None:
           elem = root
        # Remove Root from category name in CSV file.
        if not self._start:
           self.name.append(elem.data["class"])
        else:
            self._start=False
        message=elem.data["message"]
        if not elem.data["deprecated"]:
           kind = "Suite"
           ident = " " * elem.ident
           if elem.kind == TestScripts.Parser.TreeElem.GROUP:
              kind = "Group"

      def printTest(self,elem, theId, theError, errorDetail,theLine,passed,cycles,params):
          message=elem.data["message"]
          if not elem.data["deprecated"]:
             kind = "Test"
             name=elem.data["class"] 
             category= "".join(list(joinit(self.name,":")))
             print("%s,%s,%d,%d,%d,\"%s\"" % (category,name,theId,passed,cycles,params))

      def pop(self):
         if self.name:
            self.name.pop()

      def end(self):
        None

class MathematicaFormatter:

      def __init__(self):
        self._hasContent=[False]
        self._toPop=[]

      def start(self):
          None

      def printGroup(self,elem,theId):
        if self._hasContent[len(self._hasContent)-1]:
           print(",",end="")
        
        print("<|") 
        self._hasContent[len(self._hasContent)-1] = True
        self._hasContent.append(False)
        if elem is None:
           elem = root
        message=elem.data["message"]
        if not elem.data["deprecated"]:

           kind = "Suite"
           ident = " " * elem.ident
           if elem.kind == TestScripts.Parser.TreeElem.GROUP:
              kind = "Group"
           print("\"%s\" ->" % (message))
           #if kind == "Suite":
           print("{",end="")
           self._toPop.append("}")
           #else:
           #   self._toPop.append("")

      def printTest(self,elem, theId, theError,errorDetail,theLine,passed,cycles,params):
          message=elem.data["message"]
          if not elem.data["deprecated"]:
             kind = "Test"
             ident = " " * elem.ident
             p="FAILED"
             if passed == 1:
                p="PASSED"
             parameters=""
             if params:
                parameters = "%s" % params
             if self._hasContent[len(self._hasContent)-1]:
               print(",",end="")
             print("<|\"NAME\" -> \"%s\",\"ID\" -> %d,\"STATUS\" -> \"%s\",\"CYCLES\" -> %d,\"PARAMS\" -> \"%s\"|>" % (message,theId,p,cycles,parameters))
             self._hasContent[len(self._hasContent)-1] = True
             #if passed != 1:
             #   print("%s Error = %d at line %d" % (ident, theError, theLine))

      def pop(self):
          print(self._toPop.pop(),end="")
          print("|>")
          self._hasContent.pop()

      def end(self):
        None

NORMAL = 1 
INTEST = 2
TESTPARAM = 3
ERRORDESC = 4

def createMissingDir(destPath):
  theDir=os.path.normpath(os.path.dirname(destPath))
  if not os.path.exists(theDir):
      os.makedirs(theDir)

def correctPath(path):
  while (path[0]=="/") or (path[0] == "\\"):
      path = path[1:]
  return(path)

def extractDataFiles(results,outputDir):
  infile = False
  f = None
  for l in results:
      if re.match(r'^.*D:[ ].*$',l):
          if infile:
            if re.match(r'^.*D:[ ]END$',l):
               infile = False 
               if f:
                 f.close()
            else:
              if f:
                m = re.match(r'^.*D:[ ](.*)$',l)
                data = m.group(1)
                f.write(data)
                f.write("\n")

          else:
            m = re.match(r'^.*D:[ ](.*)$',l)
            path = str(m.group(1))
            infile = True 
            destPath = os.path.join(outputDir,correctPath(path))
            createMissingDir(destPath)
            f = open(destPath,"w")

         

def writeBenchmark(elem,benchFile,theId,theError,passed,cycles,params,config):
  if benchFile:
    name=elem.data["class"] 
    category= elem.categoryDesc()
    old=""
    if "testData" in elem.data:
      if "oldID" in elem.data["testData"]:
         old=elem.data["testData"]["oldID"]
    benchFile.write("\"%s\",\"%s\",%d,\"%s\",%s,%d,%s\n" % (category,name,theId,old,params,cycles,config))

def getCyclesFromTrace(trace):
  if not trace:
    return(0)
  else:
    return(TestScripts.ParseTrace.getCycles(trace))

def analyseResult(resultPath,root,results,embedded,benchmark,trace,formatter):
    global resultStatus
    calibration = 0
    if trace:
      # First cycle in the trace is the calibration data
      # The noramlisation factor must be coherent with the C code one.
      calibration = int(getCyclesFromTrace(trace) / 20)
    formatter.start()
    path = []
    state = NORMAL
    prefix=""
    elem=None
    theId=None
    theError=None
    errorDetail=""
    theLine=None
    passed=0
    cycles=None
    benchFile = None
    config=""
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
                       createMissingDir(benchPath)
                       if benchFile:
                          printf("ERROR BENCH FILE %s ALREADY OPEN" % benchPath)
                          benchFile.close()
                          benchFile=None
                       benchFile=open(benchPath,"w")
                       header = "".join(list(joinit(elem.params.full,",")))
                       # A test and a benchmark are different
                       # so we don't dump a status and error
                       # A status and error in a benchmark would
                       # impact the cycles since the test
                       # would be taken into account in the measurement
                       # So benchmark are always passing and contain no test
                       #benchFile.write("ID,%s,PASSED,ERROR,CYCLES\n" % header)
                       csvheaders = ""

                       with open(os.path.join(resultPath,'currentConfig.csv'), 'r') as f:
                          reader = csv.reader(f)
                          csvheaders = next(reader, None)
                          configList = list(reader)
                          #print(configList)
                          config = "".join(list(joinit(configList[0],",")))
                          configHeaders = "".join(list(joinit(csvheaders,",")))
                       benchFile.write("CATEGORY,NAME,ID,OLDID,%s,CYCLES,%s\n" % (header,configHeaders))
   
                    formatter.printGroup(elem,theId)
      
                 # If we have detected a test, we switch to test mode
                 if re.match(r'^%s[t][ ]*$' % prefix,l):
                    state = INTEST
                 
      
                 # Pop
                 # End of suite or group
                 if re.match(r'^%sp.*$' % prefix,l):
                   if benchFile:
                      benchFile.close()
                      benchFile=None
                   path.pop()
                   formatter.pop()
           elif state == INTEST:
             if len(l) > 0:
               # In test mode, we are looking for test status.
               # A line starting with S
               # (There may be empty lines or line for data files)
               passRe = r'^%s([0-9]+)[ ]+([0-9]+)[ ]+([0-9]+)[ ]+([t0-9]+)[ ]+([YN]).*$'  % prefix
               if re.match(passRe,l):
                    # If we have found a test status then we will start again
                    # in normal mode after this.
                    
                    m = re.match(passRe,l)
                    
                    # Extract test ID, test error code, line number and status
                    theId=m.group(1)
                    theId=int(theId)
      
                    theError=m.group(2)
                    theError=int(theError)
      
                    theLine=m.group(3)
                    theLine=int(theLine)
      
                    maybeCycles = m.group(4)
                    if maybeCycles == "t":
                       cycles = getCyclesFromTrace(trace) - calibration
                    else:
                       cycles = int(maybeCycles)
   
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
   
                    
                    state = ERRORDESC
               else:
                 if re.match(r'^%sp.*$' % prefix,l):
                   if benchFile:
                      benchFile.close()
                      benchFile=None
                   path.pop()
                   formatter.pop()
                 if re.match(r'^%s[t][ ]*$' % prefix,l):
                    state = INTEST
                 else:
                    state = NORMAL
           elif state == ERRORDESC:
                    if len(l) > 0:
                       if re.match(r'^.*E:.*$',l):
                          if re.match(r'^.*E:[ ].*$',l):
                             m = re.match(r'^.*E:[ ](.*)$',l)
                             errorDetail = m.group(1)
                          else:
                             errorDetail = ""
                          state = TESTPARAM
           else:
             if len(l) > 0:
                state = INTEST 
                params=""
                if re.match(r'^.*b[ ]+([0-9,]+)$',l):
                   m=re.match(r'^.*b[ ]+([0-9,]+)$',l)
                   params=m.group(1).strip()
                   # Format the node
                   #print(elem.fullPath())
                   #createMissingDir(destPath)
                   writeBenchmark(elem,benchFile,theId,theError,passed,cycles,params,config)
                else:
                   params=""
                   writeBenchmark(elem,benchFile,theId,theError,passed,cycles,params,config)
                   # Format the node
                if not passed:
                   resultStatus=1
                formatter.printTest(elem,theId,theError,errorDetail,theLine,passed,cycles,params)

             
    formatter.end()          


def analyze(root,results,args,trace):
  # currentConfig.csv should be in the same place
  resultPath=os.path.dirname(args.r)

  if args.c:
     analyseResult(resultPath,root,results,args.e,args.b,trace,CSVFormatter())
  elif args.html:
     analyseResult(resultPath,root,results,args.e,args.b,trace,HTMLFormatter())
  elif args.m:
     analyseResult(resultPath,root,results,args.e,args.b,trace,MathematicaFormatter())
  else:
     print("")
     print(Fore.RED + "The cycles displayed by this script must not be trusted." + Style.RESET_ALL)
     print(Fore.RED + "They are just an indication. The timing code has not yet been validated." + Style.RESET_ALL)
     print("")

     analyseResult(resultPath,root,results,args.e,args.b,trace,TextFormatter())

parser = argparse.ArgumentParser(description='Parse test description')

parser.add_argument('-f', nargs='?',type = str, default="Output.pickle", help="Test description file path")
# Where the result file can be found
parser.add_argument('-r', nargs='?',type = str, default=None, help="Result file path")
parser.add_argument('-c', action='store_true', help="CSV output")
parser.add_argument('-html', action='store_true', help="HTML output")
parser.add_argument('-e', action='store_true', help="Embedded test")
# -o needed when -e is true to know where to extract the output files
parser.add_argument('-o', nargs='?',type = str, default="Output", help="Output dir path")

parser.add_argument('-b', nargs='?',type = str, default="FullBenchmark", help="Full Benchmark dir path")
parser.add_argument('-m', action='store_true', help="Mathematica output")
parser.add_argument('-t', nargs='?',type = str, default=None, help="External trace file")

args = parser.parse_args()




if args.f is not None:
    #p = parse.Parser()
    # Parse the test description file
    #root = p.parse(args.f)
    root=parse.loadRoot(args.f)
    if args.t:
       with open(args.t,"r") as trace:
         with open(args.r,"r") as results:
             analyze(root,results,args,iter(trace))
    else:
       with open(args.r,"r") as results:
           analyze(root,results,args,None)
    if args.e:
       # In FPGA mode, extract output files from stdout (result file)
       with open(args.r,"r") as results:
          extractDataFiles(results,args.o)

    sys.exit(resultStatus)
    
else:
    parser.print_help()