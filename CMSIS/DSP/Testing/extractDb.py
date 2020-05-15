import argparse
import sqlite3
import re
import pandas as pd
import numpy as np

remapNames={
}

def convertSectionName(s):
  if s in remapNames:
    return(remapNames[s])
  else:
    return(s)

class Document:
    def __init__(self,runid,date):
        self._runid = runid 
        self._date = date 
        self._sections = []

    @property
    def runid(self):
        return(self._runid)

    @property
    def date(self):
        return(self._date)

    @property
    def sections(self):
        return(self._sections)

    def addSection(self,section):
        self._sections.append(section)

    def accept(self, visitor):
      visitor.visitDocument(self)
      for element in self._sections:
          element.accept(visitor)   
      visitor.leaveDocument(self)

class Section:
  def __init__(self,name):
      self._name=convertSectionName(name)
      self._subsections = []
      self._tables = []

  def addSection(self,section):
      self._subsections.append(section)

  def addTable(self,table):
      self._tables.append(table)

  @property
  def hasChildren(self):
      return(len(self._subsections)>0)

  @property
  def name(self):
     return(self._name)

  def accept(self, visitor):
      visitor.visitSection(self)
      for element in self._subsections:
          element.accept(visitor) 
      for element in self._tables:
          element.accept(visitor)    
      visitor.leaveSection(self)   

class Table:
    def __init__(self,params,cores):
       self._params=params
       self._cores=cores
       self._rows=[] 

    def addRow(self,row):
       self._rows.append(row)

    @property
    def columns(self):
       return(self._params + self._cores)

    @property
    def params(self):
       return(self._params)

    @property
    def cores(self):
       return(self._cores)

    @property
    def rows(self):
       return(self._rows)

    def accept(self, visitor):
      visitor.visitTable(self)



class Markdown:
  def __init__(self,output):
    self._id=0
    self._output = output

    # Write columns in markdown format
  def writeColumns(self,cols):
        colStr = "".join(joinit(cols,"|"))
        self._output.write("|")
        self._output.write(colStr)
        self._output.write("|\n")
        sepStr="".join(joinit([":-:" for x in cols],"|"))
        self._output.write("|")
        self._output.write(sepStr)
        self._output.write("|\n")
    
    # Write row in markdown format
  def writeRow(self,row):
        row=[str(x) for x in row]
        rowStr = "".join(joinit(row,"|"))
        self._output.write("|")
        self._output.write(rowStr)
        self._output.write("|\n")

  def visitTable(self,table):
      self.writeColumns(table.columns)
      for row in table.rows:
        self.writeRow(row)

  def visitSection(self,section):
     self._id = self._id + 1 
     header = "".join(["#" for i in range(self._id)])
     output.write("%s %s\n" % (header,section.name))

  def leaveSection(self,section):
     self._id = self._id - 1 

  def visitDocument(self,document):
      self._output.write("Run number %d on %s\n" % (document.runid, str(document.date)))

  def leaveDocument(self,document):
      pass

styleSheet="""
<style type='text/css'>

#TOC {
  position: fixed;
  left: 0;
  top: 0;
  width: 280px;
  height: 100%;
  overflow:auto;
  margin-top:5px;
  margin-bottom:10px;
}

html {
  font-size: 16px;
}

html, body {
  background-color: #f3f2ee;
  font-family: "PT Serif", 'Times New Roman', Times, serif;
  color: #1f0909;
  line-height: 1.5em;
}

body {
  margin: auto;
  margin-top:0px;
  margin-left:280px;

}

h1,
h2,
h3,
h4,
h5,
h6 {
  font-weight: bold;
}
h1 {
  font-size: 1.875em;
  margin-top:5px;
}
h2 {
  font-size: 1.3125em;
}
h3 {
  font-size: 1.3125em;
  margin-left:1em;
}
h4 {
  font-size: 1.125em;
  margin-left:1em;
}
h5,
h6 {
  font-size: 1em;
}

#TOC h1 {
  margin-top:0em;
  margin-left:0.5em;
}

table {
  margin-bottom: 1.5em;
  font-size: 1em;
  width: 100%;
  border-collapse: collapse;
  border-spacing: 0;
  width: 100%; 
  margin-left:1em;
}
thead th,
tfoot th {
  padding: .25em .25em .25em .4em;
  text-transform: uppercase;
}
th {
  text-align: left;
}
td {
  vertical-align: top;
  padding: .25em .25em .25em .4em;
}

.ty-table-edit {
  background-color: transparent;
}
thead {
  background-color: #dadada;
}
tr:nth-child(even) {
  background: #e8e7e7;
}

ul, #myUL {
  list-style-type: none;
  padding-inline-start:10px;
}



/* Remove margins and padding from the parent ul */
#myUL {
  margin: 0;
  padding: 0;
}

/* Style the caret/arrow */
.caret {
  cursor: pointer;
  user-select: none; /* Prevent text selection */
}

/* Create the caret/arrow with a unicode, and style it */
.caret::before {
  content: "\\25B6";
  color: black;
  display: inline-block;
  margin-right: 6px;
}

/* Rotate the caret/arrow icon when clicked on (using JavaScript) */
.caret-down::before {
  transform: rotate(90deg);
}

/* Hide the nested list */
.nested {
  display: none;
}

/* Show the nested list when the user clicks on the caret/arrow (with JavaScript) */
.active {
  display: block;
}

</style>
"""

script="""<script type="text/javascript">
var toggler = document.getElementsByClassName("caret");
var i;
for (i = 0; i < toggler.length; i++) {
  toggler[i].addEventListener("click", function() {
    this.parentElement.querySelector(".nested").classList.toggle("active");
    this.classList.toggle("caret-down");
  });
}</script>"""


class HTMLToc:
  def __init__(self,output):
    self._id=0
    self._sectionID = 0
    self._output = output

  

  def visitTable(self,table):
      pass


  def visitSection(self,section):
     self._id = self._id + 1 
     self._sectionID = self._sectionID + 1
     if section.hasChildren:
        self._output.write("<li><span class=\"caret\"><a href=\"#section%d\">%s</a></span>\n" % (self._sectionID,section.name))
        self._output.write("<ul class=\"nested\">\n")
     else:
        self._output.write("<li><span><a href=\"#section%d\">%s</a></span>\n" % (self._sectionID,section.name))

  def leaveSection(self,section):
    if section.hasChildren:
       self._output.write("</ul></li>\n")

    self._id = self._id - 1 

  def visitDocument(self,document):
      self._output.write("<div id=\"TOC\"><h1>Table of content</h1><ul id=\"myUL\">\n")


  def leaveDocument(self,document):
      self._output.write("</ul></div>%s\n" % script)


class HTML:
  def __init__(self,output,regMode):
    self._id=0
    self._sectionID = 0
    self._output = output
    self._regMode = regMode

  

  def visitTable(self,table):
      output.write("<table>\n")
      output.write("<thead>\n")
      output.write("<tr>\n")
      for col in table.columns:
        output.write("<th>")
        output.write(str(col))
        output.write("</th>\n")
      output.write("</tr>\n")
      output.write("</thead>\n")
      for row in table.rows:
        output.write("<tr>\n")
        for elem in row:
            output.write("<td>")
            output.write(str(elem))
            output.write("</td>\n")
        output.write("</tr>\n")
      output.write("</table>\n")


  def visitSection(self,section):
     self._id = self._id + 1 
     self._sectionID = self._sectionID + 1
     output.write("<h%d id=\"section%d\">%s</h%d>\n" % (self._id,self._sectionID,section.name,self._id))

  def leaveSection(self,section):
     self._id = self._id - 1 

  def visitDocument(self,document):
      self._output.write("""<!doctype html>
<html>
<head>
<meta charset='UTF-8'><meta name='viewport' content='width=device-width initial-scale=1'>
<title>Benchmarks</title>%s</head><body>\n""" % styleSheet)
      if self._regMode:
         self._output.write("<h1>ECPS Benchmark Regressions</h1>\n")
      else:
         self._output.write("<h1>ECPS Benchmark Summary</h1>\n")
      self._output.write("<p>Run number %d on %s</p>\n" % (document.runid, str(document.date)))

  def leaveDocument(self,document):
    document.accept(HTMLToc(self._output))

    self._output.write("</body></html>\n")





# Command to get last runid 
lastID="""SELECT runid FROM RUN ORDER BY runid DESC LIMIT 1
"""

# Command to get last runid and date
lastIDAndDate="""SELECT date FROM RUN WHERE runid=?
"""

def getLastRunID():
  r=c.execute(lastID)
  return(int(r.fetchone()[0]))

def getrunIDDate(forID):
  r=c.execute(lastIDAndDate,(forID,))
  return(r.fetchone()[0])

runid = 1

parser = argparse.ArgumentParser(description='Generate summary benchmarks')

parser.add_argument('-b', nargs='?',type = str, default="bench.db", help="Benchmark database")
parser.add_argument('-o', nargs='?',type = str, default="full.md", help="Full summary")
parser.add_argument('-r', action='store_true', help="Regression database")
parser.add_argument('-t', nargs='?',type = str, default="md", help="md,html")

# For runid or runid range
parser.add_argument('others', nargs=argparse.REMAINDER,help="Run ID")

args = parser.parse_args()

c = sqlite3.connect(args.b)

if args.others:
   runid=int(args.others[0])
else:
   runid=getLastRunID()

# We extract data only from data tables
# Those tables below are used for descriptions
REMOVETABLES=['TESTNAME','TESTDATE','RUN','CORE', 'PLATFORM', 'COMPILERKIND', 'COMPILER', 'TYPE', 'CATEGORY', 'CONFIG']

# This is assuming the database is generated by the regression script
# So platform is the same for all benchmarks.
# Category and type is coming from the test name in the yaml
# So no need to add this information here
# Name is removed here because it is added at the beginning
REMOVECOLUMNS=['runid','name','type','platform','category','coredef','OPTIMIZED','HARDFP','FASTMATH','NEON','HELIUM','UNROLL','ROUNDING','DATE','compilerkindid','date','categoryid', 'ID', 'platformid', 'coreid', 'compilerid', 'typeid']

# Get existing benchmark tables
def getBenchTables():
    r=c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    benchtables=[]
    for table in r:
        if not table[0] in REMOVETABLES:
          benchtables.append(table[0])
    return(benchtables)

# get existing types in a table
def getExistingTypes(benchTable):
    r=c.execute("select distinct typeid from %s order by typeid desc" % benchTable).fetchall()
    result=[x[0] for x in r]
    return(result)

# Get compilers from specific type and table
allCompilers="""select distinct compilerid from %s WHERE typeid=?"""

compilerDesc="""select compiler,version from COMPILER 
  INNER JOIN COMPILERKIND USING(compilerkindid) WHERE compilerid=?"""

# Get existing compiler in a table for a specific type
# (In case report is structured by types)
def getExistingCompiler(benchTable,typeid):
    r=c.execute(allCompilers % benchTable,(typeid,)).fetchall()
    return([x[0] for x in r])

def getCompilerDesc(compilerid):
    r=c.execute(compilerDesc,(compilerid,)).fetchone()
    return(r)

# Get type name from type id
def getTypeName(typeid):
    r=c.execute("select type from TYPE where typeid=?",(typeid,)).fetchone()
    return(r[0])
 
# Diff of 2 lists 
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]


# Command to get data for specific compiler 
# and type
benchCmd="""select %s from %s
  INNER JOIN CATEGORY USING(categoryid)
  INNER JOIN PLATFORM USING(platformid)
  INNER JOIN CORE USING(coreid)
  INNER JOIN COMPILER USING(compilerid)
  INNER JOIN COMPILERKIND USING(compilerkindid)
  INNER JOIN TYPE USING(typeid)
  INNER JOIN TESTNAME USING(testnameid)
  WHERE compilerid=? AND typeid = ? AND runid = ?
  """


# Command to get test names for specific compiler 
# and type
benchNames="""select distinct name from %s
  INNER JOIN COMPILER USING(compilerid)
  INNER JOIN COMPILERKIND USING(compilerkindid)
  INNER JOIN TYPE USING(typeid)
  INNER JOIN TESTNAME USING(testnameid)
  WHERE compilerid=? AND typeid = ? AND runid = ?
  """

# Command to get columns for specific table
benchCmdColumns="""select * from %s
  INNER JOIN CATEGORY USING(categoryid)
  INNER JOIN PLATFORM USING(platformid)
  INNER JOIN CORE USING(coreid)
  INNER JOIN COMPILER USING(compilerid)
  INNER JOIN COMPILERKIND USING(compilerkindid)
  INNER JOIN TESTNAME USING(testnameid)
  INNER JOIN TYPE USING(typeid)
  """

def joinit(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

# Is not a column name finishing by id 
# (often primary key for thetable)
def isNotIDColumn(col):
    if re.match(r'^.*id$',col):
        return(False)
    else:
        return(True)
    
# Get test names
# for specific typeid and compiler (for the data)
def getTestNames(benchTable,comp,typeid):
    vals=(comp,typeid,runid)
    result=c.execute(benchNames % benchTable,vals).fetchall()
    return([x[0] for x in list(result)])

# Command to get data for specific compiler 
# and type
nbElemsInBenchAndTypeAndCompilerCmd="""select count(*) from %s
  WHERE compilerid=? AND typeid = ? AND runid = ?
  """

nbElemsInBenchAndTypeCmd="""select count(*) from %s
  WHERE typeid = ? AND runid = ?
  """

nbElemsInBenchCmd="""select count(*) from %s
  WHERE runid = ?
  """

categoryName="""select distinct category from %s
  INNER JOIN CATEGORY USING(categoryid)
  WHERE runid = ?
  """

def getCategoryName(benchTable,runid):
  result=c.execute(categoryName % benchTable,(runid,)).fetchone()
  return(result[0])

# Get nb elems in a table
def getNbElemsInBenchAndTypeAndCompilerCmd(benchTable,comp,typeid):
    vals=(comp,typeid,runid)
    result=c.execute(nbElemsInBenchAndTypeAndCompilerCmd % benchTable,vals).fetchone()
    return(result[0])

def getNbElemsInBenchAndTypeCmd(benchTable,typeid):
    vals=(typeid,runid)
    result=c.execute(nbElemsInBenchAndTypeCmd % benchTable,vals).fetchone()
    return(result[0])

def getNbElemsInBenchCmd(benchTable):
    vals=(runid,)
    result=c.execute(nbElemsInBenchCmd % benchTable,vals).fetchone()
    return(result[0])

# Get names of columns and data for a table
# for specific typeid and compiler (for the data)
def getColNamesAndData(benchTable,comp,typeid):
    cursor=c.cursor()
    result=cursor.execute(benchCmdColumns % (benchTable))
    cols= [member[0] for member in cursor.description]
    keepCols = ['name'] + [c for c in diff(cols , REMOVECOLUMNS) if isNotIDColumn(c)]
    keepColsStr = "".join(joinit(keepCols,","))
    vals=(comp,typeid,runid)
    result=cursor.execute(benchCmd % (keepColsStr,benchTable),vals)
    vals =np.array([list(x) for x in list(result)])
    return(keepCols,vals)



PARAMS=["NB","NumTaps", "NBA", "NBB", "Factor", "NumStages","VECDIM","NBR","NBC","NBI","IFFT", "BITREV"]

def regressionTableFor(name,section,ref,toSort,indexCols,field):
    data=ref.pivot_table(index=indexCols, columns='core', 
    values=[field], aggfunc='first')
       
    data=data.sort_values(toSort)
       
    cores = [c[1] for c in list(data.columns)]
    columns = diff(indexCols,['name'])

    dataTable=Table(columns,cores)
    section.addTable(dataTable)

    dataForFunc=data.loc[name]
    if type(dataForFunc) is pd.DataFrame:
       for row in dataForFunc.itertuples():
           row=list(row)
           if type(row[0]) is int:
              row=[row[0]] + row[1:]
           else: 
              row=list(row[0]) + row[1:]
           dataTable.addRow(row)
    else:
       dataTable.addRow(dataForFunc)

def formatTableByCore(typeSection,testNames,cols,vals):
    if vals.size != 0:
       ref=pd.DataFrame(vals,columns=cols)
       toSort=["name"]
       
       for param in PARAMS:
          if param in ref.columns:
             ref[param]=pd.to_numeric(ref[param])
             toSort.append(param)
       if args.r:
         #  Regression table
         ref['MAX']=pd.to_numeric(ref['MAX'])
         ref['MAXREGCOEF']=pd.to_numeric(ref['MAXREGCOEF'])
       
         indexCols=diff(cols,['core','Regression','MAXREGCOEF','MAX','version','compiler'])
         valList = ['Regression']
       else:
         ref['CYCLES']=pd.to_numeric(ref['CYCLES'])
       
         indexCols=diff(cols,['core','CYCLES','version','compiler'])
         valList = ['CYCLES']
      
       

       for name in testNames:
           if args.r:
              testSection = Section(name)
              typeSection.addSection(testSection)

              regressionSection = Section("Regression")
              testSection.addSection(regressionSection)
              regressionTableFor(name,regressionSection,ref,toSort,indexCols,'Regression')
              
              maxCyclesSection = Section("Max cycles")
              testSection.addSection(maxCyclesSection)
              regressionTableFor(name,maxCyclesSection,ref,toSort,indexCols,'MAX')
              
              maxRegCoefSection = Section("Max Reg Coef")
              testSection.addSection(maxRegCoefSection)
              regressionTableFor(name,maxRegCoefSection,ref,toSort,indexCols,'MAXREGCOEF')

           else:
              data=ref.pivot_table(index=indexCols, columns='core', 
              values=valList, aggfunc='first')
       
              data=data.sort_values(toSort)
       
              cores = [c[1] for c in list(data.columns)]
              columns = diff(indexCols,['name'])

              testSection = Section(name)
              typeSection.addSection(testSection)

              dataTable=Table(columns,cores)
              testSection.addTable(dataTable)

              dataForFunc=data.loc[name]
              if type(dataForFunc) is pd.DataFrame:
                 for row in dataForFunc.itertuples():
                     row=list(row)
                     if type(row[0]) is int:
                        row=[row[0]] + row[1:]
                     else: 
                        row=list(row[0]) + row[1:]
                     dataTable.addRow(row)
              else:
                 dataTable.addRow(dataForFunc)

# Add a report for each table
def addReportFor(document,benchName):
    nbElems = getNbElemsInBenchCmd(benchName)
    if nbElems > 0:
       categoryName = getCategoryName(benchName,document.runid)
       benchSection = Section(categoryName)
       document.addSection(benchSection)
       print("Process %s\n" % benchName)
       allTypes = getExistingTypes(benchName)
       # Add report for each type
       for aTypeID in allTypes:
           nbElems = getNbElemsInBenchAndTypeCmd(benchName,aTypeID)
           if nbElems > 0:
              typeName = getTypeName(aTypeID)
              typeSection = Section(typeName)
              benchSection.addSection(typeSection)
              ## Add report for each compiler
              allCompilers = getExistingCompiler(benchName,aTypeID)
              for compiler in allCompilers:
                  #print(compiler)
                  nbElems = getNbElemsInBenchAndTypeAndCompilerCmd(benchName,compiler,aTypeID)
                  # Print test results for table, type, compiler
                  if nbElems > 0:
                     compilerName,version=getCompilerDesc(compiler)
                     compilerSection = Section("%s (%s)" % (compilerName,version))
                     typeSection.addSection(compilerSection)
                     cols,vals=getColNamesAndData(benchName,compiler,aTypeID)
                     names=getTestNames(benchName,compiler,aTypeID)
                     formatTableByCore(compilerSection,names,cols,vals)
           




try:
      benchtables=getBenchTables()
      theDate = getrunIDDate(runid)
      document = Document(runid,theDate)
      for bench in benchtables:
          addReportFor(document,bench)
      with open(args.o,"w") as output:
          if args.t=="md":
             document.accept(Markdown(output))
          if args.t=="html":
             document.accept(HTML(output,args.r))

finally:
     c.close()

    


