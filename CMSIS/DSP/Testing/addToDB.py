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
import sqlite3
import datetime, time
import re 

# For sql table creation
MKSTRFIELD=[]
MKBOOLFIELD=['HARDFP', 'FASTMATH', 'NEON', 'HELIUM','UNROLL', 'ROUNDING','OPTIMIZED']
MKINTFIELD=['ID', 'CYCLES']
MKDATEFIELD=[]
MKKEYFIELD=['DATE','NAME','CATEGORY', 'PLATFORM', 'CORE', 'COMPILER','TYPE',"RUN"]
MKKEYFIELDID={'CATEGORY':'categoryid', 
   'NAME':'testnameid',
   'DATE':'testdateid',
   'PLATFORM':'platformid', 
   'CORE':'coreid', 
   'COMPILER':'compilerid',
   'TYPE':'typeid',
   'RUN':'runid'}

# For csv table value extraction
VALSTRFIELD=['TESTNAME','VERSION']
VALBOOLFIELD=['HARDFP', 'FASTMATH', 'NEON', 'HELIUM','UNROLL', 'ROUNDING','OPTIMIZED']
VALINTFIELD=['ID', 'CYCLES']
VALDATEFIELD=[]
# Some of those fields may be created by the parsing of other fields
VALKEYFIELD=['DATE','NAME','CATEGORY', 'PLATFORM', 'CORE', 'COMPILER','TYPE']

def joinit(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

def tableExists(c,tableName):
   req=(tableName,)
   r=c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",req)
   return(r.fetchone() != None)

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def getColumns(elem,full):
  colsToKeep=[]
  cols = list(full.columns)
  params = list(elem.params.full)
  common = diff(cols + ["TYPE","RUN"] , ['OLDID'] + params)  
 
  for field in common:
       if field in MKSTRFIELD:
          colsToKeep.append(field)
       if field in MKINTFIELD:
          colsToKeep.append(field)
       if field in MKKEYFIELD:
          colsToKeep.append(field)
       if field in MKDATEFIELD:
          colsToKeep.append(field)
       if field in MKBOOLFIELD:
          colsToKeep.append(field)
  return(colsToKeep)

def createTableIfMissing(conn,elem,tableName,full):
   if not tableExists(conn,tableName):
     sql = "CREATE TABLE %s (" % tableName
     cols = list(full.columns)
     params = list(elem.params.full)
     common = diff(cols + ["TYPE","RUN"] , ['OLDID'] + params)

     sql += "%sid INTEGER PRIMARY KEY"  % (tableName)
     start = ","   

     for field in params:
       sql += " %s\n  %s INTEGER"  % (start,field)
       start = ","

     for field in common:
       if field in MKSTRFIELD:
          sql += "%s\n  %s TEXT"  % (start,field)
       if field in MKINTFIELD:
          sql += "%s\n  %s INTEGER"  % (start,field)
       if field in MKKEYFIELD:
          sql += "%s\n  %s INTEGER"  % (start,MKKEYFIELDID[field])
       if field in MKDATEFIELD:
          sql += "%s\n  %s TEXT"  % (start,field)
       if field in MKBOOLFIELD:
          sql += "%s\n  %s INTEGER"  % (start,field)
       start = ","
     # Create foreign keys
     sql += "%sFOREIGN KEY(typeid) REFERENCES TYPE(typeid)," % start
     sql += "FOREIGN KEY(categoryid) REFERENCES CATEGORY(categoryid),"
     sql += "FOREIGN KEY(testnameid) REFERENCES TESTNAME(testnameid),"
     sql += "FOREIGN KEY(testdateid) REFERENCES TESTDATE(testdateid),"
     sql += "FOREIGN KEY(platformid) REFERENCES PLATFORM(platformid),"
     sql += "FOREIGN KEY(coreid) REFERENCES CORE(coreid),"
     sql += "FOREIGN KEY(compilerid) REFERENCES COMPILER(compilerid)"
     sql += "FOREIGN KEY(runid) REFERENCES RUN(runid)"
     sql += "  )"
     conn.execute(sql)

# Find the key or add it in a table
def findInTable(conn,table,keystr,strv,key):
    #print(sql)
    r = conn.execute("select %s from %s where %s=?" % (key,table,keystr),(strv,))
    result=r.fetchone()
    if result != None:
      return(result[0])
    else:
      conn.execute("INSERT INTO %s(%s) VALUES(?)" % (table,keystr),(strv,))
      conn.commit()
      r = conn.execute("select %s from %s where %s=?" % (key,table,keystr),(strv,))
      result=r.fetchone()
      if result != None:
         #print(result)
         return(result[0])
      else:
         return(None)

def findInCompilerTable(conn,kind,version):
    #print(sql)
    r = conn.execute("select compilerid from COMPILER where compilerkindid=? AND version=?"  , (kind,version))
    result=r.fetchone()
    if result != None:
      return(result[0])
    else:
      fullDate = datetime.datetime.now()
      dateid = findInTable(conn,"TESTDATE","date",str(fullDate),"testdateid")
      conn.execute("INSERT INTO COMPILER(compilerkindid,version,testdateid) VALUES(?,?,?)" ,(kind,version,dateid))
      conn.commit()
      r = conn.execute("select compilerid from COMPILER where compilerkindid=? AND version=? AND testdateid=?"  , (kind,version,dateid))
      result=r.fetchone()
      if result != None:
         #print(result)
         return(result[0])
      else:
         return(None)


def addRows(conn,elem,tableName,full,runid=0):
   # List of columns we have in DB which is
   # different from the columns in the table
   compilerid = 0
   platformid = 0 
   coreid = 0
   keep = getColumns(elem,full)
   cols = list(full.columns)
   params = list(elem.params.full)
   common = diff(["TYPE"] + cols , ['OLDID'] + params)  
   colNameList = [] 
   for c in params + keep:
      if c in MKKEYFIELD:
          colNameList.append(MKKEYFIELDID[c])
      else:
         colNameList.append(c)
   colNames = "".join(joinit(colNameList,","))
   #print(colNameList)
   #print(colNames)
   #print(full)
   for index, row in full.iterrows():
       sql = "INSERT INTO %s(%s) VALUES(" % (tableName,colNames)
       keys = {}

       # Get data from columns
       for field in common:
        if field in VALSTRFIELD:
            keys[field]=row[field]
            if field == "NAME":
                name = row[field]
            if field == "TESTNAME":
                testname = row[field]
                if re.match(r'^.*_f64',testname):
                  keys["TYPE"] = "f64"
                if re.match(r'^.*_f32',testname):
                  keys["TYPE"] = "f32"
                if re.match(r'^.*_f16',testname):
                  keys["TYPE"] = "f16"
                if re.match(r'^.*_q31',testname):
                  keys["TYPE"] = "q31"
                if re.match(r'^.*_q15',testname):
                  keys["TYPE"] = "q15"
                if re.match(r'^.*_q7',testname):
                  keys["TYPE"] = "q7"

                if re.match(r'^.*_s8',testname):
                  keys["TYPE"] = "s8"
                if re.match(r'^.*_u8',testname):
                  keys["TYPE"] = "u8"
                if re.match(r'^.*_s16',testname):
                  keys["TYPE"] = "s16"
                if re.match(r'^.*_u16',testname):
                  keys["TYPE"] = "u16"
                if re.match(r'^.*_s32',testname):
                  keys["TYPE"] = "s32"
                if re.match(r'^.*_u32',testname):
                  keys["TYPE"] = "u32"
                if re.match(r'^.*_s64',testname):
                  keys["TYPE"] = "s64"
                if re.match(r'^.*_u64',testname):
                  keys["TYPE"] = "u64"
            
        if field in VALINTFIELD:
            keys[field]=row[field]
        if field in VALDATEFIELD:
            keys[field]=row[field]
        if field in VALBOOLFIELD:
            keys[field]=row[field]
        
         
       keys['RUN']=runid
       # Get foreign keys and create missing data
       for field in common:
        if field in VALKEYFIELD:
            if field == "CATEGORY":
              # Remove type extension to get category name so that
              # all types are maped to same category which will
              # help for post processing.
              testField=re.sub(r'^(.*)[:]([^:]+)(F16|F32|F64|Q31|Q15|Q7)$',r'\1',row[field])
              val = findInTable(conn,"CATEGORY","category",testField,"categoryid")
              keys[field]=val
            if field == "NAME":
              val = findInTable(conn,"TESTNAME","name",row[field],"testnameid")
              keys[field]=val
            if field == "DATE":
              val = findInTable(conn,"TESTDATE","date",str(row[field]),"testdateid")
              keys[field]=val
            if field == "CORE":
              val = findInTable(conn,"CORE","coredef",row[field],"coreid")
              keys[field]=val
              coreid = val
            if field == "PLATFORM":
              val = findInTable(conn,"PLATFORM","platform",row[field],"platformid")
              keys[field]=val
              platformid = val
            if field == "TYPE":
              val = findInTable(conn,"TYPE","type",keys["TYPE"],"typeid")
              keys[field]=val
            if field == "COMPILER":
              compilerkind = findInTable(conn,"COMPILERKIND","compiler",row[field],"compilerkindid")
              compiler = findInCompilerTable(conn,compilerkind,keys["VERSION"])
              keys[field]=compiler
              compilerid = compiler

       # Generate sql command
       start = ""  
       for field in params:
         sql += " %s\n  %d"  % (start,row[field])
         start = ","
         
       for field in keep:
         if field in MKSTRFIELD or field in MKDATEFIELD:
            sql += " %s\n  \"%s\""  % (start,keys[field])
         elif field in keep:
            sql += " %s\n  %d"  % (start,keys[field])
         start = ","

       sql += "  )"
       #print(sql)
       conn.execute(sql)  
   conn.commit() 
   return({'compilerid':compilerid,'platformid':platformid,'coreid':coreid})

def addConfig(conn,config,fullDate):
  dateid = findInTable(conn,"TESTDATE","date",str(fullDate),"testdateid")
  conn.execute("INSERT INTO CONFIG(compilerid,platformid,coreid,testdateid) VALUES(?,?,?,?)" ,(config['compilerid'],config['platformid'],config['coreid'],dateid))
  conn.commit()

def getGroup(a):
    return(re.sub(r'^(.+)(F64|F32|F16|Q31|Q15|Q7|U32|U16|U8|S32|S16|S8)$',r'\1',a))

def addOneBenchmark(elem,fullPath,db,group,runid):
   if os.path.isfile(fullPath):
      full=pd.read_csv(fullPath,dtype={'OLDID': str} ,keep_default_na = False)
      fullDate = datetime.datetime.now()
      full['DATE'] = fullDate
      if group:
         tableName = getGroup(group)
      else:
         tableName = getGroup(elem.data["class"])
      conn = sqlite3.connect(db)
      createTableIfMissing(conn,elem,tableName,full)
      config = addRows(conn,elem,tableName,full,runid)
      addConfig(conn,config,fullDate)
      conn.close()


def addToDB(benchmark,dbpath,elem,group,runid):
  if not elem.data["deprecated"]:
     if elem.params:
         benchPath = os.path.join(benchmark,elem.fullPath(),"fullBenchmark.csv")
         print("Processing %s" % benchPath)
         addOneBenchmark(elem,benchPath,dbpath,group,runid)
         
     for c in elem.children:
       addToDB(benchmark,dbpath,c,group,runid)



parser = argparse.ArgumentParser(description='Generate summary benchmarks')

parser.add_argument('-f', nargs='?',type = str, default="Output.pickle", help="Pickle")
parser.add_argument('-b', nargs='?',type = str, default="FullBenchmark", help="Full Benchmark dir path")
#parser.add_argument('-e', action='store_true', help="Embedded test")
parser.add_argument('-o', nargs='?',type = str, default="bench.db", help="Benchmark database")
parser.add_argument('-r', nargs='?',type = int, default=0, help="Run ID")

parser.add_argument('others', nargs=argparse.REMAINDER, help="Suite class")

args = parser.parse_args()

if args.f is not None:
    #p = parse.Parser()
    # Parse the test description file
    #root = p.parse(args.f)
    root=parse.loadRoot(args.f)
    d.deprecate(root,args.others)
    if args.others:
      group=args.others[0] 
    else:
      group=None
    addToDB(args.b,args.o,root,group,args.r)
    
else:
    parser.print_help()