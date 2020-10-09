from pyparsing import *
import TestScripts.Parser as p
import pickle

def loadRoot(f):
    root = None
    with open(f,"rb") as inf:
         root=pickle.load(inf)
    return(root)

class Params:
    def __init__(self):
        self.full = []
        self.summary=None
        self.paramNames = None
        self.formula=""

    def __str__(self):
        return(str(self.full) + str(self.summary) + str(self.paramNames))

def disabled(a):
  return((CaselessKeyword("disabled") + "{" + a + "}") ^ a)

def parsedNodeDesc( str, loc, toks ):
    d={}
    if "class" in toks:
       d["class"]=toks["class"]
    if "folder" in toks:
       d["folder"]=toks["folder"]
    return(d)

def parseNodeSuite( str, loc, toks):
    d={}
    t = p.TreeElem(0)
    if "message" in toks:
       d["message"]=toks["message"]
    if "class" in toks["desc"]:
        d["class"] = toks["desc"]["class"]
    if "folder" in toks["desc"]:
        t.setPath(toks["desc"]["folder"])
    d["deprecated"] = False
    if(toks[0]=="disabled"):
        d["deprecated"] = True

    if "PARAMID" in toks:
        d["PARAMID"] = toks["PARAMID"]
    
    t.writeData(d)

    if "params" in toks:
        t.params=toks["params"]

    for c in toks["allTests"]:
        t.addGroup(c)
    if "allPatterns" in toks:
       for c in toks["allPatterns"]:
           t.addPattern(c["ID"],c["path"])

    if "allParams" in toks:
       for c in toks["allParams"]:
           if "path" in c:
              t.addParam(p.TreeElem.PARAMFILE,c["ID"],c["path"])
           if "numberList" in c:
              #print(c["numberList"])
              t.addParam(p.TreeElem.PARAMGEN,c["ID"],c["numberList"])

    if "allOutputs" in toks:
       for c in toks["allOutputs"]:
           t.addOutput(c["ID"],c["path"])
    return(t)

def parseNodeGroup( str, loc, toks):
    d={}
    t = p.TreeElem(0)
    if "message" in toks:
       d["message"]=toks["message"]
    if "class" in toks["desc"]:
        d["class"] = toks["desc"]["class"]
    if "folder" in toks["desc"]:
        t.setPath(toks["desc"]["folder"])
    d["deprecated"] = False
    if(toks[0]=="disabled"):
        d["deprecated"] = True
    
    t.writeData(d)

    #print(t.data["message"])
    for c in toks["contained"]:
        #print("  ",c.data["message"])
        t.addGroup(c)
    return(t)

def parseTest( str, loc, toks):
    d={}
    if "message" in toks:
       d["message"]=toks["message"]
    if "class" in toks:
        d["class"] = toks["class"]
    d["deprecated"] = False
    if(toks[0]=="disabled"):
        d["deprecated"] = True
    if "PARAMID" in toks:
        d["PARAMID"] = toks["PARAMID"]
    if "testData" in toks:
        d["testData"]=toks["testData"]
    t = p.TreeElem(0)
    t.writeData(d)
    return(t)

def getInteger( str, loc, toks):
    return(int(toks[0]))

def parseFile( str, loc, toks):
    d={}
    d["ID"] = toks["ID"]
    d["path"] = toks["path"]
    return(d)

def parseParamDesc( str, loc, toks):
    d={}
    d["ID"] = toks["ID"]
    if "path" in toks:
       d["path"] = toks["path"]
    if "numberList" in toks:
       d["numberList"] = toks["numberList"]
    return(d)
    
def parseParams( str, loc, toks):
    p = Params() 
    p.full = toks["full"]
    if "summary" in toks:
        p.summary=toks["summary"]
    if "names" in toks:
        p.paramNames=[x.strip("\"") for x in toks["names"]]
    if "formula" in toks:
        p.formula=toks["formula"].strip("\"")
    return(p)

def generatorDesc( str, loc, toks):
    d={} 
    r = list(toks["ints"])
    d["NAME"] = toks["PARAM"]
    d["INTS"] = r
    return(d)

def parseTestFields( str, loc, toks):
    if "fields" in toks:
        fields = toks["fields"]
        # merge list of dictionnaries into a dictionnary
        newFields = dict((key,d[key]) for d in fields for key in d)
        return(newFields)

def parseField( str, loc, toks):
    if toks[0] == "oldID":
       return({"oldID" : int(toks[2])})
    if toks[0] == "truc":
       return({"truc" : int(toks[2])})

class Parser:
    def __init__(self):
        self.id = 0

    def parse(self, filePath):
        string = Word(alphanums+"_ =+()")
        ident = Word( alphas+"_", alphanums+"_" )

        path = Word(alphanums+"_/.")

        folder = CaselessKeyword("folder") + "=" + path("folder")

        nodeDesc = CaselessKeyword("class") + "=" + ident("class") + Optional(folder)
        nodeDesc = nodeDesc.setParseAction(parsedNodeDesc)

        patterns = (Keyword("Pattern")  + ident("ID") + ":" + path("path")).setParseAction(parseFile)
        output = (Keyword("Output") + ident("ID") + ":" + path("path")).setParseAction(parseFile)
        
        integer =  Combine( Optional(Word("+-")) + Word(nums) ).setParseAction(getInteger)
        numberList = (ident("PARAM") + Literal("=") + "[" + delimitedList(integer,",")("ints") + "]").setParseAction(generatorDesc)
        generator = Literal("=") + "{" + OneOrMore(numberList)("numberList") + "}"
        fileOrGenerator = (":" + path("path")) | generator
        params = (Keyword("Params")  + ident("ID") + fileOrGenerator).setParseAction(parseParamDesc)

        paramValue = Literal("->") + ident("PARAMID")

        messFormat = Word(alphanums + " _/")
        message = messFormat("message")

        testField = ((Keyword("oldID") + "=" + integer("INT")) | (Keyword("truc") + "=" + integer("INT"))).setParseAction(parseField)
        testData = (Literal("{") + OneOrMore(testField)("fields") + Literal("}")).setParseAction(parseTestFields)
        test = disabled((string("message") + ":" + ident("class") + Optional(testData("testData")) +  Optional(paramValue))).setParseAction(parseTest)
        
        # paramDescription =
        # File or
        # List of int or
        # Cartesian products of list
        # Can be applied to global pattern
        # or on a per test basis
        paramDescription = ""

        full = delimitedList(ident,",")
        formula = Keyword("Formula") + dblQuotedString("formula")
        paramNames = Keyword("Names") + delimitedList(dblQuotedString,",")("names")
        summary = Keyword("Summary") + delimitedList(ident,",")("summary")
        
        paramDetails = full("full") + Optional(summary) + Optional(paramNames)+ Optional(formula)

        paramDesc=Keyword("ParamList") + Literal("{")  + paramDetails + Literal("}") 

        allTests = Keyword("Functions") + "{" + OneOrMore(test)("allTests") + "}" + Optional(paramValue)

        allPatterns =  ZeroOrMore(patterns)
        allOutputs =  ZeroOrMore(output)
        allParams = ZeroOrMore(params)
        paramList = Optional(paramDesc("params").setParseAction(parseParams))

        suiteDesc = paramList + allPatterns("allPatterns") + allOutputs("allOutputs") + allParams("allParams") + allTests

        suite = disabled(CaselessKeyword("suite") + message + Literal("{") + nodeDesc("desc") + suiteDesc + Literal("}"))
        suite = suite.setParseAction(parseNodeSuite)


        group = Forward()
        contained = OneOrMore(group | suite)

        
        group << disabled(CaselessKeyword("group") + message + Literal("{") + nodeDesc("desc") + contained("contained") + Literal("}"))
        group=group.ignore(cStyleComment | ("//" + restOfLine ))
        group = group.setParseAction(parseNodeGroup)


        tree = group.parseFile( filePath )
        tree[0].classify()
        # We compute ID of all nodes.
        tree[0].computeId()
        tree[0].reident(0)
        return(tree[0])
