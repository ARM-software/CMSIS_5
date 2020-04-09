import re 
import os 

class TreeElem:
    """ Result of the parsing of the test description.

    It is a tree of objects describing the groups, suites and tests

    Attributes:
        kind (int) : Node kind
        ident (int) : Indentation level in the test description.
          It is used to format output of test results
        parent (TreeElem) : parent of this node
        id (int) : Node id
        patterns (list) : List of pairs
         Each pair is a pattern ID and pattern path
        outputs (list) : List of pairs
         Each pair is an output ID and an output path

    """

    TEST = 1
    SUITE = 2
    GROUP = 3

    PARAMFILE = 1
    PARAMGEN = 2

    def __init__(self,ident):
        self.kind=TreeElem.TEST
        self.ident = ident 
        self._children = [] 
        self.parent = None
        self._data = None
        self.id = 1
        self._path=""
        self.patterns=[]
        self.outputs=[]
        # List of parameters files
        self.parameters=[]
        # List of arguments
        self.params = None

    def __str__(self):
        """ Convert the TreeElem into a string for debug purpose
        """
        if self.kind == TreeElem.TEST:
            g="Test"
        if self.kind == TreeElem.SUITE:
            g="Suite"
        if self.kind == TreeElem.GROUP:
            g="Group"
        a = str("%s -> %s%s(%d)\n" % (g,' ' * self.ident, str(self.data),self.id))
        if self.params:
          a = a + str(self.params.full) + "\n" + str(self.params.summary) + "\n" + str(self.params.paramNames) + "\n"
        for i in self._children:
            a = a + str(i)
        return(a)

    def setData(self,data):
        """ Set the data property of this node

        Args:
          data (list) : A list of fields for this node
              The fields are parsed and a data dictionary is created
          fpga (bool) : false in semihosting mode
        Raises:
          Nothing 
        Returns:
          Nothing
        """
        d = {} 

        # A node OFF in the list is deprecated. It won't be included
        # or executed in the final tests
        # but it will be taken into account for ID generation
        d["deprecated"] = False
        # Text message to display to the user zhen displaying test result
        # This text message is never used in any .txt,.cpp or .h
        # generated. It is only for better display of the test
        # results
        d["message"] = data[0].strip()
        # CPP class or function name to use
        if len(data) > 1:
            d["class"] = data[1].strip()
        if len(data) == 3:
            if data[2].strip() == "OFF":
               d["deprecated"] = True
            else:
                self._path = data[2].strip()
        # New path for this node (when we want a new subfolder
        # for the patterns or output of a group of suite)
        if len(data) == 4:
            self._path = data[3].strip()

        self._data = d

    @property
    def data(self):
        return(self._data)

    def writeData(self,d):
        self._data=d

    def setPath(self,p):
        self._path=p

    @property
    def path(self):
        return(self._path)

    @property
    def children(self):
        return(self._children)

    def _fullPath(self):
      if self.parent:
         return(os.path.join(self.parent._fullPath() , self.path))
      else:
         return("")

    def fullPath(self):
      return(os.path.normpath(self._fullPath()))

    def categoryDesc(self):
      if self.parent:
         p = self.parent.categoryDesc() 
         if p and self.path:
            return(p + ":" + self.path)
         if p:
            return(p)
         if self.path:
            return(self.path)
      else:
         return("")

    def addGroup(self,g):
        """ Add a group to this node

        Args:
          g (TreeElem) : group to add
        Raises:
          Nothing 
        Returns:
          Nothing
        """
        g.parent = self
        self._children.append(g)

    def classify(self):
        """ Compute the node kind recursively

        Node kind is infered from the tree structure and not present
        in the test description.
        A suite is basically a leaf of the tree and only contain tests.
        A group is containing a suite or another group.

        """
        r = TreeElem.TEST
        for c in self._children:
            c.classify()

       
        for c in self._children:
            if c.kind == TreeElem.TEST and r != TreeElem.GROUP:
                r = TreeElem.SUITE
            if c.kind == TreeElem.SUITE:
                r = TreeElem.GROUP
            if c.kind == TreeElem.GROUP:
                r = TreeElem.GROUP
        self.kind = r

    def computeId(self):
        """ Compute the node ID and the node param ID
        """
        i = 1
        for c in self._children:
            c.id = i
            if not "PARAMID" in c.data and "PARAMID" in self.data:
              c.data["PARAMID"] = self.data["PARAMID"]
            c.computeId()
            i = i + 1

        self.parameterToID={}
        # PARAM ID is starting at 0
        paramId=0
        if self.parameters:
           for (paramKind,pID,pPath) in self.parameters:
              self.parameterToID[pID]=paramId
              paramId = paramId + 1

    def reident(self,current,d=2):
        """ Recompute identation lebel
        """
        self.ident=current
        for c in self._children:
            c.reident(current+d)
        
    def findIdentParent(self,newIdent):
        """ Find parent of this node based on the new identation level

        Find the node which is the parent of this node with indentation level
        newIdent.

        Args:
          newIdent (int) : identation of a new node read in the descriptino file

        """
        if self.ident < newIdent:
            return(self)
        else:
            return(self.parent.findIdentParent(newIdent))

    
    def __getitem__(self, i):
        return(self._children[i])

    def __iter__(self):
      self._currentPos = 0
      return(self)

    def __next__(self):
      oldPos = self._currentPos
      self._currentPos = self._currentPos + 1
      if (oldPos >= len(self._children)):
        raise StopIteration
      return(self._children[oldPos])

    def addPattern(self,theId,thePath):
        """ Add a new pattern

        Args:
          theId (int) : pattern ID
          thePath (str) : pattern path

        """
        self.patterns.append((theId,thePath))
        #print(thePath)
        #print(self.patterns)

    def addParam(self,paramKind,theId,theData):
        """ Add a new parameter file

        Args:
          paramKind (int) : parameter kind (path or generator)
          theId (int) : parameter ID
          thePath (str or list) : parameter path or generator data

        """
        self.parameters.append((paramKind,theId,theData))
        #print(thePath)
        #print(self.patterns)

    def addOutput(self,theId,thePath):
        """ Add a new output

        Args:
          theId (int) : output ID
          thePath (str) : output path

        """
        self.outputs.append((theId,thePath))

    def parse(self,filePath):
       """ Parser the test description file

        Args:
          filePath (str) : Path to the description file
       """
       root = None 
       current = None 
       with open(filePath,"r") as ins:
          for line in ins:
              # Compute identation level
              identLevel = 0
              if re.match(r'^([ \t]+)[^ \t].*$',line):
                 leftSpaces=re.sub(r'^([ \t]+)[^ \t].*$',r'\1',line.rstrip())
                 #print("-%s-" % leftSpaces)
                 identLevel = len(leftSpaces)
              # Remove comments
              line = re.sub(r'^(.*)//.*$',r'\1',line).rstrip()
              # If line is not just a comment
              if line:
                 regPat = r'^[ \t]+Pattern[ \t]+([a-zA-Z0-9_]+)[ \t]*:[ \t]*(.+)$'
                 regOutput = r'^[ \t]+Output[ \t]+([a-zA-Z0-9_]+)[ \t]*:[ \t]*(.+)$'
                 # If a pattern line is detected, we record it
                 if re.match(regPat,line):
                    m = re.match(regPat,line) 
                    patternID = m.group(1).strip()
                    patternPath = m.group(2).strip()
                    #print(patternID)
                    #print(patternPath)
                    if identLevel > current.ident:
                        current.addPattern(patternID,patternPath)
                 # If an output line is detected, we record it
                 elif re.match(regOutput,line):
                    m = re.match(regOutput,line) 
                    outputID = m.group(1).strip()
                    outputPath = m.group(2).strip()
                    #print(patternID)
                    #print(patternPath)
                    if identLevel > current.ident:
                        current.addOutput(outputID,outputPath)
                 else:
                    #if current is None:
                    #   print("  -> %d" % (identLevel))
                    #else:
                    #   print("%d -> %d" % (current.ident,identLevel))
                    # Separate line into components
                    data = line.split(':')
                    # Remove empty strings
                    data = [item for item in data if item]
                    # If it is the first node we detect, it is the root node
                    if root is None:
                       root = TreeElem(identLevel)
                       root.setData(data)
                       current = root 
                    else:
                        # We analyse and set the data
                        newItem = TreeElem(identLevel)
                        newItem.setData(data)
                        # New identation then it is a group (or suite)
                        if identLevel > current.ident:
                           #print( ">")
                           current.addGroup(newItem)
                           current = newItem 
                        # Same identation, we add to parent
                        elif identLevel == current.ident:
                           #print( "==")
                           current.parent.addGroup(newItem)
                        else:
                           #print("<")
                           #print("--")
                           #print(identLevel)
                           # Smaller identation we need to find the parent where to
                           # attach this node.
                           current = current.findIdentParent(identLevel)
                           current.addGroup(newItem)
                           current = newItem
   
                    #print(identLevel)
                    #print(data)  

       # Identify suites, groups and tests
       # Above we are just adding TreeElement but we don't yet know their
       # kind. So we classify them to now if we have group, suite or test
       root.classify()
       # We compute ID of all nodes.
       root.computeId()
       return(root)

