

class Hierarchy:
    def __init__(self,name,subsections=None):
        self._parent = None
        self._name=name 
        self._sections = []
        if subsections is not None:
           for s in subsections:
             self.addSection(s)

    @property
    def parent(self):
        return(self._parent)

    @property
    def sections(self):
        return(self._sections)

    def addSection(self,section):
        self._sections.append(section)

    @property
    def hasChildren(self):
        return(len(self._sections)>0)

    @property
    def name(self):
        return(self._name)



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

class Section(Hierarchy):
  def __init__(self,name):
      super(Section, self).__init__(name)
      self._content = []

  def addContent(self,content):
      self._content.append(content)

  @property
  def hasContent(self):
      return(len(self._content) > 0 or any([x.hasContent for x in self.sections]))


  def accept(self, visitor):
      if self.hasContent:
         visitor.visitSection(self)
         for element in self.sections:
             element.accept(visitor) 
         for element in self._content:
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

class Text:
    def __init__(self,text):
       self._text = text

    @property
    def text(self):
       return(self._text)

    def accept(self, visitor):
      visitor.visitText(self)

class BarChart:
    def __init__(self,data):
       self._data = data

    @property
    def data(self):
       return(self._data)

    def accept(self, visitor):
      visitor.visitBarChart(self)

class History:
    def __init__(self,data,runid):
       self._data = data
       minId = runid-9 
       if minId < 0:
          minId = 0
       self._runids = list(range(minId,runid+1))

    @property
    def data(self):
       return(self._data)

    @property
    def runids(self):
       return(self._runids)

    def accept(self, visitor):
      visitor.visitHistory(self)