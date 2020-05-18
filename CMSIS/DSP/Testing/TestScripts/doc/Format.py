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
     self._output.write("%s %s\n" % (header,section.name))

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
  width: 290px;
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
  margin-left:290px;

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
  margin-left:1em;
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

.firstcore {
  border-left-color: black;
  border-left-style: solid;
  border-left-width: 1px;
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
      self._output.write("<table>\n")
      self._output.write("<thead>\n")
      self._output.write("<tr>\n")
      firstCore = False
      for col in table.params:
        firstCore = True
        self._output.write("<th class=\"param\">")
        self._output.write(str(col))
        self._output.write("</th>\n")
      for col in table.cores:
        if firstCore:
           self._output.write("<th class=\"firstcore\">")
        else:
           self._output.write("<th class=\"core\">")
        self._output.write(str(col))
        self._output.write("</th>\n")
        firstCore = False
      self._output.write("</tr>\n")
      self._output.write("</thead>\n")
      
      nbParams = len(table.params)
      for row in table.rows:
        self._output.write("<tr>\n")
        i = 0
        for elem in row:
            if i < nbParams:
               self._output.write("<td class=\"param\">")
               self._output.write(str(elem))
               self._output.write("</td>\n")
            elif i == nbParams and nbParams != 0:
               self._output.write("<td class=\"firstcore\">")
               self._output.write(str(elem))
               self._output.write("</td>\n")
            else:
               self._output.write("<td class=\"core\">")
               self._output.write(str(elem))
               self._output.write("</td>\n")
            i = i + 1
        self._output.write("</tr>\n")
      self._output.write("</table>\n")


  def visitSection(self,section):
     self._id = self._id + 1 
     self._sectionID = self._sectionID + 1
     self._output.write("<h%d id=\"section%d\">%s</h%d>\n" % (self._id,self._sectionID,section.name,self._id))

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


