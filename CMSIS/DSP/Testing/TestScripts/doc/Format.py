import math
from datetime import date

NORMALFORMAT=0
BYCFORMAT=1
BYDFORMAT=2

def joinit(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

# To format, in HTML, the cores in the right order.
# First we order the categories
# Then we order the cores in each category
# The final ORDEREDCORES is what is used
# to order tjhe values
# Since some cores may be missing, each atble display
# is computing a rstricted ordered core list with only the available cores.
CORTEXCATEGORIES=["Cortex-M","Cortex-R","Cortex-A"]
CORECATEGORIES={"Cortex-M":["m0","m4", "m7", "m33" , "m55 scalar", "m55 mve","m55 autovec"],
"Cortex-R":["r8","r52"],
"Cortex-A":["a32"]
}
ORDEREDCORES=[]
for cat in CORTEXCATEGORIES:
  cores=[] 
  if cat in CORECATEGORIES:
     for core in CORECATEGORIES[cat]:
       cores.append(core)
  else:
    print("Error core %s not found" % cat)
    quit()
  ORDEREDCORES += cores

ORDEREDTYPES=["q7","q15","q31","u32","f16","f32","f64"]

class Markdown:
  def __init__(self,output):
    self._id=0
    self._output = output

  def visitBarChart(self,data):
      pass

  def visitHistory(self,data):
      pass

  def visitText(self,text):
      self._output.write(text)

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
      if document.runidHeader:
         self._output.write("Document generated for run ids : %s\n" % document.runidHeader)

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
  background-color: #E5ECEB;
  font-family: "Lato";
  font-style: normal; font-variant: normal;
  color: #002B49;
  line-height: 1.5em;
}

body {
  margin: auto;
  margin-top:0px;
  margin-left:290px;

}

.NA {
  color: #999999;
}

.testname {
  color: #0091BD;
  font-size: 1.125em;
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
  background-color: #979ea3;
}
tr:nth-child(even) {
  background: #d7dadc;
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
  border-left-width: 2px;
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

barscript="""    <script src="https://d3js.org/d3.v5.js"></script>



<script type="text/javascript">

histwidth=400;
histheight=200;
histmargin={left:40,right:100,bottom:40,top:10};

function legend(color,svg)
{
    const g = svg
      .attr("transform", `translate(${histwidth},0)`)
      .attr("text-anchor", "end")
      .attr("font-family", "sans-serif")
      .attr("font-size", 9)
    .selectAll("g")
    .data(color.domain().slice().reverse())
    .join("g")
      .attr("transform", (d, i) => `translate(0,${i * 20})`);

  g.append("rect")
      .attr("x", -19)
      .attr("width", 19)
      .attr("height", 19)
      .attr("fill", color);

  g.append("text")
      .attr("x", -24)
      .attr("y", 9.5)
      .attr("dy", "0.35em")
      .text(d => d);

}
  
function myhist(data,theid)
{
    var x,y,xAxis,yAxis,svg,color;

    

color = d3.scaleOrdinal()
    .domain(data.series.map(d => d['name']))
    .range(["#FF6B00",
"#FFC700",
"#95D600",
"#00C1DE",
"#0091BD",
"#002B49",
"#333E48",
"#7D868C",
"#E5ECEB"]);
    
    svg = d3.select(theid).insert("svg")
      .attr("viewBox", [0, 0, histwidth, histheight]);


sx = d3.scaleLinear()
    .domain(d3.extent(data.dates))
    .range([histmargin.left,histwidth - histmargin.right]);

sy = d3.scaleLinear()
    .domain([0, d3.max(data.series, d => d3.max(d.values,q => q[1]))]).nice()
    .range([histheight - histmargin.bottom, histmargin.top]);

xAxis = g => g
    .attr("transform", `translate(0,${histheight - histmargin.bottom})`)
    .call(d3.axisBottom(sx).tickValues(data.dates).ticks(histwidth / 80,"d").
        tickSizeOuter(0));

svg.append("text")
    .attr("class", "x label")
    .attr("text-anchor", "end")
    .attr("x", histwidth/2.0)
    .attr("y", histheight - 6)
    .text("RUN ID");

yAxis = g => g
    .attr("transform", `translate(${histmargin.left},0)`)
    .call(d3.axisLeft(sy))
    .call(g => g.select(".domain").remove());



line = d3.line()
    .x(d => sx(data.dates[d[0]])
    )
    .y(d => sy(d[1]));

svg.append("g")
      .call(xAxis);

  svg.append("g")
      .call(yAxis);

  const path = svg.append("g")
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
    .selectAll("path")
    .data(data.series)
    .join("path")
      .style("mix-blend-mode", "multiply")
      .attr("stroke", d => color(d.name))
      .attr("d", d => line(d.values));

// Legend

 

svg.append("g")
      .call(d => legend(color,d)); 
  //svg.call(hover, path);


}

function mybar(data,theid)
{
    var width,height,margin,x,y,xAxis,yAxis,svg,color;

    width=400;
    height=100;
    margin={left:40,right:10,bottom:40,top:10};

    
    svg = d3.select(theid).insert("svg")
      .attr("viewBox", [0, 0, width, height]);;

x = d3.scaleBand()
    .domain(d3.range(data.length))
    .range([margin.left, width - margin.right])
    .padding(0.1);

y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)]).nice()
    .range([height - margin.bottom, margin.top]);

xAxis = g => g
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).tickFormat(i => data[i].name).tickSizeOuter(0));

yAxis = g => g
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(4, data.format))
    .call(g => g.select(".domain").remove())
    .call(g => g.append("text")
        .attr("x", -margin.left)
        .attr("y", 10)
        .attr("fill", "currentColor")
        .attr("text-anchor", "start")
        .text(data.y));

color = "steelblue"
   
  svg.append("g")
      .attr("fill", color)
    .selectAll("rect")
    .data(data)
    .join("rect")
      .attr("x", (d, i) => x(i))
      .attr("y", d => y(d.value))
      .attr("height", d => y(0) - y(d.value))
      .attr("width", x.bandwidth());

  svg.append("g")
      .call(xAxis);

  svg.append("g")
      .call(yAxis);

}
</script>"""


class HTMLToc:
  def __init__(self,output):
    self._id=0
    self._sectionID = 0
    self._output = output

  

  def visitTable(self,table):
      pass

  def visitBarChart(self,data):
      pass

  def visitHistory(self,data):
      pass

  def visitText(self,text):
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

def permutation(ordered,unordered,mode):
    result=[] 
    restricted=[] 
    order = ORDEREDCORES 
    if mode == BYDFORMAT:
      order = ORDEREDTYPES
    for c in order:
      if c in unordered: 
         restricted.append(c)

    for c in unordered:
      result.append(restricted.index(c))

    return(result,restricted)

def reorder(p,v):
    result=[0 for x in v]
    for val,i in zip(v,p):
        result[i]=val 

    return(result)

class HTML:
  def __init__(self,output,regMode,ratio,reorder):
    self._id=0
    self._sectionID = 0
    self._barID = 0
    self._histID = 0
    self._output = output
    self._regMode = regMode
    self._reorder = reorder
    self._ratioMode = ratio and regMode

  def visitBarChart(self,bar):
      data=bar.data
      datastr = "".join(joinit(["{name:'%s',value:%s}" % x for x in data],","))
      #print(datastr)
      self._output.write("<p id=\"g%d\"></p>\n" % self._barID)
      self._output.write("""<script type="text/javascript">
thedata%d=[%s];
mybar(thedata%d,"#g%d");
</script>""" % (self._barID,datastr,self._barID,self._barID))

      self._barID = self._barID + 1

  def _getIndex(self,runids,data):
    return([[runids.index(x[0]),x[1]] for x in data])

  def visitHistory(self,hist):
      data=hist.data
      runidstr = "".join(joinit([str(x) for x in hist.runids],","))
      serieelems=[]
      for core in data:
        serieelems.append("{name: '%s',values: %s}" % (core,self._getIndex(hist.runids,data[core])))

      seriestr = "".join(joinit(serieelems,","))
      datastr="""{
series: [%s],
 dates: [%s]
};""" %(seriestr,runidstr);
      #print(datastr)
      self._output.write("<p id=\"hi%d\"></p>\n" % self._histID)
      self._output.write("""<script type="text/javascript">
thehdata%d=%s
myhist(thehdata%d,"#hi%d");
</script>""" % (self._histID,datastr,self._histID,self._histID))

      self._histID = self._histID + 1

  def visitText(self,text):
      self._output.write("<p>\n")
      self._output.write(text.text)
      self._output.write("</p>\n")

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

      if self._reorder == NORMALFORMAT:
         perm,restricted=permutation(ORDEREDCORES,table.cores,self._reorder)
      elif self._reorder == BYDFORMAT:
         perm,restricted=permutation(ORDEREDTYPES,table.cores,self._reorder)
      else:
         restricted = table.cores

      for col in restricted:
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

        row=list(row)

        #print(row)

        params=row[0:nbParams]
        values=row[nbParams:]

        if self._reorder == NORMALFORMAT:
          row = params + reorder(perm,values)
        elif self._reorder == BYDFORMAT:
          row = params + reorder(perm,values)
        else:
          row = params + values

        for elem in row:
            txt=str(elem)
            if txt == 'NA':
               txt = "<span class=\"NA\">" + txt + "</span>"
            if i < nbParams:
               self._output.write("<td class=\"param\">")
               self._output.write(txt)
               self._output.write("</td>\n")
            elif i == nbParams and nbParams != 0:
               self._output.write("<td class=\"firstcore\">")
               self._output.write(txt)
               self._output.write("</td>\n")
            else:
               self._output.write("<td class=\"core\">")
               self._output.write(txt)
               self._output.write("</td>\n")
            i = i + 1
        self._output.write("</tr>\n")
      self._output.write("</table>\n")


  def visitSection(self,section):
     self._id = self._id + 1 
     self._sectionID = self._sectionID + 1
     name = section.name 
     if section.isTest:
        name = "<span class=\"testname\">" + name + "</span>"
     self._output.write("<h%d id=\"section%d\">%s</h%d>\n" % (self._id,self._sectionID,name,self._id))

  def leaveSection(self,section):
     self._id = self._id - 1 

  def visitDocument(self,document):
      self._output.write("""<!doctype html>
<html>
<head>
<meta charset='UTF-8'><meta name='viewport' content='width=device-width initial-scale=1'>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lato">
<title>Benchmarks</title>%s</head><body>\n""" % styleSheet)
      if self._regMode and not self._ratioMode:
         self._output.write("<h1>ECPS Benchmark Regressions</h1>\n")
      elif self._ratioMode:
         self._output.write("<h1>ECPS Benchmark Ratios</h1>\n")
      else:
         self._output.write("<h1>ECPS Benchmark Summary</h1>\n")
      
      if document.runidHeader:
         self._output.write("<p>Document generated for run ids : %s</p>\n" % document.runidHeader)
      today = date.today()
      d2 = today.strftime("%B %d, %Y")
      self._output.write("<p>Document generated on  %s</p>\n" % d2)

      self._output.write(barscript)

  def leaveDocument(self,document):
    document.accept(HTMLToc(self._output))

    self._output.write("</body></html>\n")


