#! python

import shutil
from StringIO import StringIO
from xml.etree import ElementTree

class TestResult:

  def _extractXml(self, log, xml):
    dump = False
    log.seek(0)
    for line in log:
      if dump:
        xml.write(line)
        if line.strip() == '</report>':
          dump = False
      else:
        if line.strip() == '<?xml version="1.0"?>':
          dump = True
          xml.write(line)

  def __init__(self, log):
    self._xml = StringIO()
    self._extractXml(log, self._xml)
    self._xml.seek(0)
  
    report = ElementTree.parse(self._xml).getroot()
    summary = report[0].findall('summary')[0]
    self._tests = summary.find('tcnt').text
    self._executed = summary.find('exec').text
    self._passed = summary.find('pass').text
    self._failed = summary.find('fail').text
          
  def saveXml(self, filename):
    with open(filename, "w") as file:
      self._xml.seek(0)
      shutil.copyfileobj(self._xml, file)
    
  def getSummary(self):
    return self._tests, self._executed, self._passed, self._failed