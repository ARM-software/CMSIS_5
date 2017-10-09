#! python

import os
import shutil
from subprocess import call, Popen
from tempfile import TemporaryFile

class BuildCmd:
  def __init__(self):
    self._result = -1
    self._output = TemporaryFile(mode="r+")
  
  def getCommand(self):
    raise NotImplementedError
    
  def getArguments(self):
    return []
    
  def needsShell(self):
    return False
    
  def getOutput(self):
    return self._output

  def getLog(self):
    return None
    
  def isSuccess(self):
    return self._output == 0

  def run(self):  
    cmd = [ os.path.normpath(shutil.which(self.getCommand())) ] + self.getArguments()
    print("Running: " + ' '.join(cmd))
    try:
      with Popen(cmd, stdout = self._output, stderr = self._output, shell=self.needsShell()) as proc:
        self._result = proc.wait()
    except:
      print("Fatal error!")
    self._output.seek(0)
    print(self._output.read())
    
    logfile = self.getLog()
    if logfile != None:
      print(logfile.read())
      
    print("Command returned: {0}".format(self._result))
      
    return self._result
    
  def skip(self):
    self._result = 0
    