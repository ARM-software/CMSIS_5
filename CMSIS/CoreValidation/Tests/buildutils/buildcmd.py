#! python

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
    
  def getOutput(self):
    return self._output

  def getLog(self):
    return None
    
  def isSuccess(self):
    return self._output == 0

  def run(self):  
    cmd = [ self.getCommand() ] + self.getArguments()
    print("Running: " + ' '.join(cmd))
    try:
      self._result = call(cmd, stdout = self._output)
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
    