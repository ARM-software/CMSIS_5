#! python

from buildcmd import BuildCmd
from datetime import datetime
import mmap

class IarCmd(BuildCmd):

  def __init__(self, project, config):
    BuildCmd.__init__(self)
    self._project = project
    self._config = config
    self._log = "iar_{0}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))
    
  def getCommand(self):
    return "iarbuild.exe"
    
  def getArguments(self):
    return [ self._project, "-build", self._config ]
  
  def isSuccess(self):
    return self._result <= 1

  def getLog(self):
    try:
      return open(self._log, "r")
    except:
      return None
