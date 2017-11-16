#! python

from buildcmd import BuildCmd
from string import maketrans
from datetime import datetime
import mmap

class Uv4Cmd(BuildCmd):

  def __init__(self, project, config):
    BuildCmd.__init__(self)
    self._project = project
    self._config = config
    self._log = "UV4_{0}_{1}.log".format(self._config.translate(maketrans(" ", "_"), "()[],"), datetime.now().strftime("%Y%m%d%H%M%S"))
    
  def getCommand(self):
    return "UV4.exe"
    
  def getArguments(self):
    return [ "-t", self._config, "-cr", self._project, "-j0", "-o", self._log ]
  
  def isSuccess(self):
    return self._result <= 1

  def getLog(self):
    try:
      return open(self._log, "r")
    except:
      return None
