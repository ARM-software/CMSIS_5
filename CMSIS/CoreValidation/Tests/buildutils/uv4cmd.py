#! python

import os
import shutil
from buildcmd import BuildCmd
from datetime import datetime

class Uv4Cmd(BuildCmd):

  def __init__(self, project, config, env=os.environ):
    BuildCmd.__init__(self, env=env)
    self._project = project
    self._config = config
    cwd = os.getcwd()
    trans = { " " : "_", "(" : None, ")" : None, "[" : None, "]" : None, "," : None }
    self._log = cwd + "\\UV4_{0}_{1}.log".format(self._config.translate(trans), datetime.now().strftime("%Y%m%d%H%M%S"))

  def getCommand(self):
    return "UV4.exe"
    
  def getArguments(self):
    return [ "-t", self._config, "-r", self._project, "-j0", "-o", self._log ]
  
  def isSuccess(self):
    return self._result <= 1

  def getLog(self):
    try:
      return open(self._log, "r")
    except:
      return None
