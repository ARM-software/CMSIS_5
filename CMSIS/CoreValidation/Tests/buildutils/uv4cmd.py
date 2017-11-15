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

  def getCommand(self):
    return "uVision.com"
    
  def getArguments(self):
    return [ "-t", self._config, "-r", self._project, "-j0" ]
  
  def isSuccess(self):
    return self._result <= 1
