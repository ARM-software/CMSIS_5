#! python

import os
import shutil
from buildcmd import BuildCmd
from datetime import datetime
import mmap

class RteCmd(BuildCmd):

  def __init__(self, project, config, env=os.environ, subcmd = "build"):
    BuildCmd.__init__(self, env=env)
    self._project = project
    self._config = config
    self._subcmd = subcmd
    self._rtebuild = shutil.which("rtebuild.py")
    if not self._rtebuild:
      raise FileNotFoundError("rtebuild.py")
      
  def getCommand(self):
    return "python.exe"
    
  def getArguments(self):
    return [ os.path.normpath(self._rtebuild), "-c", os.path.abspath(self._project), "-t", self._config, self._subcmd ]

  def needsShell(self):
    return True
    
  def isSuccess(self):
    return self._result == 0

  def getLog(self):
      return None
