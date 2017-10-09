#! python

import os
import shutil
from buildcmd import BuildCmd
from datetime import datetime
import mmap

class RteCmd(BuildCmd):

  def __init__(self, project, config):
    BuildCmd.__init__(self)
    self._project = project
    self._config = config

  def getCommand(self):
    return "python.exe"
    
  def getArguments(self):
    return [ os.path.normpath(shutil.which("rtebuild.py")), "-c", self._config, os.path.abspath(self._project) ]

  def needsShell(self):
    return True
    
  def isSuccess(self):
    return self._result == 0

  def getLog(self):
      return None
