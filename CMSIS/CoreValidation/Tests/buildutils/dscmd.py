#! python

import os
from buildcmd import BuildCmd
from datetime import datetime
import mmap

class DsCmd(BuildCmd):

  def __init__(self, project, config):
    BuildCmd.__init__(self)
    self._project = project
    self._config = config
    
    workspace = os.getenv('WORKSPACE')
    if workspace:
      self._workspace = os.path.join(workspace, "eclipse")
    else:
      self._workspace = os.getenv('DSMDK_WORKSPACE')
    if not self._workspace:
      raise RuntimeError("No DS-MDK workspace found, set either DSMDK_WORKSPACE or WORKSPACE in environment!")

  def getCommand(self):
    return "eclipsec.exe"
    
  def getArguments(self):
    return [
        "-nosplash",
        "-application", "org.eclipse.cdt.managedbuilder.core.headlessbuild",
        "-data", self._workspace,
        "-import", os.path.dirname(os.path.abspath(self._project)),
        "-cleanBuild", self._config
      ]
  
  def isSuccess(self):
    return self._result == 0
