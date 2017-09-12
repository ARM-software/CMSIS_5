#! python

from buildcmd import BuildCmd

class DirCmd(BuildCmd):

  def __init__(self):
    BuildCmd.__init__(self)
    
  def getCommand(self):
    return "dir"

    