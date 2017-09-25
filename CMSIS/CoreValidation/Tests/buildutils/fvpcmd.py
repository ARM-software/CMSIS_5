#! python

from buildcmd import BuildCmd

class FvpCmd(BuildCmd):

  def __init__(self, model, app, **args):
    BuildCmd.__init__(self)
    self._model = model
    self._app = app
    self._args = args
    
  def getCommand(self):
    return self._model
    
  def getArguments(self):
    args = []
    if 'limit' in self._args:  args += [ "--cyclelimit", self._args['limit'] ] 
    if 'config' in self._args: args += [ "-f", self._args['config'] ]
    if 'target' in self._args:
      for a in self._app:
        args += [ "-a", "{0}={1}".format(self._args['target'], a ) ]
    else:
      args += [ self._app[0] ]
    return args
    