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
    if self._args.has_key('limit'):  args += [ "--cyclelimit", self._args['limit'] ] 
    if self._args.has_key('config'): args += [ "-f", self._args['config'] ]
    if self._args.has_key('target'):
      args += [ "-a", "{0}={1}".format(self._args['target'], self._app ) ]
    else:
      args += [ self._app ]
    return args
    