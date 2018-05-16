#! python

import os
import shutil
import sys

from enum import Enum

from buildutils.builder import Axis, Step, BuildStep, Builder, Filter, Compiler
 
class Target(Enum):
  CM0    = ( 'CM0'   , 'CM0_LE'            )
  CM3    = ( 'CM3'   , 'CM3_LE'            )
  CM4F   = ( 'CM4F'  , 'CM4F_LE'           )
  V8MB   = ( 'V8MB'  , 'ARMv8MBL_LE'       )
  V8MBN  = ( 'V8MBN' , 'ARMv8MBL_NS_LE'    )
  V8MM   = ( 'V8MM'  , 'ARMv8MML_LE'       )
  V8MMF  = ( 'V8MMF' , 'ARMv8MML_SP_LE'    )
  V8MMFN = ( 'V8MMFN', 'ARMv8MML_SP_NS_LE' )
  V8MMN  = ( 'V8MMN' , 'ARMv8MML_NS_LE'    )
  
  def __str__(self):
    return self.value[0]
    
PROJECTS = {
  Compiler.AC6: [ "ARM/MDK/RTX_CM.uvprojx" ],
  Compiler.GCC: [ "GCC/MDK/RTX_CM.uvprojx" ],
  Compiler.IAR: [ "IAR/IDE/RTX_CM.ewp" ]
}

TARGETS = {
  Compiler.AC6: 1,
  Compiler.GCC: 1,
  Compiler.IAR: 0
}

def target(step, config):
  return config['target'].value[TARGETS[config['compiler']]]

def create():
  compilerAxis = Axis("compiler", abbrev="c", values=[ Compiler.AC6, Compiler.GCC, Compiler.IAR ], desc="Compiler(s) to be considered.")
  targetAxis = Axis("target", abbrev="t", values=Target, desc="Target(s) to be considered.")
  
  buildStep = BuildStep("build", abbrev="b", desc="Build the selected configurations.")
  buildStep.projects = lambda step, config: PROJECTS[config['compiler']]
  buildStep.target = target
  
  filterIAR = Filter().addAxis(compilerAxis, Compiler.IAR).addAxis(targetAxis, "V8M*")

  builder = Builder()
  builder.addAxis([ compilerAxis, targetAxis ])
  builder.addStep([ buildStep  ])
  builder.addFilter([ filterIAR ])

  return builder
  