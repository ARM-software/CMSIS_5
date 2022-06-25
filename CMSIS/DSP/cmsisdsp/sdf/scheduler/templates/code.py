#
# Generated with CMSIS-DSP SDF Scripts.
# The generated code is not covered by CMSIS-DSP license.
# 
# The support classes and code is covered by CMSIS-DSP license.
#

import sys


import numpy as np
import cmsisdsp as dsp
from cmsisdsp.sdf.nodes.simu import *
from appnodes import * 
from custom import *

{% macro optionalargs() -%}
{% if config.pyOptionalArgs %}{{config.pyOptionalArgs}}{% endif %}
{% endmacro -%}

{% if config.dumpFIFO %}
DEBUGSCHED=True
{% else %}
DEBUGSCHED=False
{% endif %}

# 
# FIFO buffers
# 


{% for id in range(nbFifos) %}
FIFOSIZE{{id}}={{fifos[id].length}}
{{config.prefix}}buf{{id}}=np.zeros(FIFOSIZE{{id}},dtype=np.{{fifos[id].theType.nptype}})

{% endfor %}

def {{config.schedName}}({{optionalargs()}}):
    sdfError=0
    nbSchedule=0
{% if config.debug %}
    debugCounter={{config.debugLimit}}
{% endif %}

    #
    #  Create FIFOs objects
    #
{% for id in range(nbFifos) %}
{% if fifos[id].hasDelay %}
    fifo{{id}}=FIFO(FIFOSIZE{{id}},{{config.prefix}}buf{{id}},delay={{fifos[id].delay}})
{% else %}
    fifo{{id}}=FIFO(FIFOSIZE{{id}},{{config.prefix}}buf{{id}})
{% endif %}
{% endfor %}

    # 
    #  Create node objects
    #
{% for node in nodes %}
{% if node.hasState %}
    {{node.nodeName}} = {{node.typeName}}({{node.pythonIoTemplate()}},{{node.args}})
{% endif %}
{% endfor %}

{% if config.debug %}
    while((sdfError==0) and (debugCounter > 0)):
{% else %}
    while(sdfError==0):
{% endif %}
       nbSchedule = nbSchedule + 1

{% for s in schedule %}
       {{nodes[s].cRun(False)}}
       if sdfError < 0:
          break
{% if config.dumpFIFO %}
{% for fifoID in sched.outputFIFOs(nodes[s]) %}
       print("{{nodes[s].nodeName}}:{{fifoID[1]}}")
       fifo{{fifoID[0]}}.dump()
{% endfor %}
{% endif %}
{% endfor %}

{% if config.debug %}
       debugCounter = debugCounter - 1 
{% endif %}
    return(nbSchedule,sdfError)