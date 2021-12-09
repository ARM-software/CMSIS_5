/*

Generated with CMSIS-DSP SDF Scripts.
The generated code is not covered by CMSIS-DSP license.

The support classes and code is covered by CMSIS-DSP license.

*/

{% if config.dumpFIFO %}
#define DEBUGSCHED 1
{% endif %}

#include "arm_math.h"
#include "custom.h"
#include "GenericNodes.h"
#include "AppNodes.h"
#include "scheduler.h"

{% macro optionalargs() -%}
{% if config.cOptionalArgs %},{{config.cOptionalArgs}}{% endif %}
{% endmacro -%}

/***********

FIFO buffers

************/
{% for fifo in fifos %}
#define FIFOSIZE{{fifo.fifoID}} {{fifo.length}}
{% endfor %}

{% for buf in sched._graph._allBuffers %}
#define BUFFERSIZE{{buf._bufferID}} {{buf._length}}
{{buf._theType.ctype}} {{config.prefix}}buf{{buf._bufferID}}[BUFFERSIZE{{buf._bufferID}}]={0};

{% endfor %}

uint32_t {{config.schedName}}(int *error{{optionalargs()}})
{
    int sdfError=0;
    uint32_t nbSchedule=0;
{% if config.debug %}
    int32_t debugCounter={{config.debugLimit}};
{% endif %}

    /*
    Create FIFOs objects
    */
{% for id in range(nbFifos) %}
{% if fifos[id].hasDelay %}
    FIFO<{{fifos[id].theType.ctype}},FIFOSIZE{{id}},{{fifos[id].isArrayAsInt}}> fifo{{id}}({{config.prefix}}buf{{fifos[id].buffer._bufferID}},{{fifos[id].delay}});
{% else %}
    FIFO<{{fifos[id].theType.ctype}},FIFOSIZE{{id}},{{fifos[id].isArrayAsInt}}> fifo{{id}}({{config.prefix}}buf{{fifos[id].buffer._bufferID}});
{% endif %}
{% endfor %}

    /* 
    Create node objects
    */
{% for node in nodes %}
{% if node.hasState %}
    {{node.typeName}}<{{node.ioTemplate()}}> {{node.nodeName}}({{node.args}});
{% endif %}
{% endfor %}

    /* Run several schedule iterations */
{% if config.debug %}
    while((sdfError==0) && (debugCounter > 0))
{% else %}
    while(sdfError==0)
{% endif %}
    {
       /* Run a schedule iteration */
{% for s in schedule %}
       {{nodes[s].cRun()}}
       CHECKERROR;
{% if config.dumpFIFO %}
{% for fifoID in sched.outputFIFOs(nodes[s]) %}
       std::cout << "{{nodes[s].nodeName}}:{{fifoID[1]}}" << std::endl;
       fifo{{fifoID[0]}}.dump();
{% endfor %}
{% endif %}
{% endfor %}

{% if config.debug %}
       debugCounter--;
{% endif %}
       nbSchedule++;
    }
    *error=sdfError;
    return(nbSchedule);
}