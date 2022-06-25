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

/* List of nodes */
static NodeBase *nodeArray[{{nbNodes}}]={0};

/*

Description of the scheduling. It is a list of nodes to call.
The values are indexes in the previous array.

*/
static unsigned int schedule[{{schedLen}}]=
{ 
{{schedDescription}}
};

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

/**************
 
 Classes created for pure function calls (like some CMSIS-DSP functions)

***************/

{% for p in pureNodes %}
{% set node = pureNodes[p] %}
template<{{node.templateParameters}}> class Func{{node.pureClassID}};

template<{{node.specializedTemplateParameters}}>
class Func{{node.pureClassID}}<{{node.templateArguments}}>: public {{node.nodeKind}}<{{node.templateParametersForGeneric}}>
{
public:
    Func{{node.pureClassID}}({{node.datatypeForConstructor}}):
    {{node.nodeKind}}<{{node.templateParametersForGeneric}}>({{node.genericConstructorArgs}}){};

   int run(){
     {{node.codeArrayRun()}}
   };

};
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
    nodeArray[{{node.codeID}}]=(NodeBase*)&{{node.nodeName}};
{% else %}

    Func{{node.pureClassID}}<{{node.constructorTypes}}> func{{node.pureNodeID}}({{node.constructorArguments}});
    nodeArray[{{node.codeID}}]=(NodeBase*)&func{{node.pureNodeID}};
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
        for(unsigned long id=0 ; id < {{schedLen}}; id++)
        {
            unsigned int nodeId = schedule[id];
            sdfError = nodeArray[nodeId]->run();
            CHECKERROR;
        }
{% if config.debug %}
       debugCounter--;
{% endif %}
       nbSchedule++;
    }

    *error=sdfError;
    return(nbSchedule);
}