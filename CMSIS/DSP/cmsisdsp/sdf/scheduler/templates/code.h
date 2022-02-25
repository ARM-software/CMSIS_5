/*

Generated with CMSIS-DSP SDF Scripts.
The generated code is not covered by CMSIS-DSP license.

The support classes and code is covered by CMSIS-DSP license.

*/

#ifndef _SCHED_H_ 
#define _SCHED_H_

{% macro optionalargs() -%}
{% if config.cOptionalArgs %},{{config.cOptionalArgs}}{% endif %}
{% endmacro -%}

#ifdef   __cplusplus
extern "C"
{
#endif

extern uint32_t {{config.schedName}}(int *error{{optionalargs()}});

#ifdef   __cplusplus
}
#endif

#endif

