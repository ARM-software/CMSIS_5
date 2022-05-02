#ifndef _MFCC_DATA_H_
#define _MFCC_DATA_H_ 

{% macro iff16(s,c) -%}
{%- if configs[s][c]["hasF16"] %}
#if defined(ARM_FLOAT16_SUPPORTED)
{%- endif %}
{% endmacro -%}

{% macro endiff16(s,c) -%}
{%- if configs[s][c]["hasF16"] %}
#endif /*defined(ARM_FLOAT16_SUPPORTED) */
{%- endif %}
{% endmacro -%}

#include "arm_math_types.h"

{% if configs["hasF16"] %}
#include "arm_math_types_f16.h"
{% endif %}

#ifdef   __cplusplus
extern "C"
{
#endif


/*****

 DCT COEFFICIENTS FOR THE MFCC

*****/

{% for config in configs["dct"] %}
{{iff16("dct",config)}}
#define NB_MFCC_DCT_COEFS_{{config.upper()}} {{configs["dct"][config]['dctMatrixLength']}}
extern const {{configs["dct"][config]["ctype"]}} mfcc_dct_coefs_{{config}}[NB_MFCC_DCT_COEFS_{{config.upper()}}];
{{endiff16("dct",config)}}

{% endfor %}

/*****

 WINDOW COEFFICIENTS

*****/

{% for config in configs["window"] %}
{{iff16("window",config)}}
#define NB_MFCC_WIN_COEFS_{{config.upper()}} {{configs["window"][config]['fftlength']}}
extern const {{configs["window"][config]["ctype"]}} mfcc_window_coefs_{{config}}[NB_MFCC_WIN_COEFS_{{config.upper()}}];
{{endiff16("window",config)}}

{% endfor %}

/*****

 MEL FILTER COEFFICIENTS FOR THE MFCC

*****/

{% for config in configs["melfilter"] %}
#define NB_MFCC_NB_FILTER_{{config.upper()}} {{configs["melfilter"][config]['melFilters']}}
extern const uint32_t mfcc_filter_pos_{{config}}[NB_MFCC_NB_FILTER_{{config.upper()}}];
extern const uint32_t mfcc_filter_len_{{config}}[NB_MFCC_NB_FILTER_{{config.upper()}}];

{% endfor %}



{% for config in configs["melfilter"] %}
{{iff16("melfilter",config)}}
#define NB_MFCC_FILTER_COEFS_{{config.upper()}} {{configs["melfilter"][config]['totalLen']}}
extern const {{configs["melfilter"][config]["ctype"]}} mfcc_filter_coefs_{{config}}[NB_MFCC_FILTER_COEFS_{{config.upper()}}];
{{endiff16("melfilter",config)}}

{% endfor %}

#ifdef   __cplusplus
}
#endif

#endif

