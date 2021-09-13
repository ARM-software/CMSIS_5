#include "{{filename}}.h"

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

{% for config in configs["dct"] %}
{{iff16("dct",config)}}
const {{configs["dct"][config]["ctype"]}} mfcc_dct_coefs_{{config}}[NB_MFCC_DCT_COEFS_{{config.upper()}}]={{configs["dct"][config]["dctMatrix"]}};
{{endiff16("dct",config)}}

{% endfor %}

{% for config in configs["window"] %}
{{iff16("window",config)}}
const {{configs["window"][config]["ctype"]}} mfcc_window_coefs_{{config}}[NB_MFCC_WIN_COEFS_{{config.upper()}}]={{configs["window"][config]["winSamples"]}};
{{endiff16("window",config)}}

{% endfor %}

{% for config in configs["melfilter"] %}
const uint32_t mfcc_filter_pos_{{config}}[NB_MFCC_NB_FILTER_{{config.upper()}}]={{configs["melfilter"][config]["filtPosArray"]}};
const uint32_t mfcc_filter_len_{{config}}[NB_MFCC_NB_FILTER_{{config.upper()}}]={{configs["melfilter"][config]["filtLenArray"]}};

{% endfor %}


{% for config in configs["melfilter"] %}
{{iff16("melfilter",config)}}
const {{configs["melfilter"][config]["ctype"]}} mfcc_filter_coefs_{{config}}[NB_MFCC_FILTER_COEFS_{{config.upper()}}]={{configs["melfilter"][config]["filters"]}};
{{endiff16("melfilter",config)}}

{% endfor %}