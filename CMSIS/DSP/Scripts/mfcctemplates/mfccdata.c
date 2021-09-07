#include "{{filename}}.h"

{% for config in configs["dct"] %}
const {{configs["dct"][config]["ctype"]}} mfcc_dct_coefs_{{config}}[NB_MFCC_DCT_COEFS_{{config.upper()}}]={{configs["dct"][config]["dctMatrix"]}};

{% endfor %}

{% for config in configs["window"] %}
const {{configs["window"][config]["ctype"]}} mfcc_window_coefs_{{config}}[NB_MFCC_WIN_COEFS_{{config.upper()}}]={{configs["window"][config]["winSamples"]}};

{% endfor %}

{% for config in configs["melfilter"] %}
const uint32_t mfcc_filter_pos_{{config}}[NB_MFCC_NB_FILTER_{{config.upper()}}]={{configs["melfilter"][config]["filtPosArray"]}};
const uint32_t mfcc_filter_len_{{config}}[NB_MFCC_NB_FILTER_{{config.upper()}}]={{configs["melfilter"][config]["filtLenArray"]}};

{% endfor %}


{% for config in configs["melfilter"] %}
const {{configs["melfilter"][config]["ctype"]}} mfcc_filter_coefs_{{config}}[NB_MFCC_FILTER_COEFS_{{config.upper()}}]={{configs["melfilter"][config]["filters"]}};

{% endfor %}