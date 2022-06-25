
{% for ptr in inputs %}
       {{ptr[0]}}={{ptr[1]}}.getReadBuffer({{nb}})
{% endfor %}
{% for ptr in outputs %}
       {{ptr[0]}}={{ptr[1]}}.getWriteBuffer({{nb}})
{% endfor %}
       {{outArgs}}[:]=dsp.{{func}}({{inArgs}})
       sdfError = 0
       