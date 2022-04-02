{
{% for ptr in ptrs %}
         {{theType}}* {{ptr}};
{% endfor %}
{% for ptr in inputs %}
         {{ptr[0]}}={{ptr[1]}}.getReadBuffer({{nb}});
{% endfor %}
{% for ptr in outputs %}
         {{ptr[0]}}={{ptr[1]}}.getWriteBuffer({{nb}});
{% endfor %}
         {{func}}({{args}},{{nb}});
         sdfError = 0;
       }