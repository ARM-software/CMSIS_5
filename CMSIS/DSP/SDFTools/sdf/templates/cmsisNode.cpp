
{% for ptr in ptrs %}
         {{theType}}* {{ptr}};
{% endfor %}
{% for ptr in inputs %}
         {{ptr[0]}}=this->{{ptr[2]}}();
{% endfor %}
{% for ptr in outputs %}
         {{ptr[0]}}=this->{{ptr[2]}}();
{% endfor %}
         {{func}}({{args}},{{nb}});
         return(0);
       