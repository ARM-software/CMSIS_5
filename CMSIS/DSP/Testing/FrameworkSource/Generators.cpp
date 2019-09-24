#include "Generators.h"

void generate1(Testing::param_t *result,Testing::param_t *data,Testing::nbParameterEntries_t &nbEntries)
{
    Testing::param_t *pa,va;

    pa = data;
    int nba = *pa++;
    nbEntries = 0;
    for(int ia=0; ia < nba ; ia ++)
    {
      va = *pa++;
      *result++ = va;
      nbEntries++;
    }
}

void generate2(Testing::param_t *result,Testing::param_t *data,Testing::nbParameterEntries_t &nbEntries)
{
    Testing::param_t *pa,*paStart,va;
    int na;

    Testing::param_t *pb,*pbStart,vb;
    int nb;

    nbEntries = 0;

    na = data[0];
    paStart = &data[1];
    pa = paStart;

    nb = data[na+1];
    pbStart = &data[na+2];
    pb = pbStart;
    
    pa = paStart;
    for(int ia=0; ia < na; ia ++)
    {
       va = *pa++;

       pb = pbStart;
       for(int ib = 0; ib < nb; ib++)
       {
          vb = *pb++;

          *result++ = va;
          *result++ = vb;
          nbEntries++;

       }
    }

}

void generate3(Testing::param_t *result,Testing::param_t *data,Testing::nbParameterEntries_t &nbEntries)
{
    Testing::param_t *pa,*paStart,va;
    int na;

    Testing::param_t *pb,*pbStart,vb;
    int nb;

    Testing::param_t *pc,*pcStart,vc;
    int nc;

    nbEntries=0;

    na = data[0];
    paStart = &data[1];
    pa = paStart;

    nb = data[na+1];
    pbStart = &data[na+2];
    pb = pbStart;

    nc = data[na+1 + nb + 1];
    pcStart = &data[na+1 + nb + 1 + 1];
    pc = pcStart;

    
    pa = paStart;
    for(int ia=0; ia < na; ia ++)
    {
       va = *pa++;
       //printf("%d \n",va);

       pb = pbStart;
       for(int ib = 0; ib < nb; ib++)
       {
          vb = *pb++;
          //printf("%d %d \n",va,vb);
          
          pc = pcStart;
          for(int ic = 0; ic < nc; ic++)
          {
             vc = *pc++;
   
             *result++ = va;
             *result++ = vb;
             *result++ = vc;
             nbEntries++;
             //printf("%d %d %d\n",va,vb,vc);
          }
          
       }
    }
   // printf("OK\n");
}

void generate4(Testing::param_t *result,Testing::param_t *data,Testing::nbParameterEntries_t &nbEntries)
{
    Testing::param_t *pa,*paStart,va;
    int na;

    Testing::param_t *pb,*pbStart,vb;
    int nb;

    Testing::param_t *pc,*pcStart,vc;
    int nc;

    Testing::param_t *pd,*pdStart,vd;
    int nd;

    nbEntries=0;

    na = data[0];
    paStart = &data[1];
    pa = paStart;

    nb = data[na+1];
    pbStart = &data[na+2];
    pb = pbStart;

    nc = data[na+1 + nb + 1];
    pcStart = &data[na+1 + nb + 1 + 1];
    pc = pcStart;

    nd = data[na+1 + nb + 1 + nc + 1 ];
    pdStart = &data[na+1 + nb + 1 + nc + 1 + 1];
    pd = pdStart;
    
    pa = paStart;
    for(int ia=0; ia < na; ia ++)
    {
       va = *pa++;

       pb = pbStart;
       for(int ib = 0; ib < nb; ib++)
       {
          vb = *pb++;
          
          pc = pcStart;
          for(int ic = 0; ic < nc; ic++)
          {
             vc = *pc++;
   
             pd = pdStart;
             for(int id = 0; id < nd; id++)
             {
                vd = *pd++;
                *result++ = va;
                *result++ = vb;
                *result++ = vc;
                *result++ = vd;
                nbEntries++;
             }
          }
          
       }
    }

}
