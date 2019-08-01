#include "Softmax.h"
#include "Error.h"
#include "arm_nnfunctions.h"
#include "Test.h"

#include <cstdio>

int16_t findMaxIndex(q7_t *vec_in, int length)
{
  int16_t currentIndex=0;
  int16_t i=1;
  q7_t currentMax=vec_in[0];

  while(i<length)
  {
    if (vec_in[i] > currentMax)
    {
       currentMax = vec_in[i];
       currentIndex = i;
    }
    i++;
  }

  return(currentIndex+1);
}

int16_t differences(int16_t *pa,int16_t *pb, int length)
{
  int16_t d=0;
  int i=0;
  while(i < length)
  {
     if (*pa != *pb)
     {
       d++;
     }

     pa++;
     pb++;
     i++;
  }
  return(d);
}


    void Softmax::test_softmax_q7()
    {
       const q7_t *vec_in = input.ptr();
       q7_t *pTmp = temp.ptr();
       int16_t *pOut = output.ptr();
       int16_t maxIndex;

       for(int i=0; i <this->nbSamples;i++)
       {
          arm_softmax_q7(vec_in, this->vecDim, pTmp );
          maxIndex=findMaxIndex(pTmp,this->vecDim);
          *pOut++ = maxIndex;

          vec_in += this->vecDim;
       }

       printf("Nb diffs : %d\n",differences(ref.ptr(),output.ptr(),this->nbSamples));

       ASSERT_EQ(output,ref);
    } 

  
    void Softmax::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       switch(id)
       {
          case Softmax::TEST_SOFTMAX_Q7_1:
               ref.reload(Softmax::REF1_S16_ID,mgr);
               dims.reload(Softmax::DIMS1_S16_ID,mgr);
               input.reload(Softmax::INPUT1_Q7_ID,mgr);

               const int16_t *pDims=dims.ptr();

               this->nbSamples = pDims[0];
               this->vecDim = pDims[1];
          break; 

       }

        output.create(ref.nbSamples(),Softmax::OUTPUT_S16_ID,mgr);
        temp.create(this->vecDim,Softmax::TEMP_Q7_ID,mgr);
       

    }

    void Softmax::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        //output.dump(mgr);
        //temp.dump(mgr);
    }
