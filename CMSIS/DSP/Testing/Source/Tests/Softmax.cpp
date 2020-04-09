#include "Softmax.h"
#include <stdio.h>
#include "Error.h"
#include "arm_nnfunctions.h"
#include "Test.h"

/*

Tests have shown that, compared to a float32 implementation
there is an average error of 4.2 percent and standard deviation
of 0.89.

Which means that with 100 batches, 4 batches will give the wrong position
for the max.

But it depends highly of the vector dimension.

Regressions are giving:

Average error rate = -0.555548 + 0.246918 vecDim
Variance = -0.0112281 + 0.0382476 vecDim

So for vecDim = 21 we have
Average error rate = 4.6 percent
Variance for error rate = 0.8

This data is used to define the threshold for tests

*/
#define THRESHOLD 7.5

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
          pTmp += this->vecDim;
       }

       int diff = differences(ref.ptr(),output.ptr(),this->nbSamples);
       
       ASSERT_TRUE(100.0*diff/this->nbSamples <= THRESHOLD);
       
    } 

    void Softmax::test_softmax_with_batch_q7()
    {
       const q7_t *vec_in = input.ptr();
       q7_t *pTmp = temp.ptr();
       int16_t *pOut = output.ptr();
       int16_t maxIndex;

       arm_softmax_with_batch_q7(vec_in, this->nbSamples,this->vecDim, pTmp );

       for(int i=0; i <this->nbSamples;i++)
       {
          maxIndex=findMaxIndex(pTmp,this->vecDim);
          *pOut++ = maxIndex;
          pTmp += this->vecDim;
       }

       int diff = differences(ref.ptr(),output.ptr(),this->nbSamples);
       
       ASSERT_TRUE(100.0*diff/this->nbSamples <= THRESHOLD);
       
    } 

  
    void Softmax::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       switch(id)
       {
          case Softmax::TEST_SOFTMAX_Q7_1:
          {
               ref.reload(Softmax::REF1_S16_ID,mgr);
               dims.reload(Softmax::DIMS1_S16_ID,mgr);
               input.reload(Softmax::INPUT1_Q7_ID,mgr);

               const int16_t *pDims=dims.ptr();

               this->nbSamples = pDims[0];
               this->vecDim = pDims[1];
          }
          break; 
          
          case Softmax::TEST_SOFTMAX_WITH_BATCH_Q7_2:
          {
               ref.reload(Softmax::REF1_S16_ID,mgr);
               dims.reload(Softmax::DIMS1_S16_ID,mgr);
               input.reload(Softmax::INPUT1_Q7_ID,mgr);

               const int16_t *pDims=dims.ptr();

               this->nbSamples = pDims[0];
               this->vecDim = pDims[1];
          }
          break; 

       }

        output.create(ref.nbSamples(),Softmax::OUTPUT_S16_ID,mgr);
        // Used to compare bit exactness of the reference C version
        // and the optimized version.
        temp.create(this->vecDim*this->nbSamples,Softmax::TEMP_Q7_ID,mgr);
       

    }

    void Softmax::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        // Array are big so by default they are not dumped and only
        // used for debug.
        //output.dump(mgr);
        //temp.dump(mgr);
    }
