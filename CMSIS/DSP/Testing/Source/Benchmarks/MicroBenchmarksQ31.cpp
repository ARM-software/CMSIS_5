#include "MicroBenchmarksQ31.h"
#include "Error.h"

static void add_while_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t blockSize)
{
  uint32_t blkCnt;    

  blkCnt = blockSize;

  while (blkCnt > 0U)
  {
    /* C = A + B */

    /* Add and store result in destination buffer. */
    *pDst++ = (*pSrcA++) + (*pSrcB++);

    /* Decrement loop counter */
    blkCnt--;
  }
}

static void add_for_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t blockSize)
{
  uint32_t blkCnt;   
  int32_t i; 

  blkCnt = blockSize;

  for(i=0; i<blkCnt; i++)
  {
    /* C = A + B */

    /* Add and store result in destination buffer. */
    *pDst++ = (*pSrcA++) + (*pSrcB++);

  }
}

static void add_array_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        q31_t * pDst,
        uint32_t blockSize)
{
  uint32_t blkCnt;   
  int32_t i; 

  blkCnt = blockSize;

  for(i=0; i<blkCnt; i++)
  {
    /* C = A + B */

    /* Add and store result in destination buffer. */
    pDst[i] = pSrcA[i] + pSrcB[i];

  }
}
   
    void MicroBenchmarksQ31::test_while_q31()
    {     
      add_while_q31(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksQ31::test_for_q31()
    {     
      add_for_q31(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksQ31::test_array_q31()
    {     
      add_array_q31(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    
    void MicroBenchmarksQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       input1.reload(MicroBenchmarksQ31::INPUT1_Q31_ID,mgr,this->nbSamples);
       input2.reload(MicroBenchmarksQ31::INPUT2_Q31_ID,mgr,this->nbSamples);

       
       output.create(this->nbSamples,MicroBenchmarksQ31::OUT_SAMPLES_Q31_ID,mgr);

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();

    }

    void MicroBenchmarksQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
