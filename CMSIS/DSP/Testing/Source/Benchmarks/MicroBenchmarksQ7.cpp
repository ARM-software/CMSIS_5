#include "MicroBenchmarksQ7.h"
#include "Error.h"

static void add_while_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
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

static void add_for_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
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

static void add_array_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
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
   
    void MicroBenchmarksQ7::test_while_q7()
    {     
      add_while_q7(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksQ7::test_for_q7()
    {     
      add_for_q7(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksQ7::test_array_q7()
    {     
      add_array_q7(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    
    void MicroBenchmarksQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       input1.reload(MicroBenchmarksQ7::INPUT1_Q7_ID,mgr,this->nbSamples);
       input2.reload(MicroBenchmarksQ7::INPUT2_Q7_ID,mgr,this->nbSamples);

       
       output.create(this->nbSamples,MicroBenchmarksQ7::OUT_SAMPLES_Q7_ID,mgr);

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();

    }

    void MicroBenchmarksQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
