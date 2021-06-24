#include "MicroBenchmarksQ15.h"
#include "Error.h"

static void add_while_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
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

static void add_for_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
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

static void add_array_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
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
   
    void MicroBenchmarksQ15::test_while_q15()
    {     
      add_while_q15(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksQ15::test_for_q15()
    {     
      add_for_q15(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksQ15::test_array_q15()
    {     
      add_array_q15(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    
    void MicroBenchmarksQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       input1.reload(MicroBenchmarksQ15::INPUT1_Q15_ID,mgr,this->nbSamples);
       input2.reload(MicroBenchmarksQ15::INPUT2_Q15_ID,mgr,this->nbSamples);

       
       output.create(this->nbSamples,MicroBenchmarksQ15::OUT_SAMPLES_Q15_ID,mgr);

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();

    }

    void MicroBenchmarksQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
