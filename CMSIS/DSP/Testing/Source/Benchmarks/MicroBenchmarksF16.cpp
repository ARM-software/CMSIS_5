#include "MicroBenchmarksF16.h"
#include "Error.h"

static void add_while_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
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

static void add_for_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
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

static void add_array_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
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
   
    void MicroBenchmarksF16::test_while_f16()
    {     
      add_while_f16(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksF16::test_for_f16()
    {     
      add_for_f16(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    void MicroBenchmarksF16::test_array_f16()
    {     
      add_array_f16(this->inp1,this->inp2,this->outp,this->nbSamples);
    } 

    
    void MicroBenchmarksF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       input1.reload(MicroBenchmarksF16::INPUT1_F16_ID,mgr,this->nbSamples);
       input2.reload(MicroBenchmarksF16::INPUT2_F16_ID,mgr,this->nbSamples);

       
       output.create(this->nbSamples,MicroBenchmarksF16::OUT_SAMPLES_F16_ID,mgr);

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();

    }

    void MicroBenchmarksF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
