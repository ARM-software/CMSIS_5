#include "ref.h"

void ref_sub_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
  float32_t * pDst,
  uint32_t blockSize)
{
	uint32_t i;
	
	for(i=0;i<blockSize;i++)
	{
		pDst[i] = pSrcA[i] - pSrcB[i];
	}
}

void ref_sub_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
  q31_t * pDst,
  uint32_t blockSize)
{
	uint32_t i;
	
	for(i=0;i<blockSize;i++)
	{
		pDst[i] = ref_sat_q31( (q63_t)pSrcA[i] - pSrcB[i] );
	}
}

void ref_sub_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
  q15_t * pDst,
  uint32_t blockSize)
{
	uint32_t i;
	
	for(i=0;i<blockSize;i++)
	{
		pDst[i] = ref_sat_q15( (q31_t)pSrcA[i] - pSrcB[i] );
	}
}

void ref_sub_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
  q7_t * pDst,
  uint32_t blockSize)
{
	uint32_t i;
	
	for(i=0;i<blockSize;i++)
	{
		pDst[i] = ref_sat_q7( (q15_t)pSrcA[i] - pSrcB[i] );
	}
}
