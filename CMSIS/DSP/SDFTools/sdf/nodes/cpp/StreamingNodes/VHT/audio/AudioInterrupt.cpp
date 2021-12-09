#include <stddef.h>
#include "audio_drv.h"
#include "arm_vsi.h"
#ifdef _RTE_
#include "RTE_Components.h"
#endif
#include CMSIS_device_header

#include "cmsis_os2.h"



#include "RingBuffer.h"

#include "arm_math.h"

#include "SchedEvents.h"
#include "AudioConfig.h"
#include "RingConfig.h"

#include "RingInit.h"

extern osThreadId_t gStreamingThreadID;

// Number of bytes read by DMA
#define AUDIO_BLOCK_SIZE_RX    RING_BUFSIZE_RX
#define AUDIO_BLOCK_SIZE_TX    RING_BUFSIZE_TX

// Number of DMA blocks
#define AUDIO_DMA_NB_BLOCKS RING_NBBUFS




#if AUDIO_DRV_RX_ENABLED
extern ring_config_t ringConfigRX;

#ifdef __FVP_PY
__attribute__((section(".ARM.__at_0x90000000")))
#endif
#if SDF_VHT_TX_RX_ORDERING
__ALIGNED(16) static uint8_t dmaRX[AUDIO_BLOCK_SIZE_RX];
int rxCount=0;
#endif
__ALIGNED(16) static uint8_t audio_bufferRX[AUDIO_DMA_NB_BLOCKS*AUDIO_BLOCK_SIZE_RX];
static uint8_t *reservedBufRX=NULL;
#endif

#if AUDIO_DRV_TX_ENABLED
extern ring_config_t ringConfigTX;

#ifdef __FVP_PY
__attribute__((section(".ARM.__at_0x9FFF0000")))
#endif
#if SDF_VHT_TX_RX_ORDERING
__ALIGNED(16) static uint8_t dmaTX[AUDIO_BLOCK_SIZE_TX];
int txCount=0;
#endif
__ALIGNED(16) static uint8_t audio_bufferTX[AUDIO_DMA_NB_BLOCKS*AUDIO_BLOCK_SIZE_TX];
static uint8_t *reservedBufTX=NULL;
#endif

uint8_t* AudioRXBuffer()
{
#if AUDIO_DRV_RX_ENABLED
  return(audio_bufferRX);
#else
    return(NULL);
#endif
}

uint8_t* AudioTXBuffer()
{
#if AUDIO_DRV_TX_ENABLED
    return(audio_bufferTX);
#else
    return(NULL);
#endif
}

static void AudioEvent (uint32_t event) {

#if AUDIO_DRV_RX_ENABLED
  if (event & AUDIO_DRV_EVENT_RX_DATA) 
  {
    
    #if SDF_VHT_TX_RX_ORDERING
      memcpy(reservedBufRX,dmaRX,RING_BUFSIZE_RX);
      (void)AudioDrv_Control(AUDIO_DRV_CONTROL_RX_DISABLE);
      (void)AudioDrv_Control(AUDIO_DRV_CONTROL_TX_ENABLE);
    #endif
    ringInterruptReleaseBuffer(&ringConfigRX,(void *)gStreamingThreadID);
    int reservedRX=ringInterruptReserveBuffer(&ringConfigRX);
    reservedBufRX=ringGetBufferAddress(&ringConfigRX,reservedRX);

  }
#endif

#if AUDIO_DRV_TX_ENABLED
  if (event & AUDIO_DRV_EVENT_TX_DATA)
  {
    #if SDF_VHT_TX_RX_ORDERING
          memcpy(dmaTX,reservedBufTX,RING_BUFSIZE_TX);
          (void)AudioDrv_Control(AUDIO_DRV_CONTROL_TX_DISABLE);
          (void)AudioDrv_Control(AUDIO_DRV_CONTROL_RX_ENABLE);
    #endif
    ringInterruptReleaseBuffer(&ringConfigTX,(void *)gStreamingThreadID);
    int reservedTX=ringInterruptReserveBuffer(&ringConfigTX);
    reservedBufTX=ringGetBufferAddress(&ringConfigTX,reservedTX);
  }
#endif
}

int32_t AudioDrv_Setup(void) {
  int32_t ret;

  ret = AudioDrv_Initialize(AudioEvent);
  if (ret != 0) {
    return ret;
  }

#if AUDIO_DRV_RX_ENABLED

  ret = AudioDrv_Configure(AUDIO_DRV_INTERFACE_RX,
                           AUDIO_DRV_NBCHANNELS_RX,  /* single channel */
                           8U * AUDIO_DRV_CHANNEL_ENCODING_RX, /* 16 sample bits */
                           static_cast<uint32_t>(AUDIO_DRV_SAMPLINGFREQUENCY_RX));
  if (ret != 0) {
    return ret;
  }

  /* Work because user process not started yet
  */

  int reservedRX=ringInterruptReserveBuffer(&ringConfigRX);
  reservedBufRX=ringGetBufferAddress(&ringConfigRX,reservedRX);

#if SDF_VHT_TX_RX_ORDERING
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_RX,
                        dmaRX, 1,AUDIO_BLOCK_SIZE_RX);
#else
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_RX,
                        audio_bufferRX, AUDIO_DMA_NB_BLOCKS,AUDIO_BLOCK_SIZE_RX);
#endif
  if (ret != 0) {
    return ret;
  }

#if !SDF_VHT_TX_RX_ORDERING
  ret = AudioDrv_Control(AUDIO_DRV_CONTROL_RX_ENABLE);
  if (ret != 0) {
    return ret;
  }
#endif 

#endif /* AUDIO_DRV_RX_ENABLED */

#if AUDIO_DRV_TX_ENABLED
  ret = AudioDrv_Configure(AUDIO_DRV_INTERFACE_TX,
                           AUDIO_DRV_NBCHANNELS_TX,  /* single channel */
                           8U * AUDIO_DRV_CHANNEL_ENCODING_TX, /* 16 sample bits */
                           static_cast<uint32_t>(AUDIO_DRV_SAMPLINGFREQUENCY_TX));
  if (ret != 0) {
    return ret;
  }

  /* Work because user process not started yet
  */

  /* dataflow must be 1 packet ahead of the TX interrupt*/
  ringUserReserveBuffer(&ringConfigTX);
  ringUserReleaseBuffer(&ringConfigTX);

  ringUserReserveBuffer(&ringConfigTX);
  ringUserReleaseBuffer(&ringConfigTX);

  int reservedTX=ringInterruptReserveBuffer(&ringConfigTX);
  reservedBufTX=ringGetBufferAddress(&ringConfigTX,reservedTX);

#if SDF_VHT_TX_RX_ORDERING
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_TX,
                        dmaTX, 1 ,AUDIO_BLOCK_SIZE_TX);
#else
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_TX,
                        audio_bufferTX, AUDIO_DMA_NB_BLOCKS,AUDIO_BLOCK_SIZE_TX);
#endif

  if (ret != 0) {
    return ret;
  }

  ret = AudioDrv_Control(AUDIO_DRV_CONTROL_TX_ENABLE);
  if (ret != 0) {
    return ret;
  }

#endif /* AUDIO_DRV_TX_ENABLED */


  return 0;
}

