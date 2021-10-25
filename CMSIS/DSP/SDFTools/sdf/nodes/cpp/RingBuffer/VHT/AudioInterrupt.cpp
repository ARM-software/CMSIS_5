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
#include "RingConfig.h"

#include "RingInit.h"

extern osThreadId_t gAudioThreadID;

// Number of bytes read by DMA
#define AUDIO_BLOCK_SIZE    RING_BUFSIZE

// Number of DMA blocks
#define AUDIO_DMA_NB_BLOCKS RING_NBBUFS


extern int32_t AudioDrv_Setup(void);


#if RX_ENABLED
extern ring_config_t ringConfigRX;

#ifdef __FVP_PY
__attribute__((section(".ARM.__at_0x90000000")))
#endif
#if SDF_VHT_TX_RX_ORDERING
__ALIGNED(16) static uint8_t dmaRX[AUDIO_BLOCK_SIZE];
int rxCount=0;
#endif
__ALIGNED(16) static uint8_t audio_bufferRX[AUDIO_DMA_NB_BLOCKS*AUDIO_BLOCK_SIZE];
static uint8_t *reservedBufRX=NULL;
#endif

#if TX_ENABLED
extern ring_config_t ringConfigTX;

#ifdef __FVP_PY
__attribute__((section(".ARM.__at_0x9FFF0000")))
#endif
#if SDF_VHT_TX_RX_ORDERING
__ALIGNED(16) static uint8_t dmaTX[AUDIO_BLOCK_SIZE];
int txCount=0;
#endif
__ALIGNED(16) static uint8_t audio_bufferTX[AUDIO_DMA_NB_BLOCKS*AUDIO_BLOCK_SIZE];
static uint8_t *reservedBufTX=NULL;
#endif

uint8_t* AudioRXBuffer()
{
  return(audio_bufferRX);
}

uint8_t* AudioTXBuffer()
{
  return(audio_bufferTX);
}

static void AudioEvent (uint32_t event) {

#if RX_ENABLED
  if (event & AUDIO_DRV_EVENT_RX_DATA) 
  {
    
    #if SDF_VHT_TX_RX_ORDERING
      memcpy(reservedBufRX,dmaRX,RING_BUFSIZE);
      (void)AudioDrv_Control(AUDIO_DRV_CONTROL_RX_DISABLE);
      (void)AudioDrv_Control(AUDIO_DRV_CONTROL_TX_ENABLE);
    #endif
    ringInterruptReleaseBuffer(&ringConfigRX,(void *)gAudioThreadID);
    int reservedRX=ringInterruptReserveBuffer(&ringConfigRX);
    reservedBufRX=ringGetBufferAddress(&ringConfigRX,reservedRX);

  }
#endif

#if TX_ENABLED
  if (event & AUDIO_DRV_EVENT_TX_DATA)
  {
    #if SDF_VHT_TX_RX_ORDERING
          memcpy(dmaTX,reservedBufTX,RING_BUFSIZE);
          (void)AudioDrv_Control(AUDIO_DRV_CONTROL_TX_DISABLE);
          (void)AudioDrv_Control(AUDIO_DRV_CONTROL_RX_ENABLE);
    #endif
    ringInterruptReleaseBuffer(&ringConfigTX,(void *)gAudioThreadID);
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

#if RX_ENABLED

  ret = AudioDrv_Configure(AUDIO_DRV_INTERFACE_RX,
                           AUDIO_NBCHANNELS,  /* single channel */
                           8U * AUDIO_CHANNEL_ENCODING, /* 16 sample bits */
                           static_cast<uint32_t>(AUDIO_SAMPLINGFREQUENCY));
  if (ret != 0) {
    return ret;
  }

  /* Work because user process not started yet
  */

  int reservedRX=ringInterruptReserveBuffer(&ringConfigRX);
  reservedBufRX=ringGetBufferAddress(&ringConfigRX,reservedRX);

#if SDF_VHT_TX_RX_ORDERING
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_RX,
                        dmaRX, 1,AUDIO_BLOCK_SIZE);
#else
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_RX,
                        audio_bufferRX, AUDIO_DMA_NB_BLOCKS,AUDIO_BLOCK_SIZE);
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

#endif /* RX_ENABLED */

#if TX_ENABLED
  ret = AudioDrv_Configure(AUDIO_DRV_INTERFACE_TX,
                           AUDIO_NBCHANNELS,  /* single channel */
                           8U * AUDIO_CHANNEL_ENCODING, /* 16 sample bits */
                           static_cast<uint32_t>(AUDIO_SAMPLINGFREQUENCY));
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
                        dmaTX, 1 ,AUDIO_BLOCK_SIZE);
#else
  ret = AudioDrv_SetBuf(AUDIO_DRV_INTERFACE_TX,
                        audio_bufferTX, AUDIO_DMA_NB_BLOCKS,AUDIO_BLOCK_SIZE);
#endif

  if (ret != 0) {
    return ret;
  }

  ret = AudioDrv_Control(AUDIO_DRV_CONTROL_TX_ENABLE);
  if (ret != 0) {
    return ret;
  }

#endif /* TX_ENABLED */


  return 0;
}

