#include <stddef.h>
#include "video_drv.h"
#include "arm_vsi.h"
#ifdef _RTE_
#include "RTE_Components.h"
#endif
#include CMSIS_device_header

#include "cmsis_os2.h"



#include "RingBuffer.h"

#include "arm_math.h"

#include "SchedEvents.h"
#include "VideoConfig.h"
#include "RingConfig.h"

#include "RingInit.h"

extern osThreadId_t gStreamingThreadID;

// Number of bytes read by DMA
#define VIDEO_BLOCK_SIZE    RING_BUFSIZE_RX

// Number of DMA blocks
#define VIDEO_DMA_NB_BLOCKS RING_NBBUFS


extern int32_t VideoDrv_Setup(void);


extern ring_config_t ringConfigRX;

#ifdef __FVP_PY
__attribute__((section(".ARM.__at_0x90000000")))
#endif
__ALIGNED(16) static uint8_t video_bufferRX[VIDEO_DMA_NB_BLOCKS*VIDEO_BLOCK_SIZE];
static uint8_t *reservedBufRX=NULL;


uint8_t* VideoRXBuffer()
{
  return(video_bufferRX);
}


static void VideoEvent (uint32_t event) {

  if (event & VIDEO_DRV_EVENT_RX_DATA) 
  {
    
    
    ringInterruptReleaseBuffer(&ringConfigRX,(void *)gStreamingThreadID);
    int reservedRX=ringInterruptReserveBuffer(&ringConfigRX);
    reservedBufRX=ringGetBufferAddress(&ringConfigRX,reservedRX);

  }

}

int32_t VideoDrv_Setup(void) {
  int32_t ret;

  ret = VideoDrv_Initialize(VideoEvent);
  if (ret != 0) {
    return ret;
  }


  ret = VideoDrv_Configure(VIDEO_DRV_INTERFACE_RX,
                           8U * VIDEO_DRV_PIXEL_SIZE, /* 16 sample bits */
                           static_cast<uint32_t>(VIDEO_DRV_FRAME_RATE*VIDEO_DRV_WIDTH*VIDEO_DRV_HEIGHT));
  if (ret != 0) {
    return ret;
  }

  /* Work because user process not started yet
  */

  int reservedRX=ringInterruptReserveBuffer(&ringConfigRX);
  reservedBufRX=ringGetBufferAddress(&ringConfigRX,reservedRX);


  ret = VideoDrv_SetBuf(VIDEO_DRV_INTERFACE_RX,
                        video_bufferRX, VIDEO_DMA_NB_BLOCKS,VIDEO_BLOCK_SIZE);
  if (ret != 0) {
    return ret;
  }

  ret = VideoDrv_Control(VIDEO_DRV_CONTROL_RX_ENABLE);
  if (ret != 0) {
    return ret;
  }


  return 0;
}

