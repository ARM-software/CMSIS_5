/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 */

#include <stddef.h>
#include "video_drv.h"
#include "arm_vsi.h"
#ifdef _RTE_
#include "RTE_Components.h"
#endif
#include CMSIS_device_header

/* Video Peripheral definitions */
#define VideoO          ARM_VSI1                /* Video Output access struct */
#define VideoO_IRQn     ARM_VSI1_IRQn           /* Video Output Interrupt number */
#define VideoO_Handler  ARM_VSI1_Handler        /* Video Output Interrupt handler */
#define VideoI          ARM_VSI0                /* Video Input access struct */
#define VideoI_IRQn     ARM_VSI0_IRQn           /* Video Input Interrupt number */
#define VideoI_Handler  ARM_VSI0_Handler        /* Video Input Interrupt handler */

/* Video Peripheral registers */
#define CONTROL         Regs[0] /* Control receiver */
#define SAMPLE_BITS     Regs[1] /* Sample number of bits (8..32) */
#define SAMPLE_RATE     Regs[2] /* Sample rate (frame per second) */
#define STOP_SIMULATION Regs[4] /* Stop audio simulation */

/* Video Control register definitions */
#define CONTROL_ENABLE_Pos      0U                              /* CONTROL: ENABLE Position */
#define CONTROL_ENABLE_Msk      (1UL << CONTROL_ENABLE_Pos)     /* CONTROL: ENABLE Mask */

/* Driver State */
static uint8_t Initialized = 0U;

/* Event Callback */
static VideoDrv_Event_t CB_Event = NULL;


/* Video Input Interrupt Handler */
void VideoI_Handler (void) {

  VideoI->IRQ.Clear = 0x00000001U;
  __DSB();
  __ISB();
  if (CB_Event != NULL) {
    CB_Event(VIDEO_DRV_EVENT_RX_DATA);
  }
}


void VideoO_Handler (void) {

  VideoO->IRQ.Clear = 0x00000001U;
  __DSB();
  __ISB();
}


/* Initialize Video Interface */
int32_t VideoDrv_Initialize (VideoDrv_Event_t cb_event) {

  CB_Event = cb_event;

  /* Initialize Video Output peripheral */
  VideoO->Timer.Control = 0U;
  VideoO->DMA.Control   = 0U;
  VideoO->IRQ.Clear     = 0x00000001U;
  VideoO->IRQ.Enable    = 0x00000001U;
  VideoO->CONTROL       = 0U;

  /* Initialize Video Input peripheral */
  VideoI->Timer.Control = 0U;
  VideoI->DMA.Control   = 0U;
  VideoI->IRQ.Clear     = 0x00000001U;
  VideoI->IRQ.Enable    = 0x00000001U;
  VideoI->CONTROL       = 0U;

  /* Enable peripheral interrupts */
  NVIC->ISER[(((uint32_t)VideoI_IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)VideoI_IRQn) & 0x1FUL));
  __DSB();
  __ISB();

  Initialized = 1U;

  return VIDEO_DRV_OK;
}

/* De-initialize Video Interface */
int32_t VideoDrv_Uninitialize (void) {

  /* Disable peripheral interrupts */
  NVIC->ICER[(((uint32_t)VideoI_IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)VideoI_IRQn) & 0x1FUL));
  __DSB();
  __ISB();

  /* De-initialize Video Output peripheral */
  VideoO->Timer.Control = 0U;
  VideoO->DMA.Control   = 0U;
  VideoO->IRQ.Clear     = 0x00000001U;
  VideoO->IRQ.Enable    = 0x00000000U;
  VideoO->CONTROL       = 0U;

  /* De-initialize Video Input peripheral */
  VideoI->Timer.Control = 0U;
  VideoI->DMA.Control   = 0U;
  VideoI->IRQ.Clear     = 0x00000001U;
  VideoI->IRQ.Enable    = 0x00000000U;
  VideoI->CONTROL       = 0U;

  Initialized = 0U;

  return VIDEO_DRV_OK;
}

/* Configure Video Interface */
int32_t VideoDrv_Configure (uint32_t interface,  uint32_t pixel_size, uint32_t samplerate) {
  uint32_t format;

  if (Initialized == 0U) {
    return VIDEO_DRV_ERROR;
  }

  if ((pixel_size <  8*1U) ||
      (pixel_size > 8*2U)) {
    return VIDEO_DRV_ERROR_PARAMETER;
  }

  switch (interface) {
    case VIDEO_DRV_INTERFACE_RX:
      if ((VideoI->CONTROL & CONTROL_ENABLE_Msk) != 0U) {
        return VIDEO_DRV_ERROR;
      }
      VideoI->SAMPLE_BITS = pixel_size;
      VideoI->SAMPLE_RATE = samplerate;
      break;
    default:
      return VIDEO_DRV_ERROR_PARAMETER;
  }

  return VIDEO_DRV_OK;
}

/* Set Video Interface buffer */
int32_t VideoDrv_SetBuf (uint32_t interface, void *buf, uint32_t block_num, uint32_t block_size) {

  if (Initialized == 0U) {
    return VIDEO_DRV_ERROR;
  }

  switch (interface) {
    case VIDEO_DRV_INTERFACE_RX:
      if ((VideoI->DMA.Control & ARM_VSI_DMA_Enable_Msk) != 0U) {
        return VIDEO_DRV_ERROR;
      }
      VideoI->DMA.Address   = (uint32_t)buf;
      VideoI->DMA.BlockNum  = block_num;
      VideoI->DMA.BlockSize = block_size;
      break;
    default:
      return VIDEO_DRV_ERROR_PARAMETER;
  }

  return VIDEO_DRV_OK;
}

/* Control Video Interface */
int32_t VideoDrv_Control (uint32_t control) {
  uint32_t sample_size;
  uint32_t sample_rate;
  uint32_t block_size;

  if (Initialized == 0U) {
    return VIDEO_DRV_ERROR;
  }

  

  if ((control & VIDEO_DRV_CONTROL_RX_DISABLE) != 0U) {
    VideoI->Timer.Control = 0U;
    VideoI->DMA.Control   = 0U;
    VideoI->CONTROL       = 0U;
  } else if ((control & VIDEO_DRV_CONTROL_RX_ENABLE) != 0U) {
    VideoI->CONTROL       = CONTROL_ENABLE_Msk;
    VideoI->DMA.Control   = ARM_VSI_DMA_Direction_P2M |
                            ARM_VSI_DMA_Enable_Msk;
    sample_size = ((VideoI->SAMPLE_BITS + 7U) / 8U);
    sample_rate = VideoI->SAMPLE_RATE;
    if ((sample_size == 0U) || (sample_rate == 0U)) {
      VideoI->Timer.Interval = 0xFFFFFFFFU;
    } else {
      block_size = VideoI->DMA.BlockSize;
      VideoI->Timer.Interval = (1000000U * (block_size / sample_size)) / sample_rate;
    }
    VideoI->Timer.Control = ARM_VSI_Timer_Trig_DMA_Msk |
                            ARM_VSI_Timer_Trig_IRQ_Msk |
                            ARM_VSI_Timer_Periodic_Msk |
                            ARM_VSI_Timer_Run_Msk;
  }

  return VIDEO_DRV_OK;
}

/* Get received block count */
uint32_t VideoDrv_GetRxCount (void) {
  return (VideoI->Timer.Count);
}

/* Get Video Interface status */
VideoDrv_Status_t VideoDrv_GetStatus (void) {
  VideoDrv_Status_t status;
  uint32_t sr;


  if ((VideoI->CONTROL & CONTROL_ENABLE_Msk) != 0U) {
    status.rx_active = 1U;
  } else {
    status.rx_active = 0U;
  }

  return (status);
}


void VideoDrv_Stop (void)
{
  int32_t ret;
  ret = VideoDrv_Control(VIDEO_DRV_CONTROL_RX_DISABLE);
  
  VideoI->STOP_SIMULATION=1;
  
}


