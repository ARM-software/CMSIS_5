/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 */

#include <stddef.h>
#include "audio_drv.h"
#include "arm_vsi.h"
#ifdef _RTE_
#include "RTE_Components.h"
#endif
#include CMSIS_device_header

/* Audio Peripheral definitions */
#define AudioO          ARM_VSI1                /* Audio Output access struct */
#define AudioO_IRQn     ARM_VSI1_IRQn           /* Audio Output Interrupt number */
#define AudioO_Handler  ARM_VSI1_Handler        /* Audio Output Interrupt handler */
#define AudioI          ARM_VSI0                /* Audio Input access struct */
#define AudioI_IRQn     ARM_VSI0_IRQn           /* Audio Input Interrupt number */
#define AudioI_Handler  ARM_VSI0_Handler        /* Audio Input Interrupt handler */

/* Audio Peripheral registers */
#define CONTROL         Regs[0] /* Control receiver */
#define CHANNELS        Regs[1] /* Number of channels */
#define SAMPLE_BITS     Regs[2] /* Sample number of bits (8..32) */
#define SAMPLE_RATE     Regs[3] /* Sample rate (samples per second) */
#define STOP_SIMULATION Regs[4] /* Stop audio simulation */

/* Audio Control register definitions */
#define CONTROL_ENABLE_Pos      0U                              /* CONTROL: ENABLE Position */
#define CONTROL_ENABLE_Msk      (1UL << CONTROL_ENABLE_Pos)     /* CONTROL: ENABLE Mask */

/* Driver State */
static uint8_t Initialized = 0U;

/* Event Callback */
static AudioDrv_Event_t CB_Event = NULL;

/* Audio Output Interrupt Handler */
void AudioO_Handler (void) {

  AudioO->IRQ.Clear = 0x00000001U;
  __DSB();
  __ISB();
  if (CB_Event != NULL) {
    CB_Event(AUDIO_DRV_EVENT_TX_DATA);
  }
}

/* Audio Input Interrupt Handler */
void AudioI_Handler (void) {

  AudioI->IRQ.Clear = 0x00000001U;
  __DSB();
  __ISB();
  if (CB_Event != NULL) {
    CB_Event(AUDIO_DRV_EVENT_RX_DATA);
  }
}



/* Initialize Audio Interface */
int32_t AudioDrv_Initialize (AudioDrv_Event_t cb_event) {

  CB_Event = cb_event;

  /* Initialize Audio Output peripheral */
  AudioO->Timer.Control = 0U;
  AudioO->DMA.Control   = 0U;
  AudioO->IRQ.Clear     = 0x00000001U;
  AudioO->IRQ.Enable    = 0x00000001U;
  AudioO->CONTROL       = 0U;

  /* Initialize Audio Input peripheral */
  AudioI->Timer.Control = 0U;
  AudioI->DMA.Control   = 0U;
  AudioI->IRQ.Clear     = 0x00000001U;
  AudioI->IRQ.Enable    = 0x00000001U;
  AudioI->CONTROL       = 0U;

  /* Enable peripheral interrupts */
//NVIC_EnableIRQ(AudioO_IRQn);
  NVIC->ISER[(((uint32_t)AudioO_IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)AudioO_IRQn) & 0x1FUL));
//NVIC_EnableIRQ(AudioI_IRQn);
  NVIC->ISER[(((uint32_t)AudioI_IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)AudioI_IRQn) & 0x1FUL));
  __DSB();
  __ISB();

  Initialized = 1U;

  return AUDIO_DRV_OK;
}

/* De-initialize Audio Interface */
int32_t AudioDrv_Uninitialize (void) {

  /* Disable peripheral interrupts */
//NVIC_DisableIRQ(AudioO_IRQn);
  NVIC->ICER[(((uint32_t)AudioO_IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)AudioO_IRQn) & 0x1FUL));
//NVIC_DisableIRQ(AudioI_IRQn);
  NVIC->ICER[(((uint32_t)AudioI_IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)AudioI_IRQn) & 0x1FUL));
  __DSB();
  __ISB();

  /* De-initialize Audio Output peripheral */
  AudioO->Timer.Control = 0U;
  AudioO->DMA.Control   = 0U;
  AudioO->IRQ.Clear     = 0x00000001U;
  AudioO->IRQ.Enable    = 0x00000000U;
  AudioO->CONTROL       = 0U;

  /* De-initialize Audio Input peripheral */
  AudioI->Timer.Control = 0U;
  AudioI->DMA.Control   = 0U;
  AudioI->IRQ.Clear     = 0x00000001U;
  AudioI->IRQ.Enable    = 0x00000000U;
  AudioI->CONTROL       = 0U;

  Initialized = 0U;

  return AUDIO_DRV_OK;
}

/* Configure Audio Interface */
int32_t AudioDrv_Configure (uint32_t interface, uint32_t channels, uint32_t sample_bits, uint32_t sample_rate) {
  uint32_t format;

  if (Initialized == 0U) {
    return AUDIO_DRV_ERROR;
  }

  if ((channels <  1U) ||
      (channels > 32U) ||
      (sample_bits <  8U) ||
      (sample_bits > 32U) ||
      (sample_rate == 0U)) {
    return AUDIO_DRV_ERROR_PARAMETER;
  }

  switch (interface) {
    case AUDIO_DRV_INTERFACE_TX:
      if ((AudioO->CONTROL & CONTROL_ENABLE_Msk) != 0U) {
        return AUDIO_DRV_ERROR;
      }
      AudioO->CHANNELS    = channels;
      AudioO->SAMPLE_BITS = sample_bits;
      AudioO->SAMPLE_RATE = sample_rate;
      break;
    case AUDIO_DRV_INTERFACE_RX:
      if ((AudioI->CONTROL & CONTROL_ENABLE_Msk) != 0U) {
        return AUDIO_DRV_ERROR;
      }
      AudioI->CHANNELS    = channels;
      AudioI->SAMPLE_BITS = sample_bits;
      AudioI->SAMPLE_RATE = sample_rate;
      break;
    default:
      return AUDIO_DRV_ERROR_PARAMETER;
  }

  return AUDIO_DRV_OK;
}

/* Set Audio Interface buffer */
int32_t AudioDrv_SetBuf (uint32_t interface, void *buf, uint32_t block_num, uint32_t block_size) {

  if (Initialized == 0U) {
    return AUDIO_DRV_ERROR;
  }

  switch (interface) {
    case AUDIO_DRV_INTERFACE_TX:
      if ((AudioO->DMA.Control & ARM_VSI_DMA_Enable_Msk) != 0U) {
        return AUDIO_DRV_ERROR;
      }
      AudioO->DMA.Address   = (uint32_t)buf;
      AudioO->DMA.BlockNum  = block_num;
      AudioO->DMA.BlockSize = block_size;
      break;
    case AUDIO_DRV_INTERFACE_RX:
      if ((AudioI->DMA.Control & ARM_VSI_DMA_Enable_Msk) != 0U) {
        return AUDIO_DRV_ERROR;
      }
      AudioI->DMA.Address   = (uint32_t)buf;
      AudioI->DMA.BlockNum  = block_num;
      AudioI->DMA.BlockSize = block_size;
      break;
    default:
      return AUDIO_DRV_ERROR_PARAMETER;
  }

  return AUDIO_DRV_OK;
}

/* Control Audio Interface */
int32_t AudioDrv_Control (uint32_t control) {
  uint32_t sample_size;
  uint32_t sample_rate;
  uint32_t block_size;

  if (Initialized == 0U) {
    return AUDIO_DRV_ERROR;
  }

  if ((control & AUDIO_DRV_CONTROL_TX_DISABLE) != 0U) {
    AudioO->Timer.Control = 0U;
    AudioO->DMA.Control   = 0U;
    AudioO->CONTROL       = 0U;
  } else if ((control & AUDIO_DRV_CONTROL_TX_ENABLE) != 0U) {
    AudioO->CONTROL       = CONTROL_ENABLE_Msk;
    AudioO->DMA.Control   = ARM_VSI_DMA_Direction_M2P |
                            ARM_VSI_DMA_Enable_Msk;
    sample_size = AudioO->CHANNELS * ((AudioO->SAMPLE_BITS + 7U) / 8U);
    sample_rate = AudioO->SAMPLE_RATE;
    if ((sample_size == 0U) || (sample_rate == 0U)) {
      AudioO->Timer.Interval = 0xFFFFFFFFU;
    } else {
      block_size = AudioO->DMA.BlockSize;
      AudioO->Timer.Interval = (1000000U * (block_size / sample_size)) / sample_rate;
    }
    AudioO->Timer.Control = ARM_VSI_Timer_Trig_DMA_Msk |
                            ARM_VSI_Timer_Trig_IRQ_Msk |
                            ARM_VSI_Timer_Periodic_Msk |
                            ARM_VSI_Timer_Run_Msk;
  }

  if ((control & AUDIO_DRV_CONTROL_RX_DISABLE) != 0U) {
    AudioI->Timer.Control = 0U;
    AudioI->DMA.Control   = 0U;
    AudioI->CONTROL       = 0U;
  } else if ((control & AUDIO_DRV_CONTROL_RX_ENABLE) != 0U) {
    AudioI->CONTROL       = CONTROL_ENABLE_Msk;
    AudioI->DMA.Control   = ARM_VSI_DMA_Direction_P2M |
                            ARM_VSI_DMA_Enable_Msk;
    sample_size = AudioI->CHANNELS * ((AudioI->SAMPLE_BITS + 7U) / 8U);
    sample_rate = AudioI->SAMPLE_RATE;
    if ((sample_size == 0U) || (sample_rate == 0U)) {
      AudioI->Timer.Interval = 0xFFFFFFFFU;
    } else {
      block_size = AudioI->DMA.BlockSize;
      AudioI->Timer.Interval = (1000000U * (block_size / sample_size)) / sample_rate;
    }
    AudioI->Timer.Control = ARM_VSI_Timer_Trig_DMA_Msk |
                            ARM_VSI_Timer_Trig_IRQ_Msk |
                            ARM_VSI_Timer_Periodic_Msk |
                            ARM_VSI_Timer_Run_Msk;
  }

  return AUDIO_DRV_OK;
}

/* Get transmitted block count */
uint32_t AudioDrv_GetTxCount (void) {
  return (AudioO->Timer.Count);
}

/* Get received block count */
uint32_t AudioDrv_GetRxCount (void) {
  return (AudioI->Timer.Count);
}

/* Get Audio Interface status */
AudioDrv_Status_t AudioDrv_GetStatus (void) {
  AudioDrv_Status_t status;
  uint32_t sr;

  if ((AudioO->CONTROL & CONTROL_ENABLE_Msk) != 0U) {
    status.tx_active = 1U;
  } else {
    status.tx_active = 0U;
  }

  if ((AudioI->CONTROL & CONTROL_ENABLE_Msk) != 0U) {
    status.rx_active = 1U;
  } else {
    status.rx_active = 0U;
  }

  return (status);
}


void AudioDrv_Stop (void)
{
  int32_t ret;
  ret = AudioDrv_Control(AUDIO_DRV_CONTROL_TX_DISABLE);
  ret = AudioDrv_Control(AUDIO_DRV_CONTROL_RX_DISABLE);
  
  AudioO->STOP_SIMULATION=1;
  AudioI->STOP_SIMULATION=1;
  
}