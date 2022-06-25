/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 */

#ifndef __VIDEO_DRV_H
#define __VIDEO_DRV_H

#ifdef  __cplusplus
extern "C"
{
#endif

#include <stdint.h>

/* Video Interface */
#define VIDEO_DRV_INTERFACE_RX              (2U)  ///< Receiver

/* Video Control */
#define VIDEO_DRV_CONTROL_RX_ENABLE         (1UL << 1)  ///< Enable Receiver
#define VIDEO_DRV_CONTROL_RX_DISABLE        (1UL << 3)  ///< Disable Receiver

/* Video Event */
#define VIDEO_DRV_EVENT_RX_DATA             (1UL << 1)  ///< Data block received

/* Return code */
#define VIDEO_DRV_OK                        (0)  ///< Operation succeeded
#define VIDEO_DRV_ERROR                     (-1) ///< Unspecified error
#define VIDEO_DRV_ERROR_BUSY                (-2) ///< Driver is busy
#define VIDEO_DRV_ERROR_TIMEOUT             (-3) ///< Timeout occurred
#define VIDEO_DRV_ERROR_UNSUPPORTED         (-4) ///< Operation not supported
#define VIDEO_DRV_ERROR_PARAMETER           (-5) ///< Parameter error

/**
\brief Video Status
*/
typedef struct {
  uint32_t tx_active        :  1;       ///< Transmitter active
  uint32_t rx_active        :  1;       ///< Receiver active
  uint32_t reserved         : 30;
} VideoDrv_Status_t;

uint8_t* VideoRXBuffer();
int32_t VideoDrv_Setup(void);

/**
  \fn          VideoDrv_Event_t
  \brief       Video Events callback function type: void (*VideoDrv_Event_t) (uint32_t event
  \param[in]   event events notification mask
  \return      none
*/
typedef void (*VideoDrv_Event_t) (uint32_t event);

/**
  \fn          int32_t VideoDrv_Initialize (VideoDrv_Event_t cb_event)
  \brief       Initialize Video Interface.
  \param[in]   cb_event pointer to \ref VideoDrv_Event_t
  \return      return code
*/
int32_t VideoDrv_Initialize (VideoDrv_Event_t cb_event);

/**
  \fn          void VideoDrv_Stop (void);
  \brief       Stop audio simulation.
  \return      return code
*/
void VideoDrv_Stop (void);


/**
  \fn          int32_t VideoDrv_Uninitialize (void)
  \brief       De-initialize Video Interface.
  \return      return code
*/
int32_t VideoDrv_Uninitialize (void);

/**
  \fn          int32_t VideoDrv_Configure (uint32_t interface, uint32_t channels, uint32_t sample_bits, uint32_t sample_rate)
  \brief       Configure Video Interface.
  \param[in]   interface   audio interface
  \param[in]   pixel_size size in bytes
  \param[in]   samplerate samples per second
  \return      return code
*/
int32_t VideoDrv_Configure (uint32_t interface, uint32_t pixel_size,uint32_t samplerate);

/**
  \fn          int32_t VideoDrv_SetBuf (uint32_t interface, void *buf, uint32_t block_num, uint32_t block_size)
  \brief       Set Video Interface buffer.
  \param[in]   interface   audio interface
  \param[in]   buf         pointer to buffer for audio data
  \param[in]   block_num   number of blocks in buffer (must be 2^n)
  \param[in]   block_size  block size in number of samples
  \return      return code
*/
int32_t VideoDrv_SetBuf (uint32_t interface, void *buf, uint32_t block_num, uint32_t block_size);

/**
  \fn          int32_t VideoDrv_Control (uint32_t control)
  \brief       Control Video Interface.
  \param[in]   control operation
  \return      return code
*/
int32_t VideoDrv_Control (uint32_t control);


/**
  \fn          uint32_t VideoDrv_GetRxCount (void)
  \brief       Get received block count.
  \return      number of received blocks
*/
uint32_t VideoDrv_GetRxCount (void);

/**
  \fn          VideoDrv_Status_t VideoDrv_GetStatus (void)
  \brief       Get Video Interface status.
  \return      \ref VideoDrv_Status_t
*/
VideoDrv_Status_t VideoDrv_GetStatus (void);

#ifdef  __cplusplus
}
#endif

#endif /* __VIDEO_DRV_H */
