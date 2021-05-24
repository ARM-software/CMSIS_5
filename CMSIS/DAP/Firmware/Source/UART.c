/*
 * Copyright (c) 2021 ARM Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ----------------------------------------------------------------------
 *
 * $Date:        1. March 2021
 * $Revision:    V1.0.0
 *
 * Project:      CMSIS-DAP Source
 * Title:        UART.c CMSIS-DAP UART
 *
 *---------------------------------------------------------------------------*/

#include "DAP_config.h"
#include "DAP.h"

#if (DAP_UART != 0)

#ifdef DAP_FW_V1
#error "UART Communication Port not supported in DAP V1!"
#endif

#include "Driver_USART.h"

#include "cmsis_os2.h"
#include <string.h>

#define UART_TX_BUF_SIZE    1024U   /* Uart Tx Buffer (must be 2^n) */
#define UART_RX_BUF_SIZE    1024U   /* Uart Rx Buffer (must be 2^n) */
#define UART_RX_BLOCK_SIZE    32U   /* Uart Rx Block Size (must be 2^n) */
#define UART_MAX_XFER_NUM    509U

// USART Driver
#define _USART_Driver_(n)  Driver_USART##n
#define  USART_Driver_(n) _USART_Driver_(n)
extern ARM_DRIVER_USART    USART_Driver_(DAP_UART_DRIVER);
#define pUSART           (&USART_Driver_(DAP_UART_DRIVER))

// UART Configuration
static uint8_t  UartTransport = DAP_UART_TRANSPORT_USB_COM_PORT;
static uint8_t  UartControl   = 0U;
static uint32_t UartBaudrate  = 115200U;

// UART TX Buffer
static uint8_t UartTxBuf[UART_TX_BUF_SIZE];
static volatile uint32_t UartTxIndexI = 0U;
static volatile uint32_t UartTxIndexO = 0U;

// UART RX Buffer
static uint8_t UartRxBuf[UART_RX_BUF_SIZE];
static volatile uint32_t UartRxIndexI = 0U;
static volatile uint32_t UartRxIndexO = 0U;

// Uart Errors
static volatile uint8_t UartRxDataLost   = 0U;
static volatile uint8_t UartFramingError = 0U;
static volatile uint8_t UartParityError  = 0U;

// UART Transmit
static uint32_t UartTxNum = 0U;

// Function prototypes
static uint32_t UART_Start   (void);
static uint32_t UART_Stop    (void);
static void     UART_Send    (void);
static void     UART_Receive (void);


// USART Driver Callback function
//   event: event mask
static void USART_Callback (uint32_t event) {
  if (event &  ARM_USART_EVENT_SEND_COMPLETE) {
    UartTxIndexO += UartTxNum;
    UART_Send();
  }
  if (event &  ARM_USART_EVENT_RECEIVE_COMPLETE) {
    UartRxIndexI += UART_RX_BLOCK_SIZE;
    UART_Receive();
  }
  if (event &  ARM_USART_EVENT_RX_OVERFLOW) {
    UartRxDataLost = 1U;
  }
  if (event &  ARM_USART_EVENT_RX_FRAMING_ERROR) {
    UartFramingError = 1U;
  }
  if (event &  ARM_USART_EVENT_RX_PARITY_ERROR) {
    UartParityError = 1U;
  }
}

// Start UART
//   return: 1 - Success, 0 - Error
static uint32_t UART_Start (void) {
  int32_t status;
  uint32_t ret;

  UartTxNum = 0U;

  UartTxIndexI = 0U;
  UartTxIndexO = 0U;
  UartRxIndexI = 0U;
  UartRxIndexO = 0U;

  status = pUSART->Initialize(USART_Callback);

  if (status == ARM_DRIVER_OK) {
    status = pUSART->PowerControl(ARM_POWER_FULL);
  }

  if (status == ARM_DRIVER_OK) {
    status = pUSART->Control (UartControl |
                              ARM_USART_MODE_ASYNCHRONOUS |
                              ARM_USART_FLOW_CONTROL_NONE,
                              UartBaudrate);
  }

  if (status == ARM_DRIVER_OK) {
    UART_Receive();
    pUSART->Control (ARM_USART_CONTROL_TX, 1);
    pUSART->Control (ARM_USART_CONTROL_RX, 1);
  }

  if (status != ARM_DRIVER_OK) {
    pUSART->PowerControl(ARM_POWER_OFF);
    pUSART->Uninitialize();
  }

  if (status == ARM_DRIVER_OK) {
    ret = 1U;
  } else {
    ret = 0U;
  }

  return (ret);
}

// Stop UART
//   return: 1 - Success, 0 - Error
static uint32_t UART_Stop (void) {
  UartTxIndexI = 0U;
  UartTxIndexO = 0U;
  UartRxIndexI = 0U;
  UartRxIndexO = 0U;

  pUSART->Control(ARM_USART_ABORT_RECEIVE, 0U);
  pUSART->Control(ARM_USART_ABORT_SEND, 0U);
  pUSART->PowerControl(ARM_POWER_OFF);
  pUSART->Uninitialize();

  return (1U);
}

// Send available data to target via UART
static void UART_Send (void) {
  uint32_t count;
  uint32_t index;

  count = UartTxIndexI - UartTxIndexO;
  index = UartTxIndexO & (UART_TX_BUF_SIZE - 1);

  if (count != 0U) {
    if ((index + count) < UART_TX_BUF_SIZE) {
      UartTxNum = count;
    } else {
      UartTxNum = UART_TX_BUF_SIZE - index;
    }
    pUSART->Send(&UartTxBuf[index], UartTxNum);
  }
}

// Receive data from target via UART
static void UART_Receive (void) {
  uint16_t num;
  uint32_t count;
  uint32_t index;

  count = UartRxIndexI - UartRxIndexO;
  index = UartRxIndexI & (UART_RX_BUF_SIZE - 1);
  num   =  UART_RX_BLOCK_SIZE;

  if ((UART_RX_BUF_SIZE - count) >= num) {
    pUSART->Receive(&UartRxBuf[index], num);
  }
}

// Process UART Transport command and prepare response
//   request:  pointer to request data
//   response: pointer to response data
//   return:   number of bytes in response (lower 16 bits)
//             number of bytes in request (upper 16 bits)
uint32_t UART_Transport (const uint8_t *request, uint8_t *response) {
  uint8_t  transport;
  uint32_t result = 0U;

  transport = *request;
  switch (transport) {
    case DAP_UART_TRANSPORT_USB_COM_PORT:
#if (DAP_UART_USB_COM_PORT != 0U)
      if (UartTransport == DAP_UART_TRANSPORT_DAP_COMMAND) {
        result = UART_Stop();
        USB_COM_PORT_Activate(1U);
      } else {
        result = 1U;
      }
      UartTransport = transport;
#endif
      break;
    case DAP_UART_TRANSPORT_DAP_COMMAND:
      if (UartTransport != DAP_UART_TRANSPORT_DAP_COMMAND) {
#if (DAP_UART_USB_COM_PORT != 0U)
        USB_COM_PORT_Activate(0U);
#endif
        result = UART_Start();
      } else {
        result = 1U;
      }
      UartTransport = transport;
      break;
    default:
      result = 0U;
      break;
  }

  if (result != 0U) {
    *response = DAP_OK;
  } else {
    *response = DAP_ERROR;
  }

  return ((1U << 16) | 1U);
}

// Process UART Configure command and prepare response
//   request:  pointer to request data
//   response: pointer to response data
//   return:   number of bytes in response (lower 16 bits)
//             number of bytes in request (upper 16 bits)
uint32_t UART_Configure (const uint8_t *request, uint8_t *response) {
  uint8_t  control, status;
  uint32_t baudrate;
  int32_t  result;

  status   = 0U;
  control  = *request;
  baudrate = (uint32_t)(*(request+1) <<  0) |
             (uint32_t)(*(request+2) <<  8) |
             (uint32_t)(*(request+3) << 16) |
             (uint32_t)(*(request+4) << 24);

  if (UartTransport == DAP_UART_TRANSPORT_DAP_COMMAND) {
    result = pUSART->Control (control |
                              ARM_USART_MODE_ASYNCHRONOUS |
                              ARM_USART_FLOW_CONTROL_NONE,
                              baudrate);
    switch (result) {
      case ARM_USART_ERROR_BAUDRATE:
        status = 0U;
        baudrate = 0U;
        break;
      case ARM_USART_ERROR_DATA_BITS:
        status = (1U << 0);
        break;
      case ARM_USART_ERROR_STOP_BITS:
        status = (1U << 1);
        break;
      case ARM_USART_ERROR_PARITY:
        status = (1U << 2);
        break;
    }
  }

  if ((status == 0U) && (baudrate != 0U)) {
    UartControl = control;
    UartBaudrate = baudrate;
  }

  *response++ = status;
  *response++ = (uint8_t)(baudrate >>  0);
  *response++ = (uint8_t)(baudrate >>  8);
  *response++ = (uint8_t)(baudrate >> 16);
  *response   = (uint8_t)(baudrate >> 24);

  return ((5U << 16) | 5U);
}

// Process UART Transfer command and prepare response
//   request:  pointer to request data
//   response: pointer to response data
//   return:   number of bytes in response (lower 16 bits)
//             number of bytes in request (upper 16 bits)
uint32_t UART_Transfer (const uint8_t *request, uint8_t *response) {
  uint16_t status = 0U;
  uint32_t count, index;
  uint32_t tx_num, rx_num, num;
  uint8_t * data;

  if (UartTransport != DAP_UART_TRANSPORT_DAP_COMMAND) {
    return (0U);
  }

  // TX Data
  tx_num = ((uint16_t)(*(request+0) <<  0)  |
            (uint16_t)(*(request+1) <<  8));
  data = (uint8_t *)((uint32_t)request) + 2;

  if (tx_num > UART_MAX_XFER_NUM) {
    tx_num = UART_MAX_XFER_NUM;
  }

  count = UartTxIndexI - UartTxIndexO;
  index = UartTxIndexI & (UART_TX_BUF_SIZE - 1);

  if ((UART_TX_BUF_SIZE - count) >= tx_num) {
    if ((index + tx_num) < UART_TX_BUF_SIZE) {
      memcpy(&UartTxBuf[index], data, tx_num);
    } else {
      num = UART_TX_BUF_SIZE - index;
      memcpy(&UartTxBuf[index], data, num);
      memcpy(&UartTxBuf[0], data + num, tx_num - num);
    }
    UartTxIndexI += tx_num;

    if (pUSART->GetStatus().tx_busy == 0U) {
      UART_Send();
    }
  } else {
    // Tx Data lost
    status |= DAP_UART_TRANSFER_TX_DATA_LOST;
  }
  if ((UART_TX_BUF_SIZE - count) < UART_MAX_XFER_NUM) {
    // Can't accept next full TX packet
    status |= DAP_UART_TRANSFER_TX_BUSY;
  }

  // RX Data
  rx_num  = UartRxIndexI - UartRxIndexO;
  rx_num += pUSART->GetRxCount();
  data = response + 2;
  index = UartRxIndexO & (UART_RX_BUF_SIZE - 1);

  if (rx_num > UART_MAX_XFER_NUM) {
    rx_num = UART_MAX_XFER_NUM;
  }
  if ((index + rx_num) < UART_RX_BUF_SIZE) {
    memcpy(data, &UartRxBuf[index], rx_num);
  } else {
    num = UART_RX_BUF_SIZE - index;
    memcpy(data, &UartRxBuf[index], num);
    memcpy(data + num, &UartRxBuf[0], rx_num - num);
  }
  UartRxIndexO += rx_num;

  if (pUSART->GetStatus().rx_busy == 0U) {
    UART_Receive();
  }

  if (UartRxDataLost == 1U) {
    UartRxDataLost = 0U;
    status |= DAP_UART_TRANSFER_RX_DATA_LOST;
  }
  if (UartFramingError == 1U) {
    UartFramingError = 0U;
    status |= DAP_UART_TRANSFER_FRAMING_ERROR;
  }
  if (UartParityError == 1U) {
    UartParityError = 0U;
    status |= DAP_UART_TRANSFER_PARITY_ERROR;
  }

  status |= rx_num;

  *response++ = (uint8_t)(status >> 0);
  *response++ = (uint8_t)(status >> 8);

  return (((2U + tx_num) << 16) | (2U + rx_num));
}

#endif /* DAP_UART */
