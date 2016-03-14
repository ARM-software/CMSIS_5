/*
 * Copyright (c) 2015, Infineon Technologies AG
 * All rights reserved.                        
 *                                             
 * Redistribution and use in source and binary forms, with or without modification,are permitted provided that the 
 * following conditions are met:   
 *                                                                              
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
 * disclaimer.                        
 * 
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
 * disclaimer in the documentation and/or other materials provided with the distribution.                       
 * 
 * Neither the name of the copyright holders nor the names of its contributors may be used to endorse or promote 
 * products derived from this software without specific prior written permission.                                           
 *                                                                              
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE  
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE  FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR  
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                                  
 *                                                                              
 * To improve the quality of the software, users are encouraged to share modifications, enhancements or bug fixes with 
 * Infineon Technologies AG dave@infineon.com).                                                          
 *
 */

/**
 * @file UART.c
 * @date 24 July, 2015
 * @version 2.0.4
 *
 * @brief UART Driver for Infineon XMC4000
 *
 * History
 *
 * Version 2.0.4 Added fixed to prevent race conditions 
 *               and Initialize/Uninitialize, Power Control 
 *               guidelines related modifications <br>
 *
 * Version 1.0.0 Initial version<br>
 */

#include "UART.h"
#include "RTE_Device.h"
#include "RTE_Components.h"

#define ARM_USART_DRV_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(2,4)   /* driver version */

// Driver Version
static const ARM_DRIVER_VERSION DriverVersion = {
  ARM_USART_API_VERSION,
  ARM_USART_DRV_VERSION
};


#if (defined(RTE_Drivers_USART) && \
     (RTE_UART0 == 0) && \
     (RTE_UART1 == 0) && \
     (RTE_UART2 == 0) && \
     (RTE_UART3 == 0) && \
     (RTE_UART4 == 0) && \
     (RTE_UART5 == 0))
#error "UART not configured in RTE_Device.h!"
#endif

// Default UART initialization
static const XMC_UART_CH_CONFIG_t uart_default_config =
{
.baudrate = 100000U,
.data_bits =	8U,
.frame_length	= 8U,
.stop_bits = 1U,
.parity_mode = XMC_USIC_CH_PARITY_MODE_NONE
};


static const ARM_USART_CAPABILITIES DriverCapabilities =
{
  1,      ///< supports UART (Asynchronous) mode 
  0,      ///< supports Synchronous Master mode
  0,      ///< supports Synchronous Slave mode
  1,      ///< supports UART Single-wire mode
  0,      ///< supports UART IrDA mode
  0,      ///< supports UART Smart Card mode
  0,      ///< Smart Card Clock generator available
  0,      ///< RTS Flow Control available
  0,      ///< CTS Flow Control available
  1,      ///< Transmit completed event: \ref ARM_USART_EVENT_TX_COMPLETE
  0,      ///< Signal receive character timeout event: \ref ARM_USART_EVENT_RX_TIMEOUT
  0,      ///< RTS Line: 0=not available, 1=available
  0,      ///< CTS Line: 0=not available, 1=available
  0,      ///< DTR Line: 0=not available, 1=available
  0,      ///< DSR Line: 0=not available, 1=available
  0,      ///< DCD Line: 0=not available, 1=available
  0,      ///< RI Line: 0=not available, 1=available
  0,      ///< Signal CTS change event: \ref ARM_USART_EVENT_CTS
  0,      ///< Signal DSR change event: \ref ARM_USART_EVENT_DSR
  0,      ///< Signal DCD change event: \ref ARM_USART_EVENT_DCD
  0      ///< Signal RI change event: \ref ARM_USART_EVENT_RI
};

/* UART0 */
#if (RTE_UART0 != 0)

static UART_INFO UART0_Info = {0};
static XMC_GPIO_CONFIG_t UART0_rx_conf; 
static XMC_GPIO_CONFIG_t UART0_tx_conf; 


/* UART0 Resources */
UART_RESOURCES UART0_Resources = {
  RTE_UART0_TX_PORT,
  &UART0_tx_conf,
	RTE_UART0_TX_AF,
  RTE_UART0_RX_PORT,
  &UART0_rx_conf,
	RTE_UART0_RX_INPUT,
  XMC_UART0_CH0,
	USIC0_0_IRQn,
	USIC0_1_IRQn,
	RTE_UART0_TX_FIFO_SIZE,
	RTE_UART0_TX_FIFO_SIZE_NUM,
  RTE_UART0_RX_FIFO_SIZE,
	RTE_UART0_RX_FIFO_SIZE_NUM,
	&UART0_Info
};
#endif

/* UART1 */
#if (RTE_UART1 != 0)

static UART_INFO UART1_Info = {0};
static XMC_GPIO_CONFIG_t UART1_rx_conf; 
static XMC_GPIO_CONFIG_t UART1_tx_conf; 

/* UART1 Resources */
UART_RESOURCES UART1_Resources = {
  RTE_UART1_TX_PORT,
  &UART1_tx_conf,
	RTE_UART1_TX_AF,
  RTE_UART1_RX_PORT,
  &UART1_rx_conf,
  RTE_UART1_RX_INPUT,
  XMC_UART0_CH1,
	USIC0_2_IRQn,
	USIC0_3_IRQn,
  RTE_UART1_TX_FIFO_SIZE,
	RTE_UART1_TX_FIFO_SIZE_NUM,
  RTE_UART1_RX_FIFO_SIZE,
	RTE_UART1_RX_FIFO_SIZE_NUM,
	&UART1_Info
};
#endif

/* UART2 */
#if (RTE_UART2 != 0)

static UART_INFO UART2_Info = {0};
static XMC_GPIO_CONFIG_t UART2_rx_conf; 
static XMC_GPIO_CONFIG_t UART2_tx_conf; 

/* UART2 Resources */
UART_RESOURCES UART2_Resources = {
  RTE_UART2_TX_PORT,
  &UART2_tx_conf,
	RTE_UART2_TX_AF,
  RTE_UART2_RX_PORT,
  &UART2_rx_conf,
  RTE_UART2_RX_INPUT,
  XMC_UART1_CH0,
	USIC1_0_IRQn,
	USIC1_1_IRQn,
  RTE_UART2_TX_FIFO_SIZE,
	RTE_UART2_TX_FIFO_SIZE_NUM,
  RTE_UART2_RX_FIFO_SIZE,
	RTE_UART2_RX_FIFO_SIZE_NUM,
	&UART2_Info
};
#endif


/* UART3 */
#if (RTE_UART3 != 0)

static UART_INFO UART3_Info = {0};
static XMC_GPIO_CONFIG_t UART3_rx_conf; 
static XMC_GPIO_CONFIG_t UART3_tx_conf; 


/* UART3 Resources */
UART_RESOURCES UART3_Resources = {
  RTE_UART3_TX_PORT,
  &UART3_tx_conf,
	RTE_UART3_TX_AF,
  RTE_UART3_RX_PORT,
  &UART3_rx_conf,
  RTE_UART3_RX_INPUT,
  XMC_UART1_CH1,
	USIC1_2_IRQn,
	USIC1_3_IRQn,
  RTE_UART3_TX_FIFO_SIZE,
	RTE_UART3_TX_FIFO_SIZE_NUM,
  RTE_UART3_RX_FIFO_SIZE,
	RTE_UART3_RX_FIFO_SIZE_NUM,
	&UART3_Info
};
#endif


/* UART4 */
#if (RTE_UART4 != 0)

static UART_INFO UART4_Info = {0};
static XMC_GPIO_CONFIG_t UART4_rx_conf; 
static XMC_GPIO_CONFIG_t UART4_tx_conf; 

/* UART4 Resources */
UART_RESOURCES UART4_Resources = {
  RTE_UART4_TX_PORT,
  &UART4_tx_conf,
	RTE_UART4_TX_AF,
  RTE_UART4_RX_PORT,
  &UART4_rx_conf,
	RTE_UART4_RX_INPUT,
  XMC_UART2_CH0,
	USIC2_0_IRQn,
	USIC2_1_IRQn,
  RTE_UART4_TX_FIFO_SIZE,
	RTE_UART4_TX_FIFO_SIZE_NUM,
  RTE_UART4_RX_FIFO_SIZE,
	RTE_UART4_RX_FIFO_SIZE_NUM,
	&UART4_Info
};
#endif


/* UART5 */
#if (RTE_UART5 != 0)


static UART_INFO UART5_Info = {0};
static XMC_GPIO_CONFIG_t UART5_rx_conf; 
static XMC_GPIO_CONFIG_t UART5_tx_conf; 

/* UART5 Resources */
UART_RESOURCES UART5_Resources = {
  RTE_UART5_TX_PORT,
  &UART5_tx_conf,
	RTE_UART5_TX_AF,
  RTE_UART5_RX_PORT,
  &UART5_rx_conf,
	RTE_UART5_RX_INPUT,
  XMC_UART2_CH1,
	USIC2_2_IRQn,
	USIC2_3_IRQn,
  RTE_UART5_TX_FIFO_SIZE,
	RTE_UART5_TX_FIFO_SIZE_NUM,
  RTE_UART5_RX_FIFO_SIZE,
	RTE_UART5_RX_FIFO_SIZE_NUM,
	&UART5_Info
};

#endif


/* UART Resources */
static UART_RESOURCES  *uart[6] = {
#if (RTE_UART0 != 0)
  &UART0_Resources,
#else
  NULL,
#endif
#if (RTE_UART1 != 0)
  &UART1_Resources,
#else
  NULL,
#endif
#if (RTE_UART2 != 0)
  &UART2_Resources,
#else
  NULL,
#endif
#if (RTE_UART3 != 0)
  &UART3_Resources,
#else
  NULL,
#endif
#if (RTE_UART4 != 0)
  &UART4_Resources,
#else
  NULL,
#endif
#if (RTE_UART5 != 0)
  &UART5_Resources,
#else
  NULL,
#endif
};

/**
  \fn          ARM_DRV_VERSION USARTX_GetVersion (void)
  \brief       Get driver version.
  \return      \ref ARM_DRV_VERSION
*/
static ARM_DRIVER_VERSION USARTX_GetVersion (void) {
 return DriverVersion;
}


/**
  \fn          ARM_UART_CAPABILITIES UART_GetCapabilities (UART_RESOURCES  *uart)
  \brief       Get driver capabilities.
  \param[in]   uart    Pointer to UART resources
  \return      \ref USART_CAPABILITIES
*/
static ARM_USART_CAPABILITIES UART_GetCapabilities (UART_RESOURCES  *uart) {
  return  DriverCapabilities;
}

#if (RTE_UART0 != 0)
  static ARM_USART_CAPABILITIES UART0_GetCapabilities (void) {
    return UART_GetCapabilities (uart[0]);
  }
#endif

#if (RTE_UART1 != 0)
  static ARM_USART_CAPABILITIES UART1_GetCapabilities (void) {
    return UART_GetCapabilities (uart[1]);
  }
#endif

#if (RTE_UART2 != 0)
  static ARM_USART_CAPABILITIES UART2_GetCapabilities (void) {
    return UART_GetCapabilities (uart[2]);
  }
#endif

#if (RTE_UART3 != 0)
  static ARM_USART_CAPABILITIES UART3_GetCapabilities (void) {
    return UART_GetCapabilities (uart[3]);
  }
#endif

#if (RTE_UART4 != 0)
  static ARM_USART_CAPABILITIES UART4_GetCapabilities (void) {
    return UART_GetCapabilities (uart[4]);
  }
#endif

#if (RTE_UART5 != 0)
  static ARM_USART_CAPABILITIES UART5_GetCapabilities (void) {
    return UART_GetCapabilities (uart[5]);
  }
#endif

/**
  \fn          ARM_USART_STATUS UART_Initialize (ARM_USART_SignalEvent_t cb_event,
	                                               UART_RESOURCES *uart)
  \brief       Initialize UART Interface.
  \param[in]   cb_event Pointer to \ref ARM_USART_SignalEvent
  \param[in]   uart Pointer to USART resources
  \return      \ref ARM_USART_STATUS
*/
static int32_t UART_Initialize (ARM_USART_SignalEvent_t cb_event, UART_RESOURCES *uart) {
	if(((uart->info->flags)&UART_INITIALIZED) == 0)
	{
  // Initialize USART Run-Time Resources
  uart->info->cb_event= cb_event;

  uart->info->status.tx_busy          = 0;
  uart->info->status.rx_busy          = 0;
  uart->info->status.tx_underflow     = 0;
  uart->info->status.rx_overflow      = 0;
  uart->info->status.rx_break         = 0;
  uart->info->status.rx_framing_error = 0;
  uart->info->status.rx_parity_error  = 0;

  uart->info->flags |= UART_INITIALIZED;
	return ARM_DRIVER_OK;
	}else  return ARM_DRIVER_OK;
}

#if (RTE_UART0 != 0)
  static int32_t UART0_Initialize (ARM_USART_SignalEvent_t cb_event) {
    return UART_Initialize (cb_event, uart[0]);
  }
#endif

#if (RTE_UART1 != 0)
  static int32_t UART1_Initialize (ARM_USART_SignalEvent_t cb_event) {
    return UART_Initialize (cb_event, uart[1]);
  }
#endif

#if (RTE_UART2 != 0)
  static int32_t UART2_Initialize (ARM_USART_SignalEvent_t cb_event) {
    return UART_Initialize (cb_event, uart[2]);
  }
#endif

#if (RTE_UART3 != 0)
  static int32_t UART3_Initialize (ARM_USART_SignalEvent_t cb_event) {
    return UART_Initialize (cb_event, uart[3]);
  }
#endif

  #if (RTE_UART4 != 0)
  static int32_t UART4_Initialize (ARM_USART_SignalEvent_t cb_event) {
    return UART_Initialize (cb_event, uart[4]);
  }
#endif

#if (RTE_UART5 != 0)
  static int32_t UART5_Initialize (ARM_USART_SignalEvent_t cb_event) {
    return UART_Initialize (cb_event, uart[5]);
  }
#endif


	/**
  \fn          int32_t UART_Uninitialize (UART_RESOURCES *uart)
  \brief       De-initialize USART Interface.
  \param[in]   uart  Pointer to USART resources
  \return      \ref execution_status
*/
static int32_t UART_Uninitialize (UART_RESOURCES *uart) {
	
  // Reset UART status flags
  uart->info->flags = 0;
	uart->info->flags &=~UART_INITIALIZED;
  return ARM_DRIVER_OK;
}
#if (RTE_UART0 != 0)
  static int32_t UART0_Uninitialize (void) {
    return UART_Uninitialize (uart[0]);
  }
#endif	
#if (RTE_UART1 != 0)
  static int32_t UART1_Uninitialize (void) {
    return UART_Uninitialize (uart[1]);
  }
#endif
	#if (RTE_UART2 != 0)
  static int32_t UART2_Uninitialize (void) {
    return UART_Uninitialize (uart[2]);
  }
#endif
	#if (RTE_UART3 != 0)
  static int32_t UART3_Uninitialize (void) {
    return UART_Uninitialize (uart[3]);
  }
#endif
	#if (RTE_UART4 != 0)
  static int32_t UART4_Uninitialize (void) {
    return UART_Uninitialize (uart[4]);
  }
#endif
	#if (RTE_UART5 != 0)
  static int32_t UART5_Uninitialize (void) {
    return UART_Uninitialize (uart[5]);
  }
#endif
/**
  \fn          ARM_USART_STATUS UART_PowerControl (ARM_POWER_STATE  state,
                                                   UART_RESOURCES   *uart)
  \brief       Controls UART Interface Power.
  \param[in]   state    Power state
  \param[in]   uart Pointer to USART resources
  \return      \ref USART_STATUS
*/
static int32_t UART_PowerControl (ARM_POWER_STATE state, UART_RESOURCES  *uart) {
	if(state == ARM_POWER_FULL) 
		{
	     if(((uart->info->flags)&UART_POWERED) == 0)
	     {
	       if(uart->tx_fifo_size_num > uart->rx_fifo_size_num)
	       {
	          uart->info->tx_fifo_pointer = 0;
		        uart->info->rx_fifo_pointer = uart->tx_fifo_size_num;
	       }
	       else
	       {
		       uart->info->tx_fifo_pointer = uart->rx_fifo_size_num;
		       uart->info->rx_fifo_pointer = 0;
	       }

          XMC_UART_CH_Init(uart->uart,&uart_default_config);	 
          XMC_USIC_CH_TXFIFO_Configure(uart->uart,uart->info->tx_fifo_pointer,(XMC_USIC_CH_FIFO_SIZE_t)uart->tx_fifo_size_reg,1U); 
	        XMC_UART_CH_SetInputSource(uart->uart,XMC_UART_CH_INPUT_RXD,uart->input);
          NVIC_ClearPendingIRQ(uart->irq_rx_num);
	        NVIC_ClearPendingIRQ(uart->irq_tx_num);
#if(UC_FAMILY == XMC4)
          NVIC_SetPriority(uart->irq_rx_num,NVIC_EncodePriority(NVIC_GetPriorityGrouping(),0U,0U)); 
		      NVIC_SetPriority(uart->irq_tx_num,NVIC_EncodePriority(NVIC_GetPriorityGrouping(),0U,0U)); 
#else
	        NVIC_SetPriority(uart->irq_rx_num,3U); 
	        NVIC_SetPriority(uart->irq_tx_num,3U); 	
#endif
          NVIC_EnableIRQ(uart->irq_rx_num);
          NVIC_EnableIRQ(uart->irq_tx_num);	
	 
	        uart->info->flags |= UART_POWERED;
			}
    } 
    else if(state == ARM_POWER_OFF ) 
    {
      XMC_UART_CH_Stop(uart->uart);	
		  uart->info->flags &=~UART_POWERED;
    } else return ARM_DRIVER_ERROR_UNSUPPORTED;

  return ARM_DRIVER_OK;
}
#if (RTE_UART0 != 0)
static int32_t UART0_PowerControl (ARM_POWER_STATE state) {
  return UART_PowerControl (state, uart[0]);
}
#endif
#if (RTE_UART1 != 0)
static int32_t UART1_PowerControl (ARM_POWER_STATE state) {
  return UART_PowerControl (state, uart[1]);
}
#endif
#if (RTE_UART2 != 0)
static int32_t UART2_PowerControl (ARM_POWER_STATE state) {
  return UART_PowerControl (state, uart[2]);
}
#endif
#if (RTE_UART3 != 0)
static int32_t UART3_PowerControl (ARM_POWER_STATE state) {
  return UART_PowerControl (state, uart[3]);
}
#endif
#if (RTE_UART4 != 0)
static int32_t UART4_PowerControl (ARM_POWER_STATE state) {
  return UART_PowerControl (state, uart[4]);
}
#endif
#if (RTE_UART5 != 0)
static int32_t UART5_PowerControl (ARM_POWER_STATE state) {
  return UART_PowerControl (state, uart[5]);
}
#endif
/**
  \fn          int32_t UART_Send (const uint8_t   *data,
                                  uint32_t num,
                                  UART_RESOURCES   *uart)
  \brief       Write data to UART transmitter.
  \param[in]   data  Pointer to buffer with data to write to UART transmitter
  \param[in]   num  Data buffer size in bytes
  \param[in]   uart Pointer to UART resources
  \return      driver status
*/
static int32_t UART_Send (const void *data, uint32_t num, UART_RESOURCES  *uart){

  uint8_t SRno = 0U;
  
  if ((data == NULL) || (num == 0)) {
    // Invalid parameters
    return ARM_DRIVER_ERROR_PARAMETER;
  }

  if ((uart->info->flags & UART_POWERED) == 0) {
    // UART is not powered 
    return ARM_DRIVER_ERROR;
  }

  if (uart->info->status.tx_busy == 1) {
    // Send is not completed yet
    return ARM_DRIVER_ERROR_BUSY;
  }

  uart->info->status.tx_busy = 1;
  // Save transmit buffer info
  uart->info->xfer.tx_buf = (uint8_t *)data;
  uart->info->xfer.tx_num = num;
  uart->info->xfer.tx_cnt = 0;
#if(UC_FAMILY == XMC4)
  if((uart->irq_tx_num) < 90)
  { 
	  SRno= uart->irq_tx_num - 84;
  }
  else if((uart->irq_tx_num) < 96)
  {
   SRno=uart->irq_tx_num - 90;
  }
  else
  {
   SRno=uart->irq_tx_num - 96;
  }
#elif (UC_FAMILY == XMC1)
  SRno=uart->irq_tx_num - 9;
#endif
	
  if(uart->rx_fifo_size_reg==NO_FIFO)
  {
    XMC_USIC_CH_EnableEvent(uart->uart,XMC_USIC_CH_EVENT_TRANSMIT_BUFFER);
		XMC_USIC_CH_SetInterruptNodePointer(uart->uart,XMC_USIC_CH_INTERRUPT_NODE_POINTER_TRANSMIT_BUFFER,SRno);
  }
	else
	{
    XMC_USIC_CH_TXFIFO_Flush(uart->uart); 
    XMC_USIC_CH_TXFIFO_EnableEvent(uart->uart,XMC_USIC_CH_TXFIFO_EVENT_CONF_STANDARD); 
    XMC_USIC_CH_TXFIFO_SetInterruptNodePointer(uart->uart,XMC_USIC_CH_TXFIFO_INTERRUPT_NODE_POINTER_STANDARD,SRno);											 
	}
  /* Trigger standard tranmit interrupt */
  XMC_USIC_CH_TriggerServiceRequest(uart->uart,SRno);	

 return ARM_DRIVER_OK;
}

#if (RTE_UART0 != 0)
  static int32_t UART0_Send (const void *data, uint32_t num) {
    return UART_Send (data, num, uart[0]);
  }
#endif

#if (RTE_UART1 != 0)
  static int32_t UART1_Send (const void *data, uint32_t num) {
    return UART_Send (data, num, uart[1]);
  }
#endif

#if (RTE_UART2 != 0)
  static int32_t UART2_Send (const void *data, uint32_t num) {
    return UART_Send (data, num, uart[2]);
  }
#endif

#if (RTE_UART3 != 0)
  static int32_t UART3_Send (const void *data, uint32_t num) {
    return UART_Send (data, num, uart[3]);
  }
#endif

#if (RTE_UART4 != 0)
  static int32_t UART4_Send (const void *data, uint32_t num) {
    return UART_Send (data, num, uart[4]);
  }
#endif

#if (RTE_UART5 != 0)
  static int32_t UART5_Send (const void *data, uint32_t num) {
    return UART_Send (data, num, uart[5]);
  }
#endif


/**
  \fn          int32_t UART_Receive (uint8_t *data, uint32_t num,
                                      UART_RESOURCES *uart)
  \brief       Read data from UART receiver.
  \param[out]  data  Pointer to buffer for data read from USART receiver
  \param[in]   size  Data buffer size in bytes
  \param[in]   uart Pointer to UART resources
  \return      driver status
*/
static int32_t UART_Receive (void *data, uint32_t num,UART_RESOURCES *uart) {
																														
	 uint8_t SRno = 0U;																
  if ((data == NULL) || (num == 0)) {
    // Invalid parameters
    return ARM_DRIVER_ERROR_PARAMETER;
  }

  if ((uart->info->flags & UART_POWERED) == 0) {
    // UART is not powered
    return ARM_DRIVER_ERROR;
  }

  // Check if receiver is busy
  if (uart->info->status.rx_busy == 1) 
    return ARM_DRIVER_ERROR_BUSY;
#if (UC_FAMILY == XMC4) 	
  if((uart->irq_rx_num) < 90)
  { 
	  SRno= uart->irq_rx_num - 84;
  }
  else if((uart->irq_rx_num) < 96)
  {
    SRno=uart->irq_rx_num - 90;
  }
  else
  {
    SRno=uart->irq_rx_num - 96;
  }
#elif (UC_FAMILY == XMC1)
  SRno=uart->irq_rx_num - 9;
#endif
	
  // Set RX busy flag
  uart->info->status.rx_busy = 1;
	

  // Save number of data to be received
  uart->info->xfer.rx_num = num;

  // Clear RX status
  uart->info->status.rx_break          = 0;
  uart->info->status.rx_framing_error  = 0;
  uart->info->status.rx_overflow       = 0;
  uart->info->status.rx_parity_error   = 0;

  // Save receive buffer info
  uart->info->xfer.rx_buf = (uint8_t *)data;
  uart->info->xfer.rx_cnt = 0;														
	
	if(uart->rx_fifo_size_reg==NO_FIFO)
  {
    XMC_USIC_CH_EnableEvent(uart->uart,XMC_USIC_CH_EVENT_STANDARD_RECEIVE |
		                                   XMC_USIC_CH_EVENT_ALTERNATIVE_RECEIVE);
		XMC_USIC_CH_SetInterruptNodePointer(uart->uart,XMC_USIC_CH_INTERRUPT_NODE_POINTER_RECEIVE,SRno);
		XMC_USIC_CH_SetInterruptNodePointer(uart->uart,XMC_USIC_CH_INTERRUPT_NODE_POINTER_ALTERNATE_RECEIVE,SRno);
  }
	else
	{
    XMC_USIC_CH_RXFIFO_Flush(uart->uart); 		
    XMC_USIC_CH_RXFIFO_EnableEvent(uart->uart,XMC_USIC_CH_RXFIFO_EVENT_CONF_STANDARD | 
		                                          XMC_USIC_CH_RXFIFO_EVENT_CONF_ALTERNATE);
    XMC_USIC_CH_RXFIFO_SetInterruptNodePointer(uart->uart,XMC_USIC_CH_RXFIFO_INTERRUPT_NODE_POINTER_STANDARD,SRno);
    XMC_USIC_CH_RXFIFO_SetInterruptNodePointer(uart->uart,XMC_USIC_CH_RXFIFO_INTERRUPT_NODE_POINTER_ALTERNATE,SRno);
	if(num <= uart->rx_fifo_size_num)
	{
	  XMC_USIC_CH_RXFIFO_Configure(uart->uart,uart->info->rx_fifo_pointer,(XMC_USIC_CH_FIFO_SIZE_t)uart->rx_fifo_size_reg,(uart->info->xfer.rx_num)- 1U); 
	}
	else
	{
		XMC_USIC_CH_RXFIFO_Configure(uart->uart,uart->info->rx_fifo_pointer,(XMC_USIC_CH_FIFO_SIZE_t)uart->rx_fifo_size_reg,uart->rx_fifo_size_num - 1U); 
	}			
 }
 return ARM_DRIVER_OK;
}

#if (RTE_UART0 != 0)
  static int32_t UART0_Receive (void *data, uint32_t num) {
    return UART_Receive (data, num, uart[0]);
  }
#endif

#if (RTE_UART1 != 0)
  static int32_t UART1_Receive (void *data, uint32_t num) {
    return UART_Receive (data, num, uart[1]);
  }
#endif

#if (RTE_UART2 != 0)
  static int32_t UART2_Receive (void *data, uint32_t num) {
    return UART_Receive (data, num, uart[2]);
  }
#endif

#if (RTE_UART3 != 0)
  static int32_t UART3_Receive (void *data, uint32_t num) {
    return UART_Receive (data, num, uart[3]);
  }
#endif

#if (RTE_UART4 != 0)
  static int32_t UART4_Receive (void *data, uint32_t num) {
    return UART_Receive (data, num, uart[4]);
  }
#endif

#if (RTE_UART5 != 0)
  static int32_t UART5_Receive (void *data, uint32_t num) {
    return UART_Receive (data, num, uart[5]);
  }
#endif

/**
  \fn          int32_t UART_Transfer (const void *data_out,
                                             void *data_in,
                                             uint32_t num,
                                             UART_RESOURCES  *uart)
  \brief       Start sending/receiving data to/from USART transmitter/receiver.
  \param[in]   data_out  Pointer to buffer with data to send to UART transmitter
  \param[out]  data_in   Pointer to buffer for data to receive from UART receiver
  \param[in]   num       Number of data items to transfer
  \param[in]   uart     Pointer to UART resources
  \return      \ref execution_status
*/
static int32_t UART_Transfer (const void  *data_out,
                                     void  *data_in,
                                     uint32_t  num,
                                     UART_RESOURCES  *uart) {

  return ARM_DRIVER_OK;
}
#if (RTE_UART0 != 0)
  static int32_t UART0_Transfer (const void *data_out,void  *data_in,uint32_t num) {
    return UART_Transfer (data_out, data_in, num, uart[0]);
  }
#endif
#if (RTE_UART1 != 0)
  static int32_t UART1_Transfer (const void *data_out,void  *data_in,uint32_t num) {
    return UART_Transfer (data_out, data_in, num, uart[1]);
  }
#endif
#if (RTE_UART2 != 0)
  static int32_t UART2_Transfer (const void *data_out,void  *data_in,uint32_t num) {
    return UART_Transfer (data_out, data_in, num, uart[2]);
  }
#endif
#if (RTE_UART3 != 0)
  static int32_t UART3_Transfer (const void *data_out,void  *data_in,uint32_t num) {
    return UART_Transfer (data_out, data_in, num, uart[3]);
  }
#endif
#if (RTE_UART4 != 0)
  static int32_t UART4_Transfer (const void *data_out,void  *data_in,uint32_t num) {
    return UART_Transfer (data_out, data_in, num, uart[4]);
  }
#endif
#if (RTE_UART5 != 0)
  static int32_t UART5_Transfer (const void *data_out,void  *data_in,uint32_t num) {
    return UART_Transfer (data_out, data_in, num, uart[5]);
  }
#endif
	
/**
  \fn          uint32_t UART_GetTxCount (UART_RESOURCES *uart)
  \brief       Get transmitted data count.
  \param[in]   uart     Pointer to UART resources
  \return      number of data items transmitted
*/
static uint32_t UART_GetTxCount (UART_RESOURCES *uart) {
  return uart->info->xfer.tx_cnt;
}
#if (RTE_UART0 != 0)
  static uint32_t UART0_GetTxCount (void) {
    return UART_GetTxCount (uart[0]);
  }
#endif
#if (RTE_UART1 != 0)
  static uint32_t UART1_GetTxCount (void) {
    return UART_GetTxCount (uart[1]);
  }
#endif
#if (RTE_UART2 != 0)
  static uint32_t UART2_GetTxCount (void) {
    return UART_GetTxCount (uart[2]);
  }
#endif
#if (RTE_UART3 != 0)
  static uint32_t UART3_GetTxCount (void) {
    return UART_GetTxCount (uart[3]);
  }
#endif
#if (RTE_UART4 != 0)
  static uint32_t UART4_GetTxCount (void) {
    return UART_GetTxCount (uart[4]);
  }
#endif
#if (RTE_UART5 != 0)
  static uint32_t UART5_GetTxCount (void) {
    return UART_GetTxCount (uart[5]);
  }
#endif
/**
  \fn          uint32_t UART_GetRxCount (UART_RESOURCES *uart)
  \brief       Get received data count.
  \param[in]   uart     Pointer to UART resources
  \return      number of data items received
*/
static uint32_t UART_GetRxCount (UART_RESOURCES *uart) {
  return uart->info->xfer.rx_cnt;
}
#if (RTE_UART0 != 0)
  static uint32_t UART0_GetRxCount (void) {
    return UART_GetRxCount (uart[0]);
  }
#endif
#if (RTE_UART1 != 0)
  static uint32_t UART1_GetRxCount (void) {
    return UART_GetRxCount (uart[1]);
  }
#endif
#if (RTE_UART2 != 0)
  static uint32_t UART2_GetRxCount (void) {
    return UART_GetRxCount (uart[2]);
  }
#endif
#if (RTE_UART3 != 0)
  static uint32_t UART3_GetRxCount (void) {
    return UART_GetRxCount (uart[3]);
  }
#endif
#if (RTE_UART4 != 0)
  static uint32_t UART4_GetRxCount (void) {
    return UART_GetRxCount (uart[4]);
  }
#endif
#if (RTE_UART5 != 0)
static uint32_t UART5_GetRxCount (void) {
  return UART_GetRxCount (uart[0]);
 }
#endif
/**
  \fn          ARM_USART_STATUS UART_GetStatus (UART_RESOURCES *uart)
  \brief       Get UART status.
  \param[in]   uart     Pointer to USART resources
  \return      UART status \ref ARM_USART_STATUS
*/
static ARM_USART_STATUS UART_GetStatus (UART_RESOURCES *usart) {
	ARM_USART_STATUS status;

  status.tx_busy          = usart->info->status.tx_busy;;
  status.rx_busy          = usart->info->status.rx_busy;
  status.tx_underflow     = usart->info->status.tx_underflow;
  status.rx_overflow      = usart->info->status.rx_overflow;
  status.rx_break         = usart->info->status.rx_break;
  status.rx_framing_error = usart->info->status.rx_framing_error;
  status.rx_parity_error  = usart->info->status.rx_parity_error;
  return status;
}
#if (RTE_UART0 != 0)
  static ARM_USART_STATUS UART0_GetStatus (void) {
    return UART_GetStatus(uart[0]);
  }
#endif
#if (RTE_UART1 != 0)
  static ARM_USART_STATUS UART1_GetStatus (void) {
    return UART_GetStatus(uart[1]);
  }
#endif
#if (RTE_UART2 != 0)
  static ARM_USART_STATUS UART2_GetStatus (void) {
    return UART_GetStatus(uart[2]);
  }
#endif
#if (RTE_UART3 != 0)
  static ARM_USART_STATUS UART3_GetStatus (void) {
    return UART_GetStatus(uart[3]);
  }
#endif
#if (RTE_UART4 != 0)
  static ARM_USART_STATUS UART4_GetStatus (void) {
    return UART_GetStatus(uart[4]);
  }
#endif
#if (RTE_UART5 != 0)
  static ARM_USART_STATUS UART5_GetStatus (void) {
    return UART_GetStatus(uart[5]);
  }
#endif
	
/**
  \fn          ARM_USART_STATUS UART_SetModemControl (ARM_USART_MODEM_CONTROL control,
                                                     UART_RESOURCES  *uart)
  \brief       Set USART Modem Control line state.
  \param[in]   control  \ref ARM_USART_MODEM_CONTROL
  \param[in]   uart Pointer to USART resources
  \return      \ref ARM_USART_STATUS
*/
static int32_t UART_SetModemControl (ARM_USART_MODEM_CONTROL control,
                                      UART_RESOURCES *uart) {

  return ARM_DRIVER_ERROR;
}

#if (RTE_UART0 != 0)
  static int32_t UART0_SetModemControl (ARM_USART_MODEM_CONTROL control) {
    return UART_SetModemControl (control, uart[0]);
  }
#endif

#if (RTE_UART1 != 0)
  static int32_t UART1_SetModemControl (ARM_USART_MODEM_CONTROL control) {
    return UART_SetModemControl (control, uart[1]);
  }
#endif

#if (RTE_UART2 != 0)
  static int32_t UART2_SetModemControl (ARM_USART_MODEM_CONTROL control) {
    return UART_SetModemControl (control, uart[2]);
  }
#endif

#if (RTE_UART3 != 0)
  static int32_t UART3_SetModemControl (ARM_USART_MODEM_CONTROL control) {
    return UART_SetModemControl (control, uart[3]);
  }
#endif

#if (RTE_UART4 != 0)
  static int32_t UART4_SetModemControl (ARM_USART_MODEM_CONTROL control) {
    return UART_SetModemControl (control, uart[4]);
  }
#endif

#if (RTE_UART5 != 0)
  static int32_t UART5_SetModemControl (ARM_USART_MODEM_CONTROL control) {
    return UART_SetModemControl (control, uart[5]);
  }
#endif


/**
  \fn          ARM_USART_MODEM_STATUS UART_GetModemStatus (UART_RESOURCES  *uart)
  \brief       Get USART Modem Status lines state.
  \param[in]   uart Pointer to USART resources
  \return      \ref ARM_USART_MODEM_STATUS
*/
static ARM_USART_MODEM_STATUS UART_GetModemStatus (UART_RESOURCES  *uart) {
  ARM_USART_MODEM_STATUS mst = { 0, 0, 0, 0,};
  return mst;
}

#if (RTE_UART0 != 0)
  static ARM_USART_MODEM_STATUS UART0_GetModemStatus (void) {
    return UART_GetModemStatus (uart[0]);
  }
#endif

#if (RTE_UART1 != 0)
  static ARM_USART_MODEM_STATUS UART1_GetModemStatus (void) {
    return UART_GetModemStatus (uart[1]);
  }
#endif

#if (RTE_UART2 != 0)
  static ARM_USART_MODEM_STATUS UART2_GetModemStatus (void) {
    return UART_GetModemStatus (uart[2]);
  }
#endif

#if (RTE_UART3 != 0)
  static ARM_USART_MODEM_STATUS UART3_GetModemStatus (void) {
    return UART_GetModemStatus (uart[3]);
  }
#endif

#if (RTE_UART4 != 0)
  static ARM_USART_MODEM_STATUS UART4_GetModemStatus (void) {
    return UART_GetModemStatus (uart[4]);
  }
#endif

#if (RTE_UART5 != 0)
  static ARM_USART_MODEM_STATUS UART5_GetModemStatus (void) {
    return UART_GetModemStatus (uart[5]);
  }
#endif

/**
  \fn          int32_t UART_Control (uint32_t control,
                                      uint32_t arg,
                                      UART_RESOURCES  *uart)
  \brief       Control UART Interface.
  \param[in]   control  Operation
  \param[in]   arg      Argument of operation (optional)
  \param[in]   uart    Pointer to USART resources
  \return      common \ref execution_status and driver specific \ref usart_execution_status
*/
static int32_t UART_Control (uint32_t          control,
                              uint32_t          arg,
                              UART_RESOURCES   *uart) {

  switch (control & ARM_USART_CONTROL_Msk) {
    case ARM_USART_MODE_ASYNCHRONOUS:
		XMC_USIC_CH_SetBaudrate(uart->uart, arg, 16UL);
		  break;
    case ARM_USART_MODE_SYNCHRONOUS_MASTER:
      return ARM_USART_ERROR_MODE;
    case ARM_USART_MODE_SYNCHRONOUS_SLAVE:
      return ARM_USART_ERROR_MODE;
    case ARM_USART_MODE_SINGLE_WIRE:
		  XMC_UART_CH_Start(uart->uart);
		  XMC_USIC_CH_SetBaudrate(uart->uart, arg, 16UL);
		 
		  // Configure TX pin		
	    uart->pin_tx_config->mode = (XMC_GPIO_MODE_t)(XMC_GPIO_MODE_OUTPUT_OPEN_DRAIN | uart->pin_tx_alternate_function); 
#if(UC_FAMILY == XMC4)

		  uart->pin_tx_config->output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_MEDIUM_EDGE;
#endif
      break;
    case ARM_USART_MODE_IRDA:
      return ARM_USART_ERROR_MODE;
    case ARM_USART_MODE_SMART_CARD:   
    return ARM_USART_ERROR_MODE;

    // Default TX value
    case ARM_USART_SET_DEFAULT_TX_VALUE:
      return ARM_USART_ERROR_MODE;
    // IrDA pulse
    case ARM_USART_SET_IRDA_PULSE:
      return ARM_USART_ERROR_MODE;

    // SmartCard guard time
    case ARM_USART_SET_SMART_CARD_GUARD_TIME:
      return ARM_USART_ERROR_MODE;
    // SmartCard clock
    case ARM_USART_SET_SMART_CARD_CLOCK:
      return ARM_USART_ERROR_MODE;

     // SmartCard NACK
    case ARM_USART_CONTROL_SMART_CARD_NACK:
      return ARM_USART_ERROR_MODE;

    // Control TX
    case ARM_USART_CONTROL_TX:
     if (arg) 
		 {
       XMC_UART_CH_Start(uart->uart);
		   uart->info->flags |= UART_TX_ENABLED;
       if ((uart->info->mode != ARM_USART_MODE_SINGLE_WIRE )) 
			 {
	       uart->pin_tx_config->mode =(XMC_GPIO_MODE_t)(XMC_GPIO_MODE_OUTPUT_PUSH_PULL | uart->pin_tx_alternate_function);
       }
#if(UC_FAMILY == XMC4)

			   uart->pin_tx_config->output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_MEDIUM_EDGE;
#endif
       // Configure TX pin	
       XMC_GPIO_Init(uart->pin_tx.port,uart->pin_tx.pin, uart->pin_tx_config); 
			 
     
     }	
	   else
	   {
	     uart->info->flags &= ~UART_TX_ENABLED;	
	   }
		 
     return ARM_DRIVER_OK;

    // Control RX
    case ARM_USART_CONTROL_RX:
			XMC_UART_CH_Start(uart->uart);
      if (arg) {
      if ((uart->info->mode != ARM_USART_MODE_SINGLE_WIRE )) {
        // USART RX pin function selected
       uart->pin_rx_config->mode = XMC_GPIO_MODE_INPUT_PULL_UP;
       XMC_GPIO_Init(uart->pin_rx.port,uart->pin_rx.pin, uart->pin_rx_config);
       }
        uart->info->flags |= UART_RX_ENABLED;
      }
			else
			{
			uart->info->flags &= ~UART_RX_ENABLED;
			
			}
	
      return ARM_DRIVER_OK;

    // Control break
    case ARM_USART_CONTROL_BREAK:
      return ARM_USART_ERROR_MODE;

    // Abort Send
    case ARM_USART_ABORT_SEND:
      // Disable transmit holding register empty interrupt
      XMC_USIC_CH_TXFIFO_DisableEvent(uart->uart,XMC_USIC_CH_TXFIFO_EVENT_CONF_STANDARD); 
      XMC_USIC_CH_TXFIFO_Flush(uart->uart);
		  // Clear Send active flag
      uart->info->status.tx_busy=0;
    return ARM_DRIVER_OK;

    // Abort receive
    case ARM_USART_ABORT_RECEIVE:
       // Disable receive data available interrupt
       XMC_USIC_CH_RXFIFO_DisableEvent(uart->uart,XMC_USIC_CH_RXFIFO_EVENT_CONF_STANDARD);
       XMC_USIC_CH_RXFIFO_DisableEvent(uart->uart,XMC_USIC_CH_RXFIFO_EVENT_CONF_ALTERNATE);
			 XMC_USIC_CH_RXFIFO_Flush(uart->uart);
       // Clear RX busy status
		   uart->info->status.rx_busy = 0;
      return ARM_DRIVER_OK;

    // Abort transfer
    case ARM_USART_ABORT_TRANSFER:
      return ARM_USART_ERROR_MODE;

    // Unsupported command
    default: return ARM_DRIVER_ERROR_UNSUPPORTED;
  }

  // Check if Receiver/Transmitter is busy
  if ( uart->info->status.rx_busy ||
      (uart->info->status.tx_busy)) {
    return ARM_DRIVER_ERROR_BUSY;
  }
	
  // USART Data bits
  switch (control & ARM_USART_DATA_BITS_Msk) {
    case ARM_USART_DATA_BITS_5: 
		 XMC_UART_CH_SetWordLength(uart->uart, 5);
		 XMC_UART_CH_SetFrameLength(uart->uart, 5);
		break;
    case ARM_USART_DATA_BITS_6: 
		 XMC_UART_CH_SetWordLength(uart->uart, 6);
		 XMC_UART_CH_SetFrameLength(uart->uart, 6);
    case ARM_USART_DATA_BITS_7: 
		 XMC_UART_CH_SetWordLength(uart->uart, 7);
		 XMC_UART_CH_SetFrameLength(uart->uart, 7);
		break;
    case ARM_USART_DATA_BITS_8: 
			 XMC_UART_CH_SetWordLength(uart->uart, 8);
		 XMC_UART_CH_SetFrameLength(uart->uart, 8);
		break;
		 case ARM_USART_DATA_BITS_9: 
			 XMC_UART_CH_SetWordLength(uart->uart, 9);
		 XMC_UART_CH_SetFrameLength(uart->uart, 9); 
		break;
    default: return ARM_USART_ERROR_DATA_BITS;
  }

  // UART Parity
  switch (control & ARM_USART_PARITY_Msk) {
		
    case ARM_USART_PARITY_NONE:  uart->uart->CCR = XMC_USIC_CH_PARITY_MODE_NONE; break;
    case ARM_USART_PARITY_EVEN:  uart->uart->CCR = XMC_USIC_CH_PARITY_MODE_EVEN; break;
    case ARM_USART_PARITY_ODD:  uart->uart->CCR = XMC_USIC_CH_PARITY_MODE_ODD; break;
  }

  // USART Stop bits
  switch (control & ARM_USART_STOP_BITS_Msk) {
    case ARM_USART_STOP_BITS_1: 	uart->uart->PCR_ASCMode &= ~((uint32_t)USIC_CH_PCR_ASCMode_STPB_Msk);   break;
    case ARM_USART_STOP_BITS_2: 
				uart->uart->PCR_ASCMode &= ~((uint32_t)USIC_CH_PCR_ASCMode_STPB_Msk); 
		    uart->uart->PCR_ASCMode |= ((uint32_t)USIC_CH_PCR_ASCMode_STPB_Msk); 
		    break;
    default: return ARM_USART_ERROR_STOP_BITS;
  }

// UART Flow Control
  switch (control & ARM_USART_FLOW_CONTROL_Msk) {
    default: break;
  }
	
// UART Clock Polarity
  switch (control & ARM_USART_CPOL_Msk) {		
    default: break;
  }
// UART Clock Phase
  switch (control & ARM_USART_CPHA_Msk) {		
    default: break;
  }
  return ARM_DRIVER_OK;
}
															
#if (RTE_UART0 != 0)
  static int32_t UART0_Control (uint32_t  control,uint32_t arg) {
    return UART_Control (control,arg,uart[0]);
  }
#endif
															
#if (RTE_UART1 != 0)
  static int32_t UART1_Control (uint32_t  control,uint32_t arg) {
    return UART_Control (control,arg,uart[1]);
  }
#endif
																
#if (RTE_UART2 != 0)
  static int32_t UART2_Control (uint32_t  control,uint32_t arg) {
    return UART_Control (control,arg,uart[2]);
  }
#endif
																
#if (RTE_UART3 != 0)
  static int32_t UART3_Control (uint32_t  control,uint32_t arg) {
    return UART_Control (control,arg,uart[3]);
  }
#endif
																
#if (RTE_UART4 != 0)
  static int32_t UART4_Control (uint32_t  control,uint32_t arg) {
    return UART_Control (control,arg,uart[4]);
  }
#endif
#if (RTE_UART5 != 0)
  static int32_t UART5_Control (uint32_t  control,uint32_t arg) {
    return UART_Control (control,arg,uart[5]);
  }
#endif
	
/**
  \fn          void UART_IRQHandler (UART_RESOURCES  *uart)
  \brief       UART Interrupt handler.
  \param[in]   uart  Pointer to UART resources
*/
static void UART_IRQHandler (UART_RESOURCES  *uart,uint8_t irq) {
	// Read interrupt
  if (uart->irq_rx_num== irq) 
	{
    if(uart->rx_fifo_size_reg==NO_FIFO)
    {
	     uart->info->xfer.rx_buf[uart->info->xfer.rx_cnt++] =(uint8_t)(uint8_t)XMC_UART_CH_GetReceivedData(uart->uart);;
	     if(uart->info->xfer.rx_cnt == uart->info->xfer.rx_num)
	     {	 
		     // Clear RX busy flag and set receive transfer complete event
			   uart->info->status.rx_busy = 0;
		 
		     XMC_USIC_CH_DisableEvent(uart->uart,XMC_USIC_CH_EVENT_ALTERNATIVE_RECEIVE);
		     XMC_USIC_CH_DisableEvent(uart->uart,XMC_USIC_CH_EVENT_STANDARD_RECEIVE);
			   if (uart->info->cb_event) uart->info->cb_event(ARM_USART_EVENT_RECEIVE_COMPLETE);	
	    }
	  }
	  else
	  {
      while((XMC_USIC_CH_RXFIFO_IsEmpty(uart->uart) == false))
      { 
        /* Read the data from FIFO buffer */
        uart->info->xfer.rx_buf[uart->info->xfer.rx_cnt++] =(uint8_t)XMC_UART_CH_GetReceivedData(uart->uart);
	      if(uart->info->xfer.rx_cnt == uart->info->xfer.rx_num)
        {	 

          // Clear RX busy flag and set receive transfer complete event
          uart->info->status.rx_busy = 0;
          
			    XMC_USIC_CH_RXFIFO_DisableEvent(uart->uart,XMC_USIC_CH_RXFIFO_EVENT_CONF_STANDARD);
          XMC_USIC_CH_RXFIFO_DisableEvent(uart->uart,XMC_USIC_CH_RXFIFO_EVENT_CONF_ALTERNATE);
				  if (uart->info->cb_event) uart->info->cb_event(ARM_USART_EVENT_RECEIVE_COMPLETE);	
			    break;
       }
	   }
	   if( (uart->info->xfer.rx_num-uart->info->xfer.rx_cnt) < uart->rx_fifo_size_num)
	   {
	      XMC_USIC_CH_RXFIFO_Configure(uart->uart,uart->info->rx_fifo_pointer,(XMC_USIC_CH_FIFO_SIZE_t)uart->rx_fifo_size_reg,(uart->info->xfer.rx_num-uart->info->xfer.rx_cnt)- 1U); 
	   }
	   else
	   {
		   XMC_USIC_CH_RXFIFO_Configure(uart->uart,uart->info->rx_fifo_pointer,(XMC_USIC_CH_FIFO_SIZE_t)uart->rx_fifo_size_reg,uart->rx_fifo_size_num - 1U); 
	   }
    }
  }
  // Transmit data register empty
  if (uart->irq_tx_num == irq)
  {
	 if(uart->info->xfer.tx_num > uart->info->xfer.tx_cnt)	
	 {
	  if(uart->rx_fifo_size_reg==NO_FIFO)
    {
			XMC_UART_CH_Transmit(uart->uart,uart->info->xfer.tx_buf[uart->info->xfer.tx_cnt++]);
			
	  }
	  else 
		{
      /* Write to FIFO till Fifo is full */
      while((XMC_USIC_CH_TXFIFO_IsFull(uart->uart) == false))
      { 	
				 if(uart->info->xfer.tx_num > uart->info->xfer.tx_cnt)	
	      {	
	     		XMC_UART_CH_Transmit(uart->uart,uart->info->xfer.tx_buf[uart->info->xfer.tx_cnt++]);
	      }
		    else
		    {
		      break;
		    }
      }      
    }
	}
	else
  {
    if(XMC_USIC_CH_TXFIFO_IsEmpty(uart->uart) == true)
    { 
	    if(uart->rx_fifo_size_reg==NO_FIFO)
      {
				/* Disable standard transmit and error event interrupt */
	      XMC_USIC_CH_DisableEvent(uart->uart,XMC_USIC_CH_EVENT_TRANSMIT_BUFFER);
		  }
      else		
		  {
		    /* Disable standard transmit and error event interrupt */
	      XMC_USIC_CH_TXFIFO_DisableEvent(uart->uart,XMC_USIC_CH_TXFIFO_EVENT_CONF_STANDARD);
		  }
		  while((XMC_USIC_CH_GetTransmitBufferStatus(uart->uart)& XMC_USIC_CH_TBUF_STATUS_BUSY));
		  if (uart->info->cb_event) uart->info->cb_event(ARM_USART_EVENT_SEND_COMPLETE);	
		  // Clear TX busy flag
      uart->info->status.tx_busy=0;		
	  }
  }	
 
 }    
}


#if (RTE_UART0 != 0)
  void USIC0_0_IRQHandler() {
    UART_IRQHandler (uart[0],USIC0_0_IRQn );
  }
	void USIC0_1_IRQHandler() {
    UART_IRQHandler (uart[0],USIC0_1_IRQn);
  }
#endif

#if (RTE_UART1 != 0)
   void USIC0_2_IRQHandler() {
    UART_IRQHandler (uart[1],USIC0_2_IRQn );
  }
	void USIC0_3_IRQHandler() {
    UART_IRQHandler (uart[1],USIC0_3_IRQn);
  }
#endif

#if (RTE_UART2 != 0)
  void USIC1_0_IRQHandler() {
    UART_IRQHandler (uart[2],USIC1_0_IRQn );
  }
	void USIC1_1_IRQHandler() {
    UART_IRQHandler (uart[2],USIC1_1_IRQn);
  }
#endif

#if (RTE_UART3 != 0)
  void USIC1_2_IRQHandler() {
    UART_IRQHandler (uart[3],USIC1_2_IRQn );
  }
	void USIC1_3_IRQHandler() {
    UART_IRQHandler (uart[3],USIC1_3_IRQn);
  }
#endif

#if (RTE_UART4 != 0)
  void USIC2_0_IRQHandler() {
    UART_IRQHandler (uart[4],USIC2_0_IRQn );
  }
	void USIC2_1_IRQHandler() {
    UART_IRQHandler (uart[4],USIC2_1_IRQn);
  }
#endif

#if (RTE_UART5 != 0)
  void USIC2_2_IRQHandler() {
    UART_IRQHandler (uart[5],USIC2_2_IRQn );
  }
	void USIC2_3_IRQHandler() {
    UART_IRQHandler (uart[5],USIC2_3_IRQn);
  }
#endif

#if (RTE_UART0 != 0)
ARM_DRIVER_USART Driver_USART0 = {
USARTX_GetVersion,
UART0_GetCapabilities,
UART0_Initialize,
UART0_Uninitialize,
UART0_PowerControl,
UART0_Send, 
UART0_Receive,
UART0_Transfer,
UART0_GetTxCount,
UART0_GetRxCount,
UART0_Control,
UART0_GetStatus,
UART0_SetModemControl,
UART0_GetModemStatus
};
#endif

#if (RTE_UART1 != 0)
ARM_DRIVER_USART Driver_USART1 = {
USARTX_GetVersion,
UART1_GetCapabilities,
UART1_Initialize,
UART1_Uninitialize,
UART1_PowerControl,
UART1_Send, 
UART1_Receive,
UART1_Transfer,
UART1_GetTxCount,
UART1_GetRxCount,
UART1_Control,
UART1_GetStatus,
UART1_SetModemControl,
UART1_GetModemStatus
};
#endif

#if (RTE_UART2 != 0)
ARM_DRIVER_USART Driver_USART2 = {
USARTX_GetVersion,
UART2_GetCapabilities,
UART2_Initialize,
UART2_Uninitialize,
UART2_PowerControl,
UART2_Send, 
UART2_Receive,
UART2_Transfer,
UART2_GetTxCount,
UART2_GetRxCount,
UART2_Control,
UART2_GetStatus,
UART2_SetModemControl,
UART2_GetModemStatus
};
#endif

#if (RTE_UART3 != 0)
ARM_DRIVER_USART Driver_USART3 = {
USARTX_GetVersion,
UART3_GetCapabilities,
UART3_Initialize,
UART3_Uninitialize,
UART3_PowerControl,
UART3_Send, 
UART3_Receive,
UART3_Transfer,
UART3_GetTxCount,
UART3_GetRxCount,
UART3_Control,
UART3_GetStatus,
UART3_SetModemControl,
UART3_GetModemStatus
};
#endif

#if (RTE_UART4 != 0)
ARM_DRIVER_USART Driver_USART4 = {
USARTX_GetVersion,
UART4_GetCapabilities,
UART4_Initialize,
UART4_Uninitialize,
UART4_PowerControl,
UART4_Send, 
UART4_Receive,
UART4_Transfer,
UART4_GetTxCount,
UART4_GetRxCount,
UART4_Control,
UART4_GetStatus,
UART4_SetModemControl,
UART4_GetModemStatus
};
#endif

#if (RTE_UART5 != 0)
ARM_DRIVER_USART Driver_USART5 = {
USARTX_GetVersion,
UART5_GetCapabilities,
UART5_Initialize,
UART5_Uninitialize,
UART5_PowerControl,
UART5_Send, 
UART5_Receive,
UART5_Transfer,
UART5_GetTxCount,
UART5_GetRxCount,
UART5_Control,
UART5_GetStatus,
UART5_SetModemControl,
UART5_GetModemStatus
};
#endif
