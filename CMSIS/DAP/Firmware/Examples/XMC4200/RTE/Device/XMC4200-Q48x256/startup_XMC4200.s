;*******************************************************************************
;* @file     startup_XMC4200.s
;* @brief    CMSIS Core Device Startup File for
;*           Infineon XMC4200 Device Series
;* @version  V1.2
;* @date     November 2014
;*
;* Copyright (C) 2014 Infineon Technologies AG. All rights reserved.
;*
;*
;* @par
;* Infineon Technologies AG (Infineon) is supplying this software for use with 
;* Infineon's microcontrollers.  This file can be freely distributed
;* within development tools that are supporting such microcontrollers.
;*
;* @par
;* THIS SOFTWARE IS PROVIDED AS IS.  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
;* OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
;* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
;* ARM SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR
;* CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.
;*
;******************************************************************************

;************************** Version History ************************************
; V0.1, September 2012, First version
; V1.0, February 2013, FIX for CPU prefetch bug implemented
; V1.1, August 2013,  Fix the bug of stack pointer alignment to a 8 byte boundary
; V1.2, November 2014, Disable CPU workaround. Increased stack size.
;                       Removed DAVE3 dependency
;*******************************************************************************

; ------------------ <<< Use Configuration Wizard in Context Menu >>> ------------------
           
; <h> Stack Configuration
;   <o> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
; </h>

Stack_Size      EQU     0x00000800

                AREA    STACK, NOINIT, READWRITE, ALIGN=3
Stack_Mem       SPACE   Stack_Size
__initial_sp


; <h> Heap Configuration
;   <o>  Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
; </h>

Heap_Size       EQU     0x00000200

                AREA    HEAP, NOINIT, READWRITE, ALIGN=3
__heap_base
Heap_Mem        SPACE   Heap_Size
__heap_limit

                PRESERVE8
                THUMB

                IF    :DEF:ENABLE_CPU_CM_001_WORKAROUND
                    MACRO
                    Entry $Handler
                    DCD   $Handler._Veneer
                    MEND
                ELSE
                    MACRO
                    Entry $Handler
                    DCD   $Handler
                    MEND
                ENDIF

; Vector Table Mapped to Address 0 at Reset

                AREA    RESET, DATA, READONLY
                EXPORT  __Vectors
                EXPORT  __Vectors_End
                EXPORT  __Vectors_Size
                
__Vectors       DCD     __initial_sp              ; 0 Top of Stack
                DCD     Reset_Handler             ; 1 Reset Handler
                Entry   NMI_Handler               ; 2 NMI Handler
                Entry   HardFault_Handler         ; 3 Hard Fault Handler
                Entry   MemManage_Handler         ; 4 MPU Fault Handler
                Entry   BusFault_Handler          ; 5 Bus Fault Handler
                Entry   UsageFault_Handler        ; 6 Usage Fault Handler
                DCD     0                         ; 7 Reserved
                DCD     0                         ; 8 Reserved
                DCD     0                         ; 9 Reserved
                DCD     0                         ; 10 Reserved
                Entry   SVC_Handler               ; 11 SVCall Handler
                Entry   DebugMon_Handler          ; 12 Debug Monitor Handler
                DCD     0                         ; 13 Reserved
                Entry   PendSV_Handler            ; 14 PendSV Handler
                Entry   SysTick_Handler           ; 15 SysTick Handler

; Interrupt Handlers for Service Requests (SR) from XMC4200 Peripherals */
                Entry   SCU_0_IRQHandler          ; Handler name for SR SCU_0     
                Entry   ERU0_0_IRQHandler         ; Handler name for SR ERU0_0    
                Entry   ERU0_1_IRQHandler         ; Handler name for SR ERU0_1    
                Entry   ERU0_2_IRQHandler         ; Handler name for SR ERU0_2    
                Entry   ERU0_3_IRQHandler         ; Handler name for SR ERU0_3     
                Entry   ERU1_0_IRQHandler         ; Handler name for SR ERU1_0    
                Entry   ERU1_1_IRQHandler         ; Handler name for SR ERU1_1    
                Entry   ERU1_2_IRQHandler         ; Handler name for SR ERU1_2    
                Entry   ERU1_3_IRQHandler         ; Handler name for SR ERU1_3    
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                Entry   PMU0_0_IRQHandler         ; Handler name for SR PMU0_0    
                DCD     0                         ; Not Available                 
                Entry   VADC0_C0_0_IRQHandler     ; Handler name for SR VADC0_C0_0  
                Entry   VADC0_C0_1_IRQHandler     ; Handler name for SR VADC0_C0_1  
                Entry   VADC0_C0_2_IRQHandler     ; Handler name for SR VADC0_C0_1  
                Entry   VADC0_C0_3_IRQHandler     ; Handler name for SR VADC0_C0_3  
                Entry   VADC0_G0_0_IRQHandler     ; Handler name for SR VADC0_G0_0  
                Entry   VADC0_G0_1_IRQHandler     ; Handler name for SR VADC0_G0_1  
                Entry   VADC0_G0_2_IRQHandler     ; Handler name for SR VADC0_G0_2  
                Entry   VADC0_G0_3_IRQHandler     ; Handler name for SR VADC0_G0_3  
                Entry   VADC0_G1_0_IRQHandler     ; Handler name for SR VADC0_G1_0  
                Entry   VADC0_G1_1_IRQHandler     ; Handler name for SR VADC0_G1_1  
                Entry   VADC0_G1_2_IRQHandler     ; Handler name for SR VADC0_G1_2  
                Entry   VADC0_G1_3_IRQHandler     ; Handler name for SR VADC0_G1_3  
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                Entry   DAC0_0_IRQHandler         ; Handler name for SR DAC0_0    
                Entry   DAC0_1_IRQHandler         ; Handler name for SR DAC0_1    
                Entry   CCU40_0_IRQHandler        ; Handler name for SR CCU40_0   
                Entry   CCU40_1_IRQHandler        ; Handler name for SR CCU40_1   
                Entry   CCU40_2_IRQHandler        ; Handler name for SR CCU40_2   
                Entry   CCU40_3_IRQHandler        ; Handler name for SR CCU40_3   
                Entry   CCU41_0_IRQHandler        ; Handler name for SR CCU41_0   
                Entry   CCU41_1_IRQHandler        ; Handler name for SR CCU41_1   
                Entry   CCU41_2_IRQHandler        ; Handler name for SR CCU41_2   
                Entry   CCU41_3_IRQHandler        ; Handler name for SR CCU41_3   
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                Entry   CCU80_0_IRQHandler        ; Handler name for SR CCU80_0   
                Entry   CCU80_1_IRQHandler        ; Handler name for SR CCU80_1   
                Entry   CCU80_2_IRQHandler        ; Handler name for SR CCU80_2   
                Entry   CCU80_3_IRQHandler        ; Handler name for SR CCU80_3   
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                Entry   POSIF0_0_IRQHandler       ; Handler name for SR POSIF0_0  
                Entry   POSIF0_1_IRQHandler       ; Handler name for SR POSIF0_1  
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                Entry   HRPWM_0_IRQHandler        ; Handler name for SR HRPWM_0   
                Entry   HRPWM_1_IRQHandler        ; Handler name for SR HRPWM_1   
                Entry   HRPWM_2_IRQHandler        ; Handler name for SR HRPWM_2   
                Entry   HRPWM_3_IRQHandler        ; Handler name for SR HRPWM_3   
                Entry   CAN0_0_IRQHandler         ; Handler name for SR CAN0_0    
                Entry   CAN0_1_IRQHandler         ; Handler name for SR CAN0_1    
                Entry   CAN0_2_IRQHandler         ; Handler name for SR CAN0_2    
                Entry   CAN0_3_IRQHandler         ; Handler name for SR CAN0_3    
                Entry   CAN0_4_IRQHandler         ; Handler name for SR CAN0_4    
                Entry   CAN0_5_IRQHandler         ; Handler name for SR CAN0_5    
                Entry   CAN0_6_IRQHandler         ; Handler name for SR CAN0_6    
                Entry   CAN0_7_IRQHandler         ; Handler name for SR CAN0_7    
                Entry   USIC0_0_IRQHandler        ; Handler name for SR USIC0_0   
                Entry   USIC0_1_IRQHandler        ; Handler name for SR USIC0_1   
                Entry   USIC0_2_IRQHandler        ; Handler name for SR USIC0_2   
                Entry   USIC0_3_IRQHandler        ; Handler name for SR USIC0_3   
                Entry   USIC0_4_IRQHandler        ; Handler name for SR USIC0_4   
                Entry   USIC0_5_IRQHandler        ; Handler name for SR USIC0_5   
                Entry   USIC1_0_IRQHandler        ; Handler name for SR USIC1_0   
                Entry   USIC1_1_IRQHandler        ; Handler name for SR USIC1_1   
                Entry   USIC1_2_IRQHandler        ; Handler name for SR USIC1_2   
                Entry   USIC1_3_IRQHandler        ; Handler name for SR USIC1_3   
                Entry   USIC1_4_IRQHandler        ; Handler name for SR USIC1_4   
                Entry   USIC1_5_IRQHandler        ; Handler name for SR USIC1_5   
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                Entry   LEDTS0_0_IRQHandler       ; Handler name for SR LEDTS0_0  
                DCD     0                         ; Not Available                 
                Entry   FCE0_0_IRQHandler         ; Handler name for SR FCE0_0    
                Entry   GPDMA0_0_IRQHandler       ; Handler name for SR GPDMA0_0  
                DCD     0                         ; Not Available                 
                Entry   USB0_0_IRQHandler         ; Handler name for SR USB0_0    
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
                DCD     0                         ; Not Available                 
__Vectors_End

__Vectors_Size  EQU     __Vectors_End - __Vectors

                AREA    |.text|, CODE, READONLY

; Reset Handler

Reset_Handler   PROC
                EXPORT  Reset_Handler           [WEAK]
                IMPORT  SystemInit
                IMPORT  __main
                LDR     SP, =__initial_sp                
                LDR     R0, =SystemInit
                BLX     R0
                LDR     R0, =__main
                BX      R0
                ENDP

; Dummy Exception Handlers (infinite loops which can be modified)
             
Default_Handler PROC
                EXPORT  NMI_Handler             [WEAK]
                EXPORT  HardFault_Handler       [WEAK]
                EXPORT  MemManage_Handler       [WEAK]
                EXPORT  BusFault_Handler        [WEAK]
                EXPORT  UsageFault_Handler      [WEAK]
                EXPORT  SVC_Handler             [WEAK]
                EXPORT  DebugMon_Handler        [WEAK]
                EXPORT  PendSV_Handler          [WEAK]
                EXPORT  SysTick_Handler         [WEAK]
                
                EXPORT  SCU_0_IRQHandler        [WEAK]
                EXPORT  ERU0_0_IRQHandler       [WEAK]
                EXPORT  ERU0_1_IRQHandler       [WEAK]
                EXPORT  ERU0_2_IRQHandler       [WEAK]
                EXPORT  ERU0_3_IRQHandler       [WEAK]
                EXPORT  ERU1_0_IRQHandler       [WEAK]
                EXPORT  ERU1_1_IRQHandler       [WEAK]
                EXPORT  ERU1_2_IRQHandler       [WEAK]
                EXPORT  ERU1_3_IRQHandler       [WEAK]
                EXPORT  PMU0_0_IRQHandler       [WEAK]
                EXPORT  VADC0_C0_0_IRQHandler   [WEAK]
                EXPORT  VADC0_C0_1_IRQHandler   [WEAK]
                EXPORT  VADC0_C0_2_IRQHandler   [WEAK]
                EXPORT  VADC0_C0_3_IRQHandler   [WEAK]
                EXPORT  VADC0_G0_0_IRQHandler   [WEAK]
                EXPORT  VADC0_G0_1_IRQHandler   [WEAK]
                EXPORT  VADC0_G0_2_IRQHandler   [WEAK]
                EXPORT  VADC0_G0_3_IRQHandler   [WEAK]
                EXPORT  VADC0_G1_0_IRQHandler   [WEAK]
                EXPORT  VADC0_G1_1_IRQHandler   [WEAK]
                EXPORT  VADC0_G1_2_IRQHandler   [WEAK]
                EXPORT  VADC0_G1_3_IRQHandler   [WEAK]
                EXPORT  DSD0_0_IRQHandler       [WEAK]
                EXPORT  DSD0_1_IRQHandler       [WEAK]
                EXPORT  DSD0_2_IRQHandler       [WEAK]
                EXPORT  DSD0_3_IRQHandler       [WEAK]
                EXPORT  DSD0_4_IRQHandler       [WEAK]
                EXPORT  DSD0_5_IRQHandler       [WEAK]
                EXPORT  DSD0_6_IRQHandler       [WEAK]
                EXPORT  DSD0_7_IRQHandler       [WEAK]
                EXPORT  DAC0_0_IRQHandler       [WEAK]
                EXPORT  DAC0_1_IRQHandler       [WEAK]
                EXPORT  CCU40_0_IRQHandler      [WEAK]
                EXPORT  CCU40_1_IRQHandler      [WEAK]
                EXPORT  CCU40_2_IRQHandler      [WEAK]
                EXPORT  CCU40_3_IRQHandler      [WEAK]
                EXPORT  CCU41_0_IRQHandler      [WEAK]
                EXPORT  CCU41_1_IRQHandler      [WEAK]
                EXPORT  CCU41_2_IRQHandler      [WEAK]
                EXPORT  CCU41_3_IRQHandler      [WEAK]
                EXPORT  CCU80_0_IRQHandler      [WEAK]
                EXPORT  CCU80_1_IRQHandler      [WEAK]
                EXPORT  CCU80_2_IRQHandler      [WEAK]
                EXPORT  CCU80_3_IRQHandler      [WEAK]
                EXPORT  POSIF0_0_IRQHandler     [WEAK]
                EXPORT  POSIF0_1_IRQHandler     [WEAK]
                EXPORT  HRPWM_0_IRQHandler      [WEAK]
                EXPORT  HRPWM_1_IRQHandler      [WEAK]
                EXPORT  HRPWM_2_IRQHandler      [WEAK]
                EXPORT  HRPWM_3_IRQHandler      [WEAK]
                EXPORT  CAN0_0_IRQHandler       [WEAK]
                EXPORT  CAN0_1_IRQHandler       [WEAK]
                EXPORT  CAN0_2_IRQHandler       [WEAK]
                EXPORT  CAN0_3_IRQHandler       [WEAK]
                EXPORT  CAN0_4_IRQHandler       [WEAK]
                EXPORT  CAN0_5_IRQHandler       [WEAK]
                EXPORT  CAN0_6_IRQHandler       [WEAK]
                EXPORT  CAN0_7_IRQHandler       [WEAK]
                EXPORT  USIC0_0_IRQHandler      [WEAK]
                EXPORT  USIC0_1_IRQHandler      [WEAK]
                EXPORT  USIC0_2_IRQHandler      [WEAK]
                EXPORT  USIC0_3_IRQHandler      [WEAK]
                EXPORT  USIC0_4_IRQHandler      [WEAK]
                EXPORT  USIC0_5_IRQHandler      [WEAK]
                EXPORT  USIC1_0_IRQHandler      [WEAK]
                EXPORT  USIC1_1_IRQHandler      [WEAK]
                EXPORT  USIC1_2_IRQHandler      [WEAK]
                EXPORT  USIC1_3_IRQHandler      [WEAK]
                EXPORT  USIC1_4_IRQHandler      [WEAK]
                EXPORT  USIC1_5_IRQHandler      [WEAK]
                EXPORT  LEDTS0_0_IRQHandler     [WEAK]
                EXPORT  FCE0_0_IRQHandler       [WEAK]
                EXPORT  GPDMA0_0_IRQHandler     [WEAK]
                EXPORT  USB0_0_IRQHandler       [WEAK]

NMI_Handler
HardFault_Handler
MemManage_Handler
BusFault_Handler
UsageFault_Handler
SVC_Handler
DebugMon_Handler
PendSV_Handler
SysTick_Handler                
SCU_0_IRQHandler
ERU0_0_IRQHandler 
ERU0_1_IRQHandler 
ERU0_2_IRQHandler 
ERU0_3_IRQHandler 
ERU1_0_IRQHandler 
ERU1_1_IRQHandler 
ERU1_2_IRQHandler 
ERU1_3_IRQHandler 
PMU0_0_IRQHandler 
VADC0_C0_0_IRQHandler 
VADC0_C0_1_IRQHandler 
VADC0_C0_2_IRQHandler 
VADC0_C0_3_IRQHandler 
VADC0_G0_0_IRQHandler 
VADC0_G0_1_IRQHandler 
VADC0_G0_2_IRQHandler 
VADC0_G0_3_IRQHandler 
VADC0_G1_0_IRQHandler 
VADC0_G1_1_IRQHandler 
VADC0_G1_2_IRQHandler 
VADC0_G1_3_IRQHandler 
DSD0_0_IRQHandler 
DSD0_1_IRQHandler 
DSD0_2_IRQHandler 
DSD0_3_IRQHandler 
DSD0_4_IRQHandler 
DSD0_5_IRQHandler 
DSD0_6_IRQHandler 
DSD0_7_IRQHandler 
DAC0_0_IRQHandler 
DAC0_1_IRQHandler 
CCU40_0_IRQHandler
CCU40_1_IRQHandler
CCU40_2_IRQHandler
CCU40_3_IRQHandler
CCU41_0_IRQHandler
CCU41_1_IRQHandler
CCU41_2_IRQHandler
CCU41_3_IRQHandler
CCU80_0_IRQHandler
CCU80_1_IRQHandler
CCU80_2_IRQHandler
CCU80_3_IRQHandler
POSIF0_0_IRQHandler 
POSIF0_1_IRQHandler 
HRPWM_0_IRQHandler
HRPWM_1_IRQHandler
HRPWM_2_IRQHandler
HRPWM_3_IRQHandler
CAN0_0_IRQHandler 
CAN0_1_IRQHandler 
CAN0_2_IRQHandler 
CAN0_3_IRQHandler 
CAN0_4_IRQHandler 
CAN0_5_IRQHandler 
CAN0_6_IRQHandler 
CAN0_7_IRQHandler 
USIC0_0_IRQHandler
USIC0_1_IRQHandler
USIC0_2_IRQHandler
USIC0_3_IRQHandler
USIC0_4_IRQHandler
USIC0_5_IRQHandler
USIC1_0_IRQHandler
USIC1_1_IRQHandler
USIC1_2_IRQHandler
USIC1_3_IRQHandler
USIC1_4_IRQHandler
USIC1_5_IRQHandler
LEDTS0_0_IRQHandler 
FCE0_0_IRQHandler 
GPDMA0_0_IRQHandler 
USB0_0_IRQHandler 

                B       .

                ENDP

                IF     :DEF:ENABLE_CPU_CM_001_WORKAROUND

                MACRO
                Insert_ExceptionHandlerVeneer $Handler_Func                    
$Handler_Func._Veneer\
                    PROC
                    EXPORT  $Handler_Func._Veneer
                    LDR     R0, =$Handler_Func
                    PUSH    {LR}  ;/* Breaks AAPCS */
                    SUB     SP,#4     ;/* Restores AAPCS */
                    BLX     R0
                    ADD     SP,#4
                    POP     {PC}
                    ALIGN
                    LTORG
                    ENDP
                MEND
                
                Insert_ExceptionHandlerVeneer  NMI_Handler
                Insert_ExceptionHandlerVeneer  HardFault_Handler
                Insert_ExceptionHandlerVeneer  MemManage_Handler
                Insert_ExceptionHandlerVeneer  BusFault_Handler 
                Insert_ExceptionHandlerVeneer  UsageFault_Handler
                Insert_ExceptionHandlerVeneer  SVC_Handler
                Insert_ExceptionHandlerVeneer  DebugMon_Handler
                Insert_ExceptionHandlerVeneer  PendSV_Handler
                Insert_ExceptionHandlerVeneer SysTick_Handler

                Insert_ExceptionHandlerVeneer  SCU_0_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU0_0_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU0_1_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU0_2_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU0_3_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU1_0_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU1_1_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU1_2_IRQHandler
                Insert_ExceptionHandlerVeneer  ERU1_3_IRQHandler
                Insert_ExceptionHandlerVeneer  PMU0_0_IRQHandler
                Insert_ExceptionHandlerVeneer  VADC0_C0_0_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_C0_1_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_C0_2_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_C0_3_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G0_0_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G0_1_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G0_2_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G0_3_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G1_0_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G1_1_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G1_2_IRQHandler   
                Insert_ExceptionHandlerVeneer  VADC0_G1_3_IRQHandler   
                Insert_ExceptionHandlerVeneer  DSD0_0_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_1_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_2_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_3_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_4_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_5_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_6_IRQHandler       
                Insert_ExceptionHandlerVeneer  DSD0_7_IRQHandler       
                Insert_ExceptionHandlerVeneer  DAC0_0_IRQHandler       
                Insert_ExceptionHandlerVeneer  DAC0_1_IRQHandler       
                Insert_ExceptionHandlerVeneer  CCU40_0_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU40_1_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU40_2_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU40_3_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU41_0_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU41_1_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU41_2_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU41_3_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU80_0_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU80_1_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU80_2_IRQHandler      
                Insert_ExceptionHandlerVeneer  CCU80_3_IRQHandler      
                Insert_ExceptionHandlerVeneer  POSIF0_0_IRQHandler     
                Insert_ExceptionHandlerVeneer  POSIF0_1_IRQHandler     
                Insert_ExceptionHandlerVeneer  HRPWM_0_IRQHandler      
                Insert_ExceptionHandlerVeneer  HRPWM_1_IRQHandler      
                Insert_ExceptionHandlerVeneer  HRPWM_2_IRQHandler      
                Insert_ExceptionHandlerVeneer  HRPWM_3_IRQHandler      
                Insert_ExceptionHandlerVeneer  CAN0_0_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_1_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_2_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_3_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_4_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_5_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_6_IRQHandler       
                Insert_ExceptionHandlerVeneer  CAN0_7_IRQHandler       
                Insert_ExceptionHandlerVeneer  USIC0_0_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC0_1_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC0_2_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC0_3_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC0_4_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC0_5_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC1_0_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC1_1_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC1_2_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC1_3_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC1_4_IRQHandler      
                Insert_ExceptionHandlerVeneer  USIC1_5_IRQHandler      
                Insert_ExceptionHandlerVeneer  LEDTS0_0_IRQHandler     
                Insert_ExceptionHandlerVeneer  FCE0_0_IRQHandler       
                Insert_ExceptionHandlerVeneer  GPDMA0_0_IRQHandler     
                Insert_ExceptionHandlerVeneer  USB0_0_IRQHandler       
                ENDIF
                
                ALIGN

; User Initial Stack & Heap

                IF      :DEF:__MICROLIB

                EXPORT  __initial_sp
                EXPORT  __heap_base
                EXPORT  __heap_limit

                ELSE

                IMPORT  __use_two_region_memory
                EXPORT  __user_initial_stackheap
__user_initial_stackheap

                LDR     R0, =  Heap_Mem
                LDR     R1, =(Stack_Mem + Stack_Size)
                LDR     R2, = (Heap_Mem +  Heap_Size)
                LDR     R3, = Stack_Mem
                BX      LR

                ALIGN

                ENDIF


                END
