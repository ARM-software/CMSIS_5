        .global Reset_Handler
        .global __StackLimit
        .global _ttywrch
        .global _sys_exit
        .syntax unified
        .text
        .thumb
        .thumb_func
        .align  2
        .global  Reset_Handler


        .type   _ttywrch, %function

 _ttywrch:
        PUSH     {r3,lr}
        MOV      r1,sp
        STRB     r0,[sp,#0]
        MOVS     r0,#3
        BKPT     #0xab
        POP      {r3,pc}
        MOVS     r0,r0

        .type   _sys_exit, %function

_sys_exit:
        LDR      r1,=0x20026
        MOVS     r0,#0x18
        BKPT     #0xab
loop:   B        loop

        .type   Reset_Handler, %function

Reset_Handler:  
               
                

                LDR      R0,=ResetC_Handler
                BLX      R0

        .end