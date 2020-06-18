#include "cmsis_compiler.h"
#include "cmsis_cp15.h"
#include "mem_ARMCA32.h"

// TTB base address
#define TTB_BASE ((uint32_t*)__TTB_BASE)


void MMU_CreateTranslationTable(void)
{

    /* Set location of level 1 page table
    ; 31:14 - Translation table base addr (31:14-TTBCR.N, TTBCR.N is 0 out of reset)
    ; 13:7  - 0x0
    ; 6     - IRGN[0] 0x1  (Inner WB WA)
    ; 5     - NOS     0x0  (Non-shared)
    ; 4:3   - RGN     0x01 (Outer WB WA)
    ; 2     - IMP     0x0  (Implementation Defined)
    ; 1     - S       0x0  (Non-shared)
    ; 0     - IRGN[1] 0x0  (Inner WB WA) */
    __set_TTBR0(__TTB_BASE);
    __ISB();

    /* Set up domain access control register
    ; We set domain 0 to Client and all other domains to No Access.
    ; All translation table entries specify domain 0 */
    __set_DACR(0xFFFFFFFF);
    __ISB();
}
