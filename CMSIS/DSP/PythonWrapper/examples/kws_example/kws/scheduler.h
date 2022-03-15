/*

Generated with CMSIS-DSP SDF Scripts.
The generated code is not covered by CMSIS-DSP license.

The support classes and code is covered by CMSIS-DSP license.

*/

#ifndef _SCHED_H_ 
#define _SCHED_H_

#ifdef   __cplusplus
extern "C"
{
#endif

extern uint32_t scheduler(int *error,const q15_t *window,
        const q15_t *coef_q15,
        const int coef_shift,
        const q15_t intercept_q15,
        const int intercept_shift);

#ifdef   __cplusplus
}
#endif

#endif

