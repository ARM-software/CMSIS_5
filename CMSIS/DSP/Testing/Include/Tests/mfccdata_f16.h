#ifndef _MFCC_DATA_H_
#define _MFCC_DATA_H_ 

#include "arm_math_types.h"

#include "arm_math_types_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif


/*****

 DCT COEFFICIENTS FOR THE MFCC

*****/

#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_DCT_COEFS_CONFIG1_F16 260
extern const float16_t mfcc_dct_coefs_config1_f16[NB_MFCC_DCT_COEFS_CONFIG1_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */


/*****

 WINDOW COEFFICIENTS

*****/

#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_WIN_COEFS_CONFIG1_F16 1024
extern const float16_t mfcc_window_coefs_config1_f16[NB_MFCC_WIN_COEFS_CONFIG1_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */

#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_WIN_COEFS_CONFIG2_F16 512
extern const float16_t mfcc_window_coefs_config2_f16[NB_MFCC_WIN_COEFS_CONFIG2_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */

#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_WIN_COEFS_CONFIG3_F16 256
extern const float16_t mfcc_window_coefs_config3_f16[NB_MFCC_WIN_COEFS_CONFIG3_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */


/*****

 MEL FILTER COEFFICIENTS FOR THE MFCC

*****/

#define NB_MFCC_NB_FILTER_CONFIG1_F16 20
extern const uint32_t mfcc_filter_pos_config1_f16[NB_MFCC_NB_FILTER_CONFIG1_F16];
extern const uint32_t mfcc_filter_len_config1_f16[NB_MFCC_NB_FILTER_CONFIG1_F16];

#define NB_MFCC_NB_FILTER_CONFIG2_F16 20
extern const uint32_t mfcc_filter_pos_config2_f16[NB_MFCC_NB_FILTER_CONFIG2_F16];
extern const uint32_t mfcc_filter_len_config2_f16[NB_MFCC_NB_FILTER_CONFIG2_F16];

#define NB_MFCC_NB_FILTER_CONFIG3_F16 20
extern const uint32_t mfcc_filter_pos_config3_f16[NB_MFCC_NB_FILTER_CONFIG3_F16];
extern const uint32_t mfcc_filter_len_config3_f16[NB_MFCC_NB_FILTER_CONFIG3_F16];




#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_FILTER_COEFS_CONFIG1_F16 948
extern const float16_t mfcc_filter_coefs_config1_f16[NB_MFCC_FILTER_COEFS_CONFIG1_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */

#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_FILTER_COEFS_CONFIG2_F16 473
extern const float16_t mfcc_filter_coefs_config2_f16[NB_MFCC_FILTER_COEFS_CONFIG2_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */

#if defined(ARM_FLOAT16_SUPPORTED)
#define NB_MFCC_FILTER_COEFS_CONFIG3_F16 236
extern const float16_t mfcc_filter_coefs_config3_f16[NB_MFCC_FILTER_COEFS_CONFIG3_F16];
#endif /*defined(ARM_FLOAT16_SUPPORTED) */


#ifdef   __cplusplus
}
#endif

#endif

