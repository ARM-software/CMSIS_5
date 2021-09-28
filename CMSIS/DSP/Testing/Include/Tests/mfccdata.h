#ifndef _MFCC_DATA_H_
#define _MFCC_DATA_H_ 

#include "arm_math_types.h"


#ifdef   __cplusplus
extern "C"
{
#endif


/*****

 DCT COEFFICIENTS FOR THE MFCC

*****/


#define NB_MFCC_DCT_COEFS_CONFIG1_F32 260
extern const float32_t mfcc_dct_coefs_config1_f32[NB_MFCC_DCT_COEFS_CONFIG1_F32];



#define NB_MFCC_DCT_COEFS_CONFIG1_Q31 260
extern const q31_t mfcc_dct_coefs_config1_q31[NB_MFCC_DCT_COEFS_CONFIG1_Q31];



#define NB_MFCC_DCT_COEFS_CONFIG1_Q15 260
extern const q15_t mfcc_dct_coefs_config1_q15[NB_MFCC_DCT_COEFS_CONFIG1_Q15];



/*****

 WINDOW COEFFICIENTS

*****/


#define NB_MFCC_WIN_COEFS_CONFIG1_F32 1024
extern const float32_t mfcc_window_coefs_config1_f32[NB_MFCC_WIN_COEFS_CONFIG1_F32];



#define NB_MFCC_WIN_COEFS_CONFIG1_Q31 1024
extern const q31_t mfcc_window_coefs_config1_q31[NB_MFCC_WIN_COEFS_CONFIG1_Q31];



#define NB_MFCC_WIN_COEFS_CONFIG1_Q15 1024
extern const q15_t mfcc_window_coefs_config1_q15[NB_MFCC_WIN_COEFS_CONFIG1_Q15];



#define NB_MFCC_WIN_COEFS_CONFIG2_F32 512
extern const float32_t mfcc_window_coefs_config2_f32[NB_MFCC_WIN_COEFS_CONFIG2_F32];



#define NB_MFCC_WIN_COEFS_CONFIG2_Q31 512
extern const q31_t mfcc_window_coefs_config2_q31[NB_MFCC_WIN_COEFS_CONFIG2_Q31];



#define NB_MFCC_WIN_COEFS_CONFIG2_Q15 512
extern const q15_t mfcc_window_coefs_config2_q15[NB_MFCC_WIN_COEFS_CONFIG2_Q15];



#define NB_MFCC_WIN_COEFS_CONFIG3_F32 256
extern const float32_t mfcc_window_coefs_config3_f32[NB_MFCC_WIN_COEFS_CONFIG3_F32];



#define NB_MFCC_WIN_COEFS_CONFIG3_Q31 256
extern const q31_t mfcc_window_coefs_config3_q31[NB_MFCC_WIN_COEFS_CONFIG3_Q31];



#define NB_MFCC_WIN_COEFS_CONFIG3_Q15 256
extern const q15_t mfcc_window_coefs_config3_q15[NB_MFCC_WIN_COEFS_CONFIG3_Q15];



/*****

 MEL FILTER COEFFICIENTS FOR THE MFCC

*****/

#define NB_MFCC_NB_FILTER_CONFIG1_F32 20
extern const uint32_t mfcc_filter_pos_config1_f32[NB_MFCC_NB_FILTER_CONFIG1_F32];
extern const uint32_t mfcc_filter_len_config1_f32[NB_MFCC_NB_FILTER_CONFIG1_F32];

#define NB_MFCC_NB_FILTER_CONFIG1_Q31 20
extern const uint32_t mfcc_filter_pos_config1_q31[NB_MFCC_NB_FILTER_CONFIG1_Q31];
extern const uint32_t mfcc_filter_len_config1_q31[NB_MFCC_NB_FILTER_CONFIG1_Q31];

#define NB_MFCC_NB_FILTER_CONFIG1_Q15 20
extern const uint32_t mfcc_filter_pos_config1_q15[NB_MFCC_NB_FILTER_CONFIG1_Q15];
extern const uint32_t mfcc_filter_len_config1_q15[NB_MFCC_NB_FILTER_CONFIG1_Q15];

#define NB_MFCC_NB_FILTER_CONFIG2_F32 20
extern const uint32_t mfcc_filter_pos_config2_f32[NB_MFCC_NB_FILTER_CONFIG2_F32];
extern const uint32_t mfcc_filter_len_config2_f32[NB_MFCC_NB_FILTER_CONFIG2_F32];

#define NB_MFCC_NB_FILTER_CONFIG2_Q31 20
extern const uint32_t mfcc_filter_pos_config2_q31[NB_MFCC_NB_FILTER_CONFIG2_Q31];
extern const uint32_t mfcc_filter_len_config2_q31[NB_MFCC_NB_FILTER_CONFIG2_Q31];

#define NB_MFCC_NB_FILTER_CONFIG2_Q15 20
extern const uint32_t mfcc_filter_pos_config2_q15[NB_MFCC_NB_FILTER_CONFIG2_Q15];
extern const uint32_t mfcc_filter_len_config2_q15[NB_MFCC_NB_FILTER_CONFIG2_Q15];

#define NB_MFCC_NB_FILTER_CONFIG3_F32 20
extern const uint32_t mfcc_filter_pos_config3_f32[NB_MFCC_NB_FILTER_CONFIG3_F32];
extern const uint32_t mfcc_filter_len_config3_f32[NB_MFCC_NB_FILTER_CONFIG3_F32];

#define NB_MFCC_NB_FILTER_CONFIG3_Q31 20
extern const uint32_t mfcc_filter_pos_config3_q31[NB_MFCC_NB_FILTER_CONFIG3_Q31];
extern const uint32_t mfcc_filter_len_config3_q31[NB_MFCC_NB_FILTER_CONFIG3_Q31];

#define NB_MFCC_NB_FILTER_CONFIG3_Q15 20
extern const uint32_t mfcc_filter_pos_config3_q15[NB_MFCC_NB_FILTER_CONFIG3_Q15];
extern const uint32_t mfcc_filter_len_config3_q15[NB_MFCC_NB_FILTER_CONFIG3_Q15];





#define NB_MFCC_FILTER_COEFS_CONFIG1_F32 948
extern const float32_t mfcc_filter_coefs_config1_f32[NB_MFCC_FILTER_COEFS_CONFIG1_F32];



#define NB_MFCC_FILTER_COEFS_CONFIG1_Q31 948
extern const q31_t mfcc_filter_coefs_config1_q31[NB_MFCC_FILTER_COEFS_CONFIG1_Q31];



#define NB_MFCC_FILTER_COEFS_CONFIG1_Q15 948
extern const q15_t mfcc_filter_coefs_config1_q15[NB_MFCC_FILTER_COEFS_CONFIG1_Q15];



#define NB_MFCC_FILTER_COEFS_CONFIG2_F32 473
extern const float32_t mfcc_filter_coefs_config2_f32[NB_MFCC_FILTER_COEFS_CONFIG2_F32];



#define NB_MFCC_FILTER_COEFS_CONFIG2_Q31 473
extern const q31_t mfcc_filter_coefs_config2_q31[NB_MFCC_FILTER_COEFS_CONFIG2_Q31];



#define NB_MFCC_FILTER_COEFS_CONFIG2_Q15 473
extern const q15_t mfcc_filter_coefs_config2_q15[NB_MFCC_FILTER_COEFS_CONFIG2_Q15];



#define NB_MFCC_FILTER_COEFS_CONFIG3_F32 236
extern const float32_t mfcc_filter_coefs_config3_f32[NB_MFCC_FILTER_COEFS_CONFIG3_F32];



#define NB_MFCC_FILTER_COEFS_CONFIG3_Q31 236
extern const q31_t mfcc_filter_coefs_config3_q31[NB_MFCC_FILTER_COEFS_CONFIG3_Q31];



#define NB_MFCC_FILTER_COEFS_CONFIG3_Q15 236
extern const q15_t mfcc_filter_coefs_config3_q15[NB_MFCC_FILTER_COEFS_CONFIG3_Q15];



#ifdef   __cplusplus
}
#endif

#endif

