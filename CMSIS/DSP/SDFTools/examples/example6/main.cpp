#include <cstdio>
#include <cstdint>
#include "arm_math.h"
#include "scheduler.h"
#include "mfccConfigData.h"

static arm_mfcc_instance_f32 mfcc;

int main(int argc, char const *argv[])
{
    int error;
    printf("Start\n");

    arm_mfcc_init_f32(&mfcc,
                    256,20,13,mfcc_dct_coefs_config1_f32,
                    mfcc_filter_pos_config1_f32,mfcc_filter_len_config1_f32,
                    mfcc_filter_coefs_config1_f32,
                    mfcc_window_coefs_config1_f32);

    uint32_t nb = scheduler(&error,&mfcc);
    printf("Nb = %d\n",nb);
    return 0;
}