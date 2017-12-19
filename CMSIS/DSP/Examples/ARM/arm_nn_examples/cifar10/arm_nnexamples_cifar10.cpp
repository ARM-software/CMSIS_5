#include <stdint.h>
#include <stdio.h>
#include "arm_math.h"
#define IP_X4
#include "parameter.h"
#include "shift.h"
#ifdef IP_X4
#include "weights.h"
#else
#include "weights.h"
#endif
#include "arm_nnfunctions.h"
#include "inputs.h"

/*  Smaller version of network definition for CIFAR10 from caffe examples
 *  Network statistics (HWC format)
 *
 *    32x32x3 (3kB)
 *       |
 *     Conv1 : Weight 5x5x3x32 2.34kB, Ops 32*32*32 * 2*5*5*3 4.9 MOps
 *       |
 *    32x32x32 (32kB)
 *       |
 *     Pool1 : Ops 3*3 * 16*16*32 73.7 kOps
 *       |
 *    16x16x32 (8kB)
 *       |
 *     Conv2 : 5x5x32x16 (12.8kB), Ops 16*16*32 * 2*5*5*16 6.5 MOps
 *       |
 *    16x16x16 (8kB)
 *       |
 *     Pool2 : Ops 3*3 * 8*8*16, 9.2 kOps
 *       |
 *     8x8x16 (1kB)
 *       |
 *     Conv3 : 5x5x16x32 (12.8kB), Ops 8*8*16 * 2*5*5*32 1.6 MOps
 *       |
 *    8x8x32 (2kB)
 *       |
 *     Pool3 : Ops 3*3 * 4*4*32 4.6k
 *       |
 *    4x4x32 (0.5kB)
 *       |
 *      IP1 : 4x4x32x10 (5kB), Ops 10 * 2*4*4*32 10k
 *       |
 *      10
 *
 *    Total Ops:  13.2 MOps
 *
 */

/*
 *
 *   Memory footprint
 *
 *   Weights: ~33.1kB
 *   I/O: ~3kB
 *   Buffers: ~40kB (activations) + 3.2kB (im2col buffer)
 *   Activation buffer size can be reduced if conv-pool are fused
 *
 */

// include the input and weights

static q7_t conv1_wt[CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

static q7_t conv2_wt[CONV2_IM_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IM_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

static q7_t ip1_wt[IP1_DIM * IP1_OUT] = IP1_WT;
static q7_t ip1_bias[IP1_OUT] = IP1_BIAS;

q7_t      input_data[CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA;
q7_t      output_data[IP1_OUT];

//vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
q7_t      col_buffer[2 * 5 * 5 * 32 * 2];

q7_t      scratch_buffer[32 * 32 * 10 * 4];

int main()
{

    printf("start execution\n");
    /* start the execution */

    q7_t     *img_buffer1 = scratch_buffer;
    q7_t     *img_buffer2 = img_buffer1 + 32 * 32 * 32;

    // conv1 input_data -> img_buffer1
    arm_convolve_HWC_q7_RGB(input_data, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
                            CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, img_buffer1, CONV1_OUT_DIM,
                            (q15_t *) col_buffer, NULL);

    arm_relu_q7(img_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);

    // pool1 img_buffer1 -> img_buffer2
    arm_maxpool_q7_HWC(img_buffer1, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
                       POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, img_buffer2);

    // conv2 img_buffer2 -> img_buffer1
    arm_convolve_HWC_q7_fast(img_buffer2, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM,
                             CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, img_buffer1,
                             CONV2_OUT_DIM, (q15_t *) col_buffer, NULL);

    arm_relu_q7(img_buffer1, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);

    // pool2 img_buffer1 -> img_buffer2
    arm_avepool_q7_HWC(img_buffer1, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
                       POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, img_buffer2);

// conv3 img_buffer2 -> img_buffer1
    arm_convolve_HWC_q7_fast(img_buffer2, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM,
                             CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, img_buffer1,
                             CONV3_OUT_DIM, (q15_t *) col_buffer, NULL);

    arm_relu_q7(img_buffer1, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);

    // pool3 img_buffer-> img_buffer2
    arm_avepool_q7_HWC(img_buffer1, CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM,
                       POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, img_buffer2);

#ifdef IP_X4
    arm_fully_connected_q7_opt(img_buffer2, ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias,
                               output_data, (q15_t *) img_buffer1);
#else
    arm_fully_connected_q7(img_buffer2, ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias,
                           output_data, (q15_t *) img_buffer1);
#endif

    for (int i = 0; i < 10; i++)
    {
        printf("%d: %d\n", i, output_data[i]);
    }

    return 0;
}
