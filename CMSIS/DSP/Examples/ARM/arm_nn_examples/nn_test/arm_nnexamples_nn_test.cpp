#include "main.h"

//#define TEST_SIGMOID
//#define TEST_TANH
#define TEST_POOL
#define TEST_RELU
#define TEST_IP
#define TEST_CONV

int main()
{
    printf("start tests\n");
    // common pointers for testing data
    q7_t     *test1;
    q15_t    *test2;
    q7_t     *test3;
    q15_t    *test4;

    bool      if_match;

#ifdef TEST_SIGMOID

#define SIGMOID_DIM 128

    /* This part tests the running of sigmoid functions */

    test1 = new q7_t[SIGMOID_DIM];
    test2 = new q15_t[SIGMOID_DIM];
    test3 = new q7_t[SIGMOID_DIM];
    test4 = new q15_t[SIGMOID_DIM];

    srand(1);

    for (int i = 0; i < SIGMOID_DIM; i++)
    {
        test1[i] = (rand() % 256 - 128);
        test2[i] = (rand() % 65536 - 32768);
        test3[i] = test1[i];
        test4[i] = test2[i];
    }

    arm_sigmoid_direct_q7(test3, SIGMOID_DIM, 3);

    for (int i = 0; i < SIGMOID_DIM; i++)
    {
        printf("in: %d  out: %d\n", test1[i], test3[i]);
    }

    printf("start testing q15_t sigmoid\n\n");

    arm_sigmoid_direct_q15(test4, SIGMOID_DIM, 3);

    for (int i = 0; i < SIGMOID_DIM; i++)
    {
        printf("in: %d  out: %d\n", test2[i], test4[i]);
    }

    delete[]test1;
    delete[]test2;
    delete[]test3;
    delete[]test4;

#endif

#ifdef TEST_TANH

#define TANH_DIM 128

    /* This part tests the running of sigmoid functions */

    test1 = new q7_t[TANH_DIM];
    test2 = new q15_t[TANH_DIM];
    test3 = new q7_t[TANH_DIM];
    test4 = new q15_t[TANH_DIM];

    printf("start testing sigmoid\n");

    srand(1);

    for (int i = 0; i < TANH_DIM; i++)
    {
        test1[i] = (rand() % 256 - 128);
        test2[i] = (rand() % 65536 - 32768);
        test3[i] = test1[i];
        test4[i] = test2[i];
    }

    arm_tanh_direct_q7(test3, TANH_DIM, 3);

    printf("start testing q7_t tanh\n\n");

    for (int i = 0; i < TANH_DIM; i++)
    {
        printf("in: %d  out: %d\n", test1[i], test3[i]);
    }

    printf("start testing q15_t tanh\n\n");

    arm_tanh_direct_q15(test4, TANH_DIM, 3);

    for (int i = 0; i < TANH_DIM; i++)
    {
        printf("in: %d  out: %d\n", test2[i], test4[i]);
    }

    delete[]test1;
    delete[]test2;
    delete[]test3;
    delete[]test4;

#endif

#ifdef TEST_POOL

#define POOL_IM_DIM 32
#define POOL_IM_CH 8

    test1 = new q7_t[POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH * 2];
    test2 = new q15_t[POOL_IM_DIM * POOL_IM_CH];
    test3 = new q7_t[POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH];

    srand(1);

    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        test1[i] = (rand() % 256 - 128);
    }

    q7_t     *img_in = test1 + POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH;
    q7_t     *pool_out_ref = test3;
    q7_t     *pool_out_opt = test3 + POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH / 2;

    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        test3[i] = 0;
    }

    // copy over the img input
    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        img_in[i] = test1[i];
    }

    printf("Start maxpool reference implementation\n");

    arm_maxpool_q7_HWC_ref(img_in, POOL_IM_DIM, POOL_IM_CH, 3, 0, 2, POOL_IM_DIM / 2, (q7_t *) test2, pool_out_ref);

    // copy over the img input
    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        img_in[i] = test1[i];
    }

    printf("Start maxpool opt implementation\n");

    arm_maxpool_q7_HWC(img_in, POOL_IM_DIM, POOL_IM_CH, 3, 0, 2, POOL_IM_DIM / 2, (q7_t *) test2, pool_out_opt);

    verify_results_q7(pool_out_ref, pool_out_opt, POOL_IM_DIM / 2 * POOL_IM_DIM / 2 * POOL_IM_CH);

    // copy over the img input
    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        img_in[i] = test1[i];
    }

    // copy over the img input
    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        img_in[i] = test1[i];
    }

    printf("Start avepool ref implementation\n");

    arm_avepool_q7_HWC_ref(img_in, POOL_IM_DIM, POOL_IM_CH, 3, 0, 2, POOL_IM_DIM / 2, (q7_t *) test2, pool_out_ref);

    // copy over the img input
    for (int i = 0; i < POOL_IM_DIM * POOL_IM_DIM * POOL_IM_CH; i++)
    {
        img_in[i] = test1[i];
    }

    printf("Start avepool opt implementation\n");

    arm_avepool_q7_HWC(img_in, POOL_IM_DIM, POOL_IM_CH, 3, 0, 2, POOL_IM_DIM / 2, (q7_t *) test2, pool_out_opt);

    // special check here
    if_match = true;
    for (int i = 0; i < POOL_IM_DIM / 2 * POOL_IM_DIM / 2 * POOL_IM_CH; i++)
    {
        // we tolerate at most difference of 1 here because of rounding errors
        if (pool_out_ref[i] - pool_out_opt[i] >= 2 || pool_out_opt[i] - pool_out_ref[i] >= 2)
        {
            printf("Output mismatch at %d, expected %d, actual %d\n", i, pool_out_ref[i], pool_out_opt[i]);
            if_match = false;
        }
    }
    if (if_match == true)
    {
        printf("Outputs match.\n");
    }

    delete[]test1;
    delete[]test2;
    delete[]test3;

#endif

#ifdef TEST_RELU

#define RELU_DIM 127

    test1 = new q7_t[RELU_DIM];
    test2 = new q15_t[RELU_DIM];
    test3 = new q7_t[RELU_DIM];
    printf("malloc failed\n");
    test4 = new q15_t[RELU_DIM];

    for (int i = 0; i < RELU_DIM; i++)
    {
        test1[i] = (rand() % 256 - 128);
        test2[i] = (rand() % 65536 - 32768);
        test3[i] = test1[i];
        test4[i] = test2[i];
    }

    q7_t     *relu_ref_data_q7 = test1;
    q7_t     *relu_opt_data_q7 = test3;
    q15_t    *relu_ref_data_q15 = test2;
    q15_t    *relu_opt_data_q15 = test4;

    printf("Start ref relu q7 implementation\n");

    arm_relu_q7_ref(relu_ref_data_q7, RELU_DIM);

    printf("Start opt relu q7 implementation\n");

    arm_relu_q7(relu_opt_data_q7, RELU_DIM);

    verify_results_q7(relu_ref_data_q7, relu_opt_data_q7, RELU_DIM);

    printf("Start ref relu q15 implementation\n");

    arm_relu_q15_ref(relu_ref_data_q15, RELU_DIM);

    printf("Start opt relu q15 implementation\n");

    arm_relu_q15(relu_opt_data_q15, RELU_DIM);

    verify_results_q15(relu_ref_data_q15, relu_opt_data_q15, RELU_DIM);

    delete[]test1;
    delete[]test2;
    delete[]test3;
    delete[]test4;

#endif

#ifdef TEST_IP

#define IP_ROW_DIM 127
#define IP_COL_DIM 127

    q7_t      ip_weights[IP_ROW_DIM * IP_COL_DIM] = IP2_WEIGHT;
    q7_t      ip_q7_opt_weights[IP_ROW_DIM * IP_COL_DIM] = IP4_WEIGHT;
    q7_t      ip_q7_q15_opt_weights[IP_ROW_DIM * IP_COL_DIM] = IP4_q7_q15_WEIGHT;
    q15_t     ip_q15_weights[IP_ROW_DIM * IP_COL_DIM] = IP2_WEIGHT;
    q15_t     ip_q15_opt_weights[IP_ROW_DIM * IP_COL_DIM] = IP4_WEIGHT_Q15;

    test1 = new q7_t[IP_COL_DIM + IP_ROW_DIM];
    test2 = new q15_t[IP_COL_DIM];
    test3 = new q7_t[IP_ROW_DIM * 3];
    test4 = new q15_t[IP_COL_DIM + IP_ROW_DIM * 2];

    for (int i = 0; i < IP_ROW_DIM + IP_COL_DIM; i++)
    {
        test1[i] = rand() % 256 - 100;
    }
    for (int i = 0; i < IP_ROW_DIM * 3; i++)
    {
        test3[i] = 0;
    }

    q7_t     *ip_bias_q7 = test1 + IP_COL_DIM;

    q7_t     *ip_out_q7_ref = test3;
    q7_t     *ip_out_q7_opt = test3 + IP_ROW_DIM;
    q7_t     *ip_out_q7_opt_fast = test3 + 2 * IP_ROW_DIM;
    q15_t    *ip_out_q15_ref = test4 + IP_COL_DIM;
    q15_t    *ip_out_q15_opt = test4 + IP_COL_DIM + IP_ROW_DIM;

    printf("Start ref q7 implementation\n");

    arm_fully_connected_q7_ref(test1, ip_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7, ip_out_q7_ref, test2);

    printf("Start q7 implementation\n");

    arm_fully_connected_q7(test1, ip_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7, ip_out_q7_opt, test2);

    verify_results_q7(ip_out_q7_ref, ip_out_q7_opt, IP_ROW_DIM);

    printf("Start q7 ref opt implementation\n");

    arm_fully_connected_q7_opt_ref(test1, ip_q7_opt_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7,
                                   ip_out_q7_opt_fast, test2);

    verify_results_q7(ip_out_q7_ref, ip_out_q7_opt_fast, IP_ROW_DIM);

    printf("Start q7 opt implementation\n");

    arm_fully_connected_q7_opt(test1, ip_q7_opt_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7, ip_out_q7_opt_fast,
                               test2);

    verify_results_q7(ip_out_q7_ref, ip_out_q7_opt_fast, IP_ROW_DIM);

    for (int i = 0; i < IP_ROW_DIM + IP_COL_DIM; i++)
    {
        test4[i] = (rand() % 65536 - 32768);
    }

    printf("Start ref q15 implementation\n");

    arm_fully_connected_q15_ref(test4, ip_q15_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, test2, ip_out_q15_ref, NULL);

    printf("Start q15 implementation\n");

    arm_fully_connected_q15(test4, ip_q15_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, test2, ip_out_q15_opt, NULL);

    verify_results_q15(ip_out_q15_ref, ip_out_q15_ref, IP_ROW_DIM);

    printf("Start ref opt q15 implementation\n");

    arm_fully_connected_q15_opt_ref(test4, ip_q15_opt_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, test2, ip_out_q15_opt,
                                    NULL);

    verify_results_q15(ip_out_q15_ref, ip_out_q15_opt, IP_ROW_DIM);

    printf("Start opt q15 implementation\n");

    arm_fully_connected_q15_opt(test4, ip_q15_opt_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, test2, ip_out_q15_opt, NULL);

    verify_results_q15(ip_out_q15_ref, ip_out_q15_opt, IP_ROW_DIM);

    printf("Start ref q7_q15 implementation\n");

    arm_fully_connected_mat_q7_vec_q15_ref(test4, ip_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7, ip_out_q15_ref,
                                           test2);

    printf("Start q7_q15 implementation\n");

    arm_fully_connected_mat_q7_vec_q15(test4, ip_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7, ip_out_q15_opt,
                                       test2);

    verify_results_q15(ip_out_q15_ref, ip_out_q15_opt, IP_ROW_DIM);

    printf("Start ref opt q7_q15 implementation\n");

    arm_fully_connected_mat_q7_vec_q15_opt_ref(test4, ip_q7_q15_opt_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7,
                                               ip_out_q15_opt, test2);

    verify_results_q15(ip_out_q15_ref, ip_out_q15_opt, IP_ROW_DIM);

    printf("Start opt q7_q15 implementation\n");

    arm_fully_connected_mat_q7_vec_q15_opt(test4, ip_q7_q15_opt_weights, IP_COL_DIM, IP_ROW_DIM, 0, 7, ip_bias_q7,
                                           ip_out_q15_opt, test2);

    verify_results_q15(ip_out_q15_ref, ip_out_q15_opt, IP_ROW_DIM);

    delete[]test1;
    delete[]test2;
    delete[]test3;
    delete[]test4;

#endif

#ifdef TEST_CONV

#define CONV_IM_DIM 16
#define CONV_IM_CH 16
#define CONV_KER_DIM 5
#define CONV_OUT_CH 16
#define CONV_OUT_DIM 16

    test1 = new q7_t[CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH + CONV_OUT_CH];
    test2 =
        new q15_t[CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH +
                  2 * CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH + CONV_OUT_CH];
    test3 = new q7_t[CONV_IM_DIM * CONV_IM_DIM * CONV_IM_CH + 2 * CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH];
    test4 = new q15_t[CONV_IM_DIM * CONV_IM_DIM * CONV_IM_CH + 2 * CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH];

    for (int i = 0; i < CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH + CONV_OUT_CH; i++)
    {
        test1[i] = rand() % 256 - 100;
    }

    for (int i = 0;
         i <
         CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH +
         2 * CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH + CONV_OUT_CH; i++)
    {
        test2[i] = (rand() % 65536 - 32768);
    }

    q7_t     *conv_weight_q7 = test1;
    q7_t     *conv_bias_q7 = test1 + CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH;

    q15_t    *conv_weight_q15 = test2;
    q15_t    *conv_buf = test2 + CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH;
    q15_t    *conv_bias_q15 =
        test2 + CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH +
        2 * CONV_KER_DIM * CONV_KER_DIM * CONV_IM_CH * CONV_OUT_CH;

    q7_t     *conv_im_in_q7 = test3;
    q7_t     *conv_im_out_ref_q7 = test3 + CONV_IM_DIM * CONV_IM_DIM * CONV_IM_CH;
    q7_t     *conv_im_out_opt_q7 =
        test3 + CONV_IM_DIM * CONV_IM_DIM * CONV_IM_CH + CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH;

    q15_t    *conv_im_in_q15 = test4;
    q15_t    *conv_im_out_ref_q15 = test4 + CONV_IM_DIM * CONV_IM_DIM * CONV_IM_CH;
    q15_t    *conv_im_out_opt_q15 =
        test4 + CONV_IM_DIM * CONV_IM_DIM * CONV_IM_CH + CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH;

    printf("start q7 ref implementation\n");

    arm_convolve_HWC_q7_ref(conv_im_in_q7, CONV_IM_DIM, CONV_IM_CH, conv_weight_q7,
                            CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_ref_q7,
                            CONV_OUT_DIM, conv_buf, NULL);

    printf("start q7 basic implementation\n");

    arm_convolve_HWC_q7_basic(conv_im_in_q7, CONV_IM_DIM, CONV_IM_CH, conv_weight_q7,
                              CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_opt_q7,
                              CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q7(conv_im_out_ref_q7, conv_im_out_opt_q7, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    printf("start q7 fast implementation\n");

    arm_convolve_HWC_q7_fast(conv_im_in_q7, CONV_IM_DIM, CONV_IM_CH, conv_weight_q7,
                             CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_opt_q7,
                             CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q7(conv_im_out_ref_q7, conv_im_out_opt_q7, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    // testing with RGB
    printf("start q7 ref implementation for RGB\n");

    arm_convolve_HWC_q7_ref(conv_im_in_q7, CONV_IM_DIM, 3, conv_weight_q7,
                            CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_ref_q7,
                            CONV_OUT_DIM, conv_buf, NULL);

    printf("start q7 basic implementation for RGB\n");

    arm_convolve_HWC_q7_basic(conv_im_in_q7, CONV_IM_DIM, 3, conv_weight_q7,
                              CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_opt_q7,
                              CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q7(conv_im_out_ref_q7, conv_im_out_opt_q7, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    printf("start q7 RGB implementation for RGB\n");

    arm_convolve_HWC_q7_RGB(conv_im_in_q7, CONV_IM_DIM, 3, conv_weight_q7,
                            CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_opt_q7,
                            CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q7(conv_im_out_ref_q7, conv_im_out_opt_q7, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    // testing q15

    printf("start q15 ref implementation\n");

    arm_convolve_HWC_q15_ref(conv_im_in_q15, CONV_IM_DIM, CONV_IM_CH, conv_weight_q15,
                             CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q15, 0, 15, conv_im_out_ref_q15,
                             CONV_OUT_DIM, conv_buf, NULL);

    printf("start q15 basic implementation\n");

    arm_convolve_HWC_q15_basic(conv_im_in_q15, CONV_IM_DIM, CONV_IM_CH, conv_weight_q15,
                               CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q15, 0, 15, conv_im_out_opt_q15,
                               CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q15(conv_im_out_ref_q15, conv_im_out_opt_q15, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    printf("start q15 fast implementation\n");

    arm_convolve_HWC_q15_fast(conv_im_in_q15, CONV_IM_DIM, CONV_IM_CH, conv_weight_q15,
                              CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q15, 0, 15, conv_im_out_opt_q15,
                              CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q15(conv_im_out_ref_q15, conv_im_out_opt_q15, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    // depthwise separable conv
    printf("start q7 depthwise_separable_conv ref implementation\n");

    arm_depthwise_separable_conv_HWC_q7_ref(conv_im_in_q7, CONV_IM_DIM, CONV_IM_CH, conv_weight_q7,
                                            CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_ref_q7,
                                            CONV_OUT_DIM, conv_buf, NULL);

    printf("start q7 depthwise_separable_conv implementation\n");

    arm_depthwise_separable_conv_HWC_q7(conv_im_in_q7, CONV_IM_DIM, CONV_IM_CH, conv_weight_q7,
                                        CONV_OUT_CH, CONV_KER_DIM, 2, 1, conv_bias_q7, 0, 7, conv_im_out_opt_q7,
                                        CONV_OUT_DIM, conv_buf, NULL);

    verify_results_q7(conv_im_out_ref_q7, conv_im_out_opt_q7, CONV_OUT_DIM * CONV_OUT_DIM * CONV_OUT_CH);

    delete[]test1;
    delete[]test2;
    delete[]test3;
    delete[]test4;

#endif

    return 0;
}
