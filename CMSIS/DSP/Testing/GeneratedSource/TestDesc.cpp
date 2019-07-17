#include "Test.h"

#include "BasicTestsF32.h"
    BasicTestsF32::BasicTestsF32(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&BasicTestsF32::test_add_f32);
this->addTest(2,(Client::test)&BasicTestsF32::test_add_f32);
this->addTest(3,(Client::test)&BasicTestsF32::test_add_f32);
this->addTest(4,(Client::test)&BasicTestsF32::test_sub_f32);
this->addTest(5,(Client::test)&BasicTestsF32::test_sub_f32);
this->addTest(6,(Client::test)&BasicTestsF32::test_sub_f32);
this->addTest(7,(Client::test)&BasicTestsF32::test_mult_f32);
this->addTest(8,(Client::test)&BasicTestsF32::test_mult_f32);
this->addTest(9,(Client::test)&BasicTestsF32::test_mult_f32);
this->addTest(10,(Client::test)&BasicTestsF32::test_negate_f32);
this->addTest(11,(Client::test)&BasicTestsF32::test_negate_f32);
this->addTest(12,(Client::test)&BasicTestsF32::test_negate_f32);
this->addTest(13,(Client::test)&BasicTestsF32::test_offset_f32);
this->addTest(14,(Client::test)&BasicTestsF32::test_offset_f32);
this->addTest(15,(Client::test)&BasicTestsF32::test_offset_f32);
this->addTest(16,(Client::test)&BasicTestsF32::test_scale_f32);
this->addTest(17,(Client::test)&BasicTestsF32::test_scale_f32);
this->addTest(18,(Client::test)&BasicTestsF32::test_scale_f32);
this->addTest(19,(Client::test)&BasicTestsF32::test_dot_prod_f32);
this->addTest(20,(Client::test)&BasicTestsF32::test_dot_prod_f32);
this->addTest(21,(Client::test)&BasicTestsF32::test_dot_prod_f32);
this->addTest(22,(Client::test)&BasicTestsF32::test_abs_f32);
this->addTest(23,(Client::test)&BasicTestsF32::test_abs_f32);
this->addTest(24,(Client::test)&BasicTestsF32::test_abs_f32);

    }

#include "SVMF32.h"
    SVMF32::SVMF32(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&SVMF32::test_svm_linear_predict_f32);
this->addTest(2,(Client::test)&SVMF32::test_svm_polynomial_predict_f32);
this->addTest(3,(Client::test)&SVMF32::test_svm_rbf_predict_f32);
this->addTest(4,(Client::test)&SVMF32::test_svm_sigmoid_predict_f32);
this->addTest(5,(Client::test)&SVMF32::test_svm_rbf_predict_f32);

    }

#include "BasicMathsBenchmarksF32.h"
    BasicMathsBenchmarksF32::BasicMathsBenchmarksF32(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&BasicMathsBenchmarksF32::vec_mult_f32);
this->addTest(2,(Client::test)&BasicMathsBenchmarksF32::vec_add_f32);
this->addTest(3,(Client::test)&BasicMathsBenchmarksF32::vec_sub_f32);
this->addTest(4,(Client::test)&BasicMathsBenchmarksF32::vec_abs_f32);
this->addTest(5,(Client::test)&BasicMathsBenchmarksF32::vec_negate_f32);
this->addTest(6,(Client::test)&BasicMathsBenchmarksF32::vec_offset_f32);
this->addTest(7,(Client::test)&BasicMathsBenchmarksF32::vec_scale_f32);
this->addTest(8,(Client::test)&BasicMathsBenchmarksF32::vec_dot_f32);

    }

#include "BasicMathsBenchmarksQ31.h"
    BasicMathsBenchmarksQ31::BasicMathsBenchmarksQ31(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&BasicMathsBenchmarksQ31::vec_mult_q31);
this->addTest(2,(Client::test)&BasicMathsBenchmarksQ31::vec_add_q31);
this->addTest(3,(Client::test)&BasicMathsBenchmarksQ31::vec_sub_q31);
this->addTest(4,(Client::test)&BasicMathsBenchmarksQ31::vec_abs_q31);
this->addTest(5,(Client::test)&BasicMathsBenchmarksQ31::vec_negate_q31);
this->addTest(6,(Client::test)&BasicMathsBenchmarksQ31::vec_offset_q31);
this->addTest(7,(Client::test)&BasicMathsBenchmarksQ31::vec_scale_q31);
this->addTest(8,(Client::test)&BasicMathsBenchmarksQ31::vec_dot_q31);

    }

#include "BasicMathsBenchmarksQ15.h"
    BasicMathsBenchmarksQ15::BasicMathsBenchmarksQ15(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&BasicMathsBenchmarksQ15::vec_mult_q15);
this->addTest(2,(Client::test)&BasicMathsBenchmarksQ15::vec_add_q15);
this->addTest(3,(Client::test)&BasicMathsBenchmarksQ15::vec_sub_q15);
this->addTest(4,(Client::test)&BasicMathsBenchmarksQ15::vec_abs_q15);
this->addTest(5,(Client::test)&BasicMathsBenchmarksQ15::vec_negate_q15);
this->addTest(6,(Client::test)&BasicMathsBenchmarksQ15::vec_offset_q15);
this->addTest(7,(Client::test)&BasicMathsBenchmarksQ15::vec_scale_q15);
this->addTest(8,(Client::test)&BasicMathsBenchmarksQ15::vec_dot_q15);

    }

#include "BasicMathsBenchmarksQ7.h"
    BasicMathsBenchmarksQ7::BasicMathsBenchmarksQ7(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&BasicMathsBenchmarksQ7::vec_mult_q7);
this->addTest(2,(Client::test)&BasicMathsBenchmarksQ7::vec_add_q7);
this->addTest(3,(Client::test)&BasicMathsBenchmarksQ7::vec_sub_q7);
this->addTest(4,(Client::test)&BasicMathsBenchmarksQ7::vec_abs_q7);
this->addTest(5,(Client::test)&BasicMathsBenchmarksQ7::vec_negate_q7);
this->addTest(6,(Client::test)&BasicMathsBenchmarksQ7::vec_offset_q7);
this->addTest(7,(Client::test)&BasicMathsBenchmarksQ7::vec_scale_q7);
this->addTest(8,(Client::test)&BasicMathsBenchmarksQ7::vec_dot_q7);

    }

#include "FullyConnected.h"
    FullyConnected::FullyConnected(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(2,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(3,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(4,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(5,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(6,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(7,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(8,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(9,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(10,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(11,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(12,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(13,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(14,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(15,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);

    }

#include "FullyConnectedBench.h"
    FullyConnectedBench::FullyConnectedBench(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&FullyConnectedBench::test_fully_connected_tflite_s8);

    }
