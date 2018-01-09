/*
 * Copyright (C) 2010-2017 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS-NN
 * Title:        arm_nnexamples_gru.cpp
 * Description:	 Gated Recurrent Unit Example
 *
 * Target Processor: Cortex-M4 and Cortex-M7 cores
 *
 * -------------------------------------------------------------------- */

/**
 * @ingroup groupExamples
 */

/**
 * @defgroup GRUExample Gated Recurrent Unit Example
 *
 * \par Description:
 * \par
 * Demonstrates a gated recurrent unit (GRU) example with the use of fully-connected,
 * Tanh/Sigmoid activation functions.
 *
 * \par Model definition:
 * \par
 * GRU is a type of recurrent neural network (RNN). It contains two sigmoid gates and one hidden
 * state. 
 * \par
 * The computation can be summarized as:
 * <pre>z[t] = sigmoid( W_z &middot; {h[t-1],x[t]} )
 * r[t] = sigmoid( W_r &middot; {h[t-1],x[t]} ) 
 * n[t] = tanh( W_n &middot; [r[t] &times; h[t-1], x[t] ) 
 * h[t] = (1 - z[t]) &times; h[t-1] + z[t] &times; n[t] </pre>
 * \image html GRU.gif "Gate Recurrent Unit Diagram"
 *
 * \par Variables Description:
 * \par
 * \li \c update_gate_weights, \c reset_gate_weights, \c hidden_state_weights are weights corresponding to update gate (W_z), reset gate (W_r), and hidden state (W_n).
 * \li \c update_gate_bias, \c reset_gate_bias, \c hidden_state_bias are layer bias arrays
 * \li \c test_input1, \c test_input2, \c test_history are the inputs and initial history
 *
 * \par
 * The buffer is allocated as:
 * \par
 * | reset | input | history | update | hidden_state |
 * \par
 * In this way, the concatenation is automatically done since (reset, input) and (input, history)
 * are physically concatinated in memory.
 * \par
 *  The ordering of the weight matrix should be adjusted accordingly.
 *
  *
 * 
 * \par CMSIS DSP Software Library Functions Used:
 * \par
 * - arm_fully_connected_mat_q7_vec_q15_opt()
 * - arm_nn_activations_direct_q15()
 * - arm_mult_q15()
 * - arm_offset_q15()
 * - arm_sub_q15()
 * - arm_copy_q15()
 *
 * <b> Refer  </b>
 * \link arm_nnexamples_gru.cpp \endlink
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_data.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
 
#define DIM_HISTORY 32
#define DIM_INPUT 32
#define DIM_VEC 64

#define USE_X4

#ifndef USE_X4
static q7_t update_gate_weights[DIM_VEC * DIM_HISTORY] = UPDATE_GATE_WEIGHT_X2;
static q7_t reset_gate_weights[DIM_VEC * DIM_HISTORY] = RESET_GATE_WEIGHT_X2;
static q7_t hidden_state_weights[DIM_VEC * DIM_HISTORY] = HIDDEN_STATE_WEIGHT_X2;
#else
static q7_t update_gate_weights[DIM_VEC * DIM_HISTORY] = UPDATE_GATE_WEIGHT_X4;
static q7_t reset_gate_weights[DIM_VEC * DIM_HISTORY] = RESET_GATE_WEIGHT_X4;
static q7_t hidden_state_weights[DIM_VEC * DIM_HISTORY] = HIDDEN_STATE_WEIGHT_X4;
#endif

static q7_t update_gate_bias[DIM_HISTORY] = UPDATE_GATE_BIAS;
static q7_t reset_gate_bias[DIM_HISTORY] = RESET_GATE_BIAS;
static q7_t hidden_state_bias[DIM_HISTORY] = HIDDEN_STATE_BIAS;

static q15_t test_input1[DIM_INPUT] = INPUT_DATA1;
static q15_t test_input2[DIM_INPUT] = INPUT_DATA2;
static q15_t test_history[DIM_HISTORY] = HISTORY_DATA;


q15_t     scratch_buffer[DIM_HISTORY * 4 + DIM_INPUT];

void gru_example(q15_t * scratch_input, uint16_t input_size, uint16_t history_size,
                 q7_t * weights_update, q7_t * weights_reset, q7_t * weights_hidden_state,
                 q7_t * bias_update, q7_t * bias_reset, q7_t * bias_hidden_state)
{

    q15_t    *reset = scratch_input;
    q15_t    *input = scratch_input + history_size;
    q15_t    *history = scratch_input + history_size + input_size;
    q15_t    *update = scratch_input + 2 * history_size + input_size;
    q15_t    *hidden_state = scratch_input + 3 * history_size + input_size;

    // reset gate calculation
    // the range of the output can be adjusted with bias_shift and output_shift
#ifndef USE_X4
    arm_fully_connected_mat_q7_vec_q15(input, weights_reset, input_size + history_size, history_size, 0, 15, bias_reset,
                                       reset, NULL);
#else
    arm_fully_connected_mat_q7_vec_q15_opt(input, weights_reset, input_size + history_size, history_size, 0, 15,
                                           bias_reset, reset, NULL);
#endif
    // sigmoid function, the size of the integer bit-width should be consistent with out_shift
    arm_nn_activations_direct_q15(reset, history_size, 0, ARM_SIGMOID);
    arm_mult_q15(history, reset, reset, history_size);

    // update gate calculation
    // the range of the output can be adjusted with bias_shift and output_shift
#ifndef USE_X4
    arm_fully_connected_mat_q7_vec_q15(input, weights_update, input_size + history_size, history_size, 0, 15,
                                       bias_update, update, NULL);
#else
    arm_fully_connected_mat_q7_vec_q15_opt(input, weights_update, input_size + history_size, history_size, 0, 15,
                                           bias_update, update, NULL);
#endif

    // sigmoid function, the size of the integer bit-width should be consistent with out_shift
    arm_nn_activations_direct_q15(update, history_size, 0, ARM_SIGMOID);

    // hidden state calculation
#ifndef USE_X4
    arm_fully_connected_mat_q7_vec_q15(reset, weights_hidden_state, input_size + history_size, history_size, 0, 15,
                                       bias_hidden_state, hidden_state, NULL);
#else
    arm_fully_connected_mat_q7_vec_q15_opt(reset, weights_hidden_state, input_size + history_size, history_size, 0, 15,
                                           bias_hidden_state, hidden_state, NULL);
#endif

    // tanh function, the size of the integer bit-width should be consistent with out_shift
    arm_nn_activations_direct_q15(hidden_state, history_size, 0, ARM_TANH);
    arm_mult_q15(update, hidden_state, hidden_state, history_size);

    // we calculate z - 1 here
    // so final addition becomes substraction
    arm_offset_q15(update, 0x8000, update, history_size);
    // multiply history
    arm_mult_q15(history, update, update, history_size);
    // calculate history_out
    arm_sub_q15(hidden_state, update, history, history_size);

    return;
}

int main()
{

    int       input_size = DIM_INPUT;
    int       history_size = DIM_HISTORY;

    // copy over the input data 
    arm_copy_q15(test_input1, scratch_buffer + history_size, input_size);
    arm_copy_q15(test_history, scratch_buffer + history_size + input_size, history_size);

    gru_example(scratch_buffer, input_size, history_size,
                update_gate_weights, reset_gate_weights, hidden_state_weights,
                update_gate_bias, reset_gate_bias, hidden_state_bias);

    arm_copy_q15(test_input2, scratch_buffer + history_size, input_size);
    gru_example(scratch_buffer, input_size, history_size,
                update_gate_weights, reset_gate_weights, hidden_state_weights,
                update_gate_bias, reset_gate_bias, hidden_state_bias);

    return 0;
}
