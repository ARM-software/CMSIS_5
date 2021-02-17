# CMSIS NN

## About
This page  give a quick overview of the functions available and key differences between them.

**Note:** The GitHub documentation does not follow the *develop* branch but rather the last official release in the *master* branch. Consequently, the group documentation linked to in the table table might not have the listed API. Please refer to the description in the [header](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Include/arm_nnfunctions.h) file instead.

## Support / Contact
For any questions or to reach the CMSIS-NN team, please create a new issue in https://github.com/ARM-software/CMSIS_5/issues

## Legacy vs TFL micro compliant APIs
There are two kinds of APIs available in the CMSIS-NN repository; One that supports a legacy symmetric quantization scheme[1] and one that supports TFL micro's symmetric quantization scheme. One of the main differences is how the quantization is performed. The legacy APIs have a fixed point format with power of 2 scaling. This simplifies the re-quantization to a cycle efficient shift operation. No new development is done on the legacy functions and all of the new development is on the functions that support TFL micro. The table below highlights some of the differences between the two formats for convolution related functions. The TFL micro compliant APIs in most cases have a _s8 suffix and is always specified in the API header file.

Operation | Legacy APIs | TFL micro compliant APIs|
|:-----------|:---------------------|:-------------|
Core loop | No input or filter offset | Input and/or filter offset |
Re-quantization | Shift and saturate in one instruction. ~ 5 cycles | Greater than 200 cycles for one output element
Quantization | Per layer quantization | Per-channel quantization
Output offset | No | Per-layer output offset
Fused Activation | No | Yes

## TFL micro compliant APIs
Group | API | Base Operator | Input Constraints | Additional memory required for <br/> optimizations (bytes) | DSP Optimized |  MVE Optimized | Other comments |
|:----| :---| :------------ | :---------------- | :--------------------------------------------------------| :-------------| :------------- | :------------- |
|[Conv](https://arm-software.github.io/CMSIS_5/NN/html/group__NNConv.html)||||| |  ||
||arm_convolve_wrapper_s8()|CONV|dilation = 1|n.a.| Yes | Yes |The additional memory required depends on the optimal convolution function called|
||arm_convolve_s8()|CONV|dilation = 1|4 * ker_x * ker_y * input_ch| Yes | Yes ||
||arm_convolve_1x1_s8_fast() | CONV | dilation = 1 <br/> ker_x = 1, ker_y = 1 <br/> pad = 0<br/> stride = 1<br/> input_ch % 4 = 0| 0 | Yes |Yes ||
||arm_convolve_1_n_s8() | CONV | dilation = 1 <br/> output_y % 4 = 0 | No |Yes ||
|| arm_depthwise_conv_3x3_s8() | DEPTHWISE_CONV | dilation = 1 <br/> depth_multiplier = 1 <br/> pad_x <= 1 | No|No|No| Preferred function for 3x3 kernel size for DSP extension. </br> For MVE, use arm_depthwise_conv_s8_opt()||
| | arm_depthwise_conv_s8() | DEPTHWISE_CONV | dilation = 1  | No|No|No||
|| arm_depthwise_conv_s8_opt()| DEPTHWISE_CONV | dilation = 1 <br/> depth_multiplier = 1 | DSP: 2 * ker_x * ker_y * input_ch <br/> MVE: 2 * DSP + 4 | Yes| Yes| Best case is when channels are multiple of 4 or <br/>at the least >= 4 |
|[Fully Connected](https://arm-software.github.io/CMSIS_5/NN/html/group__FC.html)||||| |  | |
|| arm_fully_connected_s8() |FULLY CONNECTED & <br/> MAT MUL  | None | 0 | Yes | Yes | |
|[Pooling](https://arm-software.github.io/CMSIS_5/NN/html/group__Pooling.html)||||| |  ||
|| arm_avgpool_s8() | AVERAGE POOL | None | input_ch * 2<br/>(DSP only) | Yes| Yes| Best case case is when channels are multiple of 4 or <br/> at the least >= 4 |
|| arm_maxpool_s8() | MAX POOL | None | None | Yes| Yes|  |
|[Softmax](https://arm-software.github.io/CMSIS_5/NN/html/group__Softmax.html)||||| |  ||
||arm_softmax_q7()| SOFTMAX | None | None | Yes | No | Not bit exact to TFLu but can be up to 70x faster |
||arm_softmax_s8()| SOFTMAX | None | None | No | Yes | Bit exact to TFLu |
||arm_softmax_u8()| SOFTMAX | None | None | No | No | Bit exact to TFLu |
|[SVDF](https://arm-software.github.io/CMSIS_5/NN/html/group__SVDF.html)||||| |  ||
||arm_svdf_s8()| SVDF | None | None | Yes | No | Bit exact to TFLu |
|[Misc](https://arm-software.github.io/CMSIS_5/NN/html/group__groupNN.html)||||| |  ||
||arm_reshape_s8()| SOFTMAX | None | None | No | No | |
||arm_elementwise_add_s8()| ELEMENTWISE ADD | None | None | Yes| Yes| Reshape is not done in this function <br/> Only minor improvements are expected |
||arm_elementwise_mul_s8()| ELEMENTWISE MUL | None | None | Yes| Yes| Reshape is not done in this function <br/> Only minor improvements are expected |
||arm_relu_q7() | RELU | None | None | Yes| No|
||arm_relu6_s8() | RELU | None | None | Yes| No|
|[Concat](https://arm-software.github.io/CMSIS_5/NN/html/group__groupNN.html)||||| |  ||
||arm_concatenation_s8_w() | CONCAT | None | None | No| No||
||arm_concatenation_s8_x() | CONCAT | None | None | No| No||
||arm_concatenation_s8_y() | CONCAT | None | None | No| No||
||arm_concatenation_s8_z() | CONCAT | None | None | No| No||


## Reference
[1] Legacy CMSIS-NN and how to use it https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn/single-page
