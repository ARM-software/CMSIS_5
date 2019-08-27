# CMSIS NN

## About
The table below gives  a quick overview of the API's in CMSIS-NN int8 library with symmetric quantization.

**Note:** The GitHub documentation does not follow the *develop* branch but rather the last official release in the *master* branch. Consequently, the group documentation linked to in the table table might not have the listed API. Please refer to the description in the [header](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Include/arm_nnfunctions.h) file instead.


Group | API | Base Operator | Input Constraints | Additional memory required for <br/> DSP optimizations (bytes) | DSP Optimized |  MVE Optimized | Other comments |
|:----| :---| :------------ | :---------------- | :--------------------------------------------------------| :-------------| :------------- | :------------- |
|[Conv](https://arm-software.github.io/CMSIS_5/NN/html/group__NNConv.html)||||| |  ||
||arm_convolve_s8()|CONV|dilation = 1|4 * ker_x * ker_y * input_ch| Yes | No ||
||arm_convolve_1x1_s8_fast() | CONV | dilation = 1 <br/> ker_x = 1, ker_y = 1 <br/> pad = 0<br/> stride = 1| 4 * input_ch | Yes |No ||
| | arm_depthwise_conv_s8() | DEPTHWISE_CONV | dilation = 1  | No|No|No||
|| arm_depthwise_conv_s8_opt()| DEPTHWISE_CONV | dilation = 1 <br/> depth_multiplier = 1 | 4 * ker_x * ker_y * input_ch | Yes| No| Best case is when channels are multiple of 4 or <br/>at the least >= 4 |
|[Fully Connected](https://arm-software.github.io/CMSIS_5/NN/html/group__FC.html)||||| |  | |
|| arm_fully_connected_s8() |FULLY CONNECTED & <br/> MAT MUL  | None | column length * 2 | Yes | No | |
|[Pooling](https://arm-software.github.io/CMSIS_5/NN/html/group__Pooling.html)||||| |  ||
|| arm_avgpool_s8() | AVERAGE POOL | None | input_ch * output_x * 2 | Yes| No| Best case case is when channels are multiple of 4 or <br/> at the least >= 4 |
|[Misc](https://arm-software.github.io/CMSIS_5/NN/html/group__groupNN.html)||||| |  ||
||arm_softmax_q7()| SOFTMAX | None | None | Yes | No | Not bit exact to TFLu but can be up to 70x faster |
||arm_elementwise_add_s8()| ELEMENTWISE ADD | None | None | Yes| No| Reshape is not done in this function <br/> Only minor improvements are expected |
||arm_elementwise_mul_s8()| ELEMENTWISE MUL | None | None | Yes| No| Reshape is not done in this function <br/> Only minor improvements are expected |
||arm_relu_q7() | RELU | None | None | Yes| No|
| |  |  | | | ||

