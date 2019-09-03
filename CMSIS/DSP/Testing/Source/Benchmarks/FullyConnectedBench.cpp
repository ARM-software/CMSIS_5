#include "FullyConnectedBench.h"
#include "Error.h"
#include "arm_nnfunctions.h"
   
    void FullyConnectedBench::test_fully_connected_tflite_s8()
    {
      

       int32_t output_mult = 1073741824;
       int16_t output_shift = -1;
       int32_t filter_offset = 1;
       int32_t input_offset = 1;
       int32_t output_offset = -1;



       for(int i=0; i < this->repeatNb; i++)
       {
          arm_fully_connected_s8((int8_t*)inp
           ,(const int8_t*)weightp
           ,input.nbSamples()
           ,output.nbSamples()
           ,1
           ,input_offset
           ,filter_offset
           ,output_mult
           ,output_shift
           ,output_offset
           ,(const int32_t*)biasp
           ,(int8_t*)outp 
           ,-128 
           ,127 
           ,tempp
           );
       }
      
    } 

    
    void FullyConnectedBench::setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->repeatNb = *it;
       

       input.reload(FullyConnectedBench::INPUT1_S8_ID,mgr);
       bias.reload(FullyConnectedBench::BIAS1_S8_ID,mgr);
       weight.reload(FullyConnectedBench::WEIGHT1_S8_ID,mgr);

       ref.reload(FullyConnectedBench::REF1_S8_ID,mgr);

       
       output.create(ref.nbSamples(),FullyConnectedBench::OUTPUT_S8_ID,mgr);
       temp.create(input.nbSamples(),FullyConnectedBench::TEMP_S16_ID,mgr);

       inp=input.ptr();
       biasp=bias.ptr();
       weightp=weight.ptr();
       outp=output.ptr();
       refp=ref.ptr();
       tempp=temp.ptr();

    }

    void FullyConnectedBench::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       
    }
