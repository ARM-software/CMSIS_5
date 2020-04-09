#include "FullyConnectedBench.h"
#include "Error.h"
#include "arm_nnfunctions.h"
   
    void FullyConnectedBench::test_fully_connected_tflite_s8()
    {
      
      for(int i=0; i < this->repeatNb; i++)
       {
          arm_fully_connected_s8((int8_t*)this->inp
            ,(const int8_t*)this->weightp
            ,colDim
            ,rowDim
            ,nb_batches
            ,input_offset
            ,filter_offset
            ,output_mult
            ,output_shift
            ,output_offset
            ,(const int32_t*)this->biasp
            ,(int8_t*)this->outp 
            ,act_min 
            ,act_max
            ,this->tempp
            );
       }
      
    } 

    
    void FullyConnectedBench::setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->repeatNb = *it;
       

        output_mult = 1077969154;
        output_shift = 2;
        filter_offset = 0;
        input_offset = 0;
        output_offset = 1;
        act_min =-128;
        act_max= 127;

          
        nb_batches=8;

        colDim=8;
        rowDim=5;

        input.reload(FullyConnectedBench::INPUT13_S8_ID,mgr);
        bias.reload(FullyConnectedBench::BIAS13_S8_ID,mgr);
        weight.reload(FullyConnectedBench::WEIGHT13_S8_ID,mgr);

        //ref.reload(FullyConnectedBench::REF13_S8_ID,mgr);

        output.create(ref.nbSamples(),FullyConnectedBench::OUTPUT_S8_ID,mgr);
        temp.create(colDim,FullyConnectedBench::TEMP_S16_ID,mgr);

        this->inp=input.ptr();
        this->biasp=bias.ptr();
        this->weightp=weight.ptr();
        this->outp=output.ptr();
        //this->refp=ref.ptr();
        this->tempp=temp.ptr();
    }

    void FullyConnectedBench::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       
    }
