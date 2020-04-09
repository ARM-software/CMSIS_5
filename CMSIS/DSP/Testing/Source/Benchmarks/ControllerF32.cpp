#include "ControllerF32.h"
#include "Error.h"

   
    void ControllerF32::test_pid_f32()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_pid_f32(&instPid, *this->pSrc++);
       }
    } 

    void ControllerF32::test_clarke_f32() 
    {
       float32_t Ialpha;
       float32_t Ibeta;
       for(int i=0; i < this->nbSamples; i++)
       {
         arm_clarke_f32(0.1,0.2,&Ialpha,&Ibeta);
       }
    }

    void ControllerF32::test_inv_clarke_f32() 
    {
       float32_t Ia;
       float32_t Ib;
       for(int i=0; i < this->nbSamples; i++)
       {
         arm_clarke_f32(0.1,0.2,&Ia,&Ib);
       }
    }

    void ControllerF32::test_park_f32() 
    {
       float32_t Id,Iq;

       for(int i=0; i < this->nbSamples; i++)
       {
          arm_park_f32(0.1,0.2,&Id,&Iq,0.1,0.2);
       }
    }

    void ControllerF32::test_inv_park_f32() 
    {
        float32_t Ialpha,Ibeta;
        
        for(int i=0; i < this->nbSamples; i++)
        {
           arm_inv_park_f32(0.1,0.2,&Ialpha,&Ibeta,0.1,0.2);
        }
    }

    void ControllerF32::test_sin_cos_f32() 
    {
        float32_t sinVal,cosVal;
        
        for(int i=0; i < this->nbSamples; i++)
        {
           arm_sin_cos_f32(0.1,&sinVal,&cosVal);
        }
    }
    
    void ControllerF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       samples.reload(ControllerF32::SAMPLES_F32_ID,mgr,this->nbSamples);
       output.create(this->nbSamples,ControllerF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
           case TEST_PID_F32_1:
              arm_pid_init_f32(&instPid,1);
           break;

       }

       this->pSrc=samples.ptr();
      this->pDst=output.ptr();
       
    }

    void ControllerF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
