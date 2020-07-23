#include "Test.h"
#include "Pattern.h"

#include "dsp/controller_functions.h"

class ControllerF32:public Client::Suite
    {
        public:
            ControllerF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ControllerF32_decl.h"
            Client::Pattern<float32_t> samples;

            Client::LocalPattern<float32_t> output;
            
            int nbSamples;

            arm_pid_instance_f32  instPid;
            float32_t *pSrc;
            float32_t *pDst;
            
            
    };
