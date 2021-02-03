#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class TransformCF32:public Client::Suite
    {
        public:
            TransformCF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformCF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::LocalPattern<float32_t> outputfft;

            Client::RefPattern<float32_t> ref;

            arm_cfft_instance_f32 varInstCfftF32;

            int ifft;

            arm_status status;
            
    };
