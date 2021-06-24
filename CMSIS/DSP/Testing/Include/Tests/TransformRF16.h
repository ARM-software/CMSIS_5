#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions_f16.h"

class TransformRF16:public Client::Suite
    {
        public:
            TransformRF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformRF16_decl.h"
            
            Client::Pattern<float16_t> input;
            Client::LocalPattern<float16_t> outputfft;
            Client::LocalPattern<float16_t> inputchanged;

            Client::RefPattern<float16_t> ref;

            arm_rfft_fast_instance_f16 instRfftF16;

            int ifft;
            
    };
