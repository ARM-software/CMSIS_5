#include "Test.h"
#include "Pattern.h"
class TransformRF32:public Client::Suite
    {
        public:
            TransformRF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformRF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::LocalPattern<float32_t> outputfft;
            Client::LocalPattern<float32_t> inputchanged;

            Client::RefPattern<float32_t> ref;

            arm_rfft_fast_instance_f32 instRfftF32;

            int ifft;
            
    };
