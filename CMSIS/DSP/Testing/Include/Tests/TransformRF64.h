#include "Test.h"
#include "Pattern.h"
class TransformRF64:public Client::Suite
    {
        public:
            TransformRF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformRF64_decl.h"
            
            Client::Pattern<float64_t> input;
            Client::LocalPattern<float64_t> outputfft;
            Client::LocalPattern<float64_t> inputchanged;

            Client::RefPattern<float64_t> ref;

            arm_rfft_fast_instance_f64 instRfftF64;

            int ifft;
            
    };
