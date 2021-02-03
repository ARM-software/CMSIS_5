#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class TransformCF64:public Client::Suite
    {
        public:
            TransformCF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformCF64_decl.h"
            
            Client::Pattern<float64_t> input;
            Client::LocalPattern<float64_t> outputfft;

            Client::RefPattern<float64_t> ref;

            arm_cfft_instance_f64 varInstCfftF64;

            int ifft;

            arm_status status;
            
    };
