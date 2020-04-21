#include "Test.h"
#include "Pattern.h"
#include "arm_math_f16.h"
class TransformCF16:public Client::Suite
    {
        public:
            TransformCF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformCF16_decl.h"
            
            Client::Pattern<float16_t> input;
            Client::LocalPattern<float16_t> outputfft;

            Client::RefPattern<float16_t> ref;

            arm_cfft_instance_f16 varInstCfftF16;

            int ifft;

            arm_status status;
            
    };
