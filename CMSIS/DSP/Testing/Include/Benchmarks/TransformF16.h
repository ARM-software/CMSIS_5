#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions_f16.h"

class TransformF16:public Client::Suite
    {
        public:
            TransformF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformF16_decl.h"
            Client::Pattern<float16_t> samples;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<float16_t> tmp;
            Client::LocalPattern<float16_t> state;
            
            int nbSamples;
            int ifft;
            int bitRev;

            float16_t *pSrc;
            float16_t *pDst;
            float16_t *pState;
            float16_t *pTmp;

            arm_cfft_instance_f16 cfftInstance;
            arm_rfft_fast_instance_f16 rfftFastInstance;

            arm_status status;

            arm_cfft_radix4_instance_f16 cfftRadix4Instance;
            arm_cfft_radix2_instance_f16 cfftRadix2Instance;
            
    };
