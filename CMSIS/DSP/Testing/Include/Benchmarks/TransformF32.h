#include "Test.h"
#include "Pattern.h"
class TransformF32:public Client::Suite
    {
        public:
            TransformF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformF32_decl.h"
            Client::Pattern<float32_t> samples;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> state;
            
            int nbSamples;
            int ifft;
            int bitRev;

            float32_t *pSrc;
            float32_t *pDst;
            float32_t *pState;

            const arm_cfft_instance_f32 *cfftInstance;
            arm_rfft_fast_instance_f32 rfftFastInstance;

            arm_dct4_instance_f32 dct4Instance;
            arm_rfft_instance_f32 rfftInstance;
            arm_cfft_radix4_instance_f32 cfftRadix4Instance;
            arm_cfft_radix2_instance_f32 cfftRadix2Instance;
            
    };
