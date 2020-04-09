#include "Test.h"
#include "Pattern.h"
class FIRF32:public Client::Suite
    {
        public:
            FIRF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRF32_decl.h"
            Client::Pattern<float32_t> coefs;
            Client::Pattern<float32_t> samples;
            Client::Pattern<float32_t> refs;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> error;
            Client::LocalPattern<float32_t> state;

            int nbTaps;
            int nbSamples;

            arm_fir_instance_f32  instFir;
            arm_lms_instance_f32  instLms;
            arm_lms_norm_instance_f32 instLmsNorm;

            const float32_t *pSrc;
            const float32_t *pCoefs;
            float32_t *pDst;
            const float32_t *pRef;
            float32_t *pErr;
            
    };
