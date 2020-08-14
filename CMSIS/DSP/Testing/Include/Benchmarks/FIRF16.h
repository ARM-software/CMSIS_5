#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions_f16.h"

class FIRF16:public Client::Suite
    {
        public:
            FIRF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRF16_decl.h"
            Client::Pattern<float16_t> coefs;
            Client::Pattern<float16_t> samples;
            Client::Pattern<float16_t> refs;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<float16_t> error;
            Client::LocalPattern<float16_t> state;

            int nbTaps;
            int nbSamples;

            arm_fir_instance_f16  instFir;

            const float16_t *pSrc;
            const float16_t *pCoefs;
            float16_t *pDst;
            const float16_t *pRef;
            float16_t *pErr;
            
    };
