#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions.h"

class SupportF32:public Client::Suite
    {
        public:
            SupportF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportF32_decl.h"
            Client::Pattern<float32_t> samples;
            Client::Pattern<float32_t> weights;
            Client::Pattern<q31_t> samplesQ31;
            Client::Pattern<q15_t> samplesQ15;
            Client::Pattern<q7_t> samplesQ7;

            Client::LocalPattern<float32_t> output;
            
            int nbSamples;

            float32_t *pSrc;
            float32_t *pWeights;

            q31_t *pSrcQ31;
            q15_t *pSrcQ15;
            q7_t *pSrcQ7;

            float32_t *pDst;
            
    };
