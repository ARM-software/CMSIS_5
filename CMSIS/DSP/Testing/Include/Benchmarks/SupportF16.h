#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions_f16.h"

class SupportF16:public Client::Suite
    {
        public:
            SupportF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportF16_decl.h"
            Client::Pattern<float16_t> samples;
            Client::Pattern<float16_t> weights;
            Client::Pattern<q15_t> samplesQ15;
            Client::Pattern<float32_t> samplesF32;

            Client::LocalPattern<float16_t> output;
            
            int nbSamples;

            float16_t *pSrc;
            float16_t *pWeights;

            float32_t *pSrcF32;
            q15_t *pSrcQ15;

            float16_t *pDst;
            
    };
