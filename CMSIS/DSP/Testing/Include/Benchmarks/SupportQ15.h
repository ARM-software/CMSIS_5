#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions.h"

class SupportQ15:public Client::Suite
    {
        public:
            SupportQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportQ15_decl.h"
            Client::Pattern<q15_t> samples;

            Client::Pattern<q31_t> samplesQ31;
            Client::Pattern<q7_t> samplesQ7;

            Client::LocalPattern<q15_t> output;
            
            int nbSamples;

            q15_t *pSrc;

            q31_t *pSrcQ31;
            q7_t *pSrcQ7;

            q15_t *pDst;
            
    };
