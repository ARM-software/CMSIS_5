#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions.h"

class SupportQ7:public Client::Suite
    {
        public:
            SupportQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportQ7_decl.h"
            Client::Pattern<q7_t> samples;


            Client::Pattern<q31_t> samplesQ31;
            Client::Pattern<q15_t> samplesQ15;

            Client::LocalPattern<q7_t> output;
            
            int nbSamples;

            q7_t *pSrc;

            q31_t *pSrcQ31;
            q15_t *pSrcQ15;

            q7_t *pDst;
            
    };
