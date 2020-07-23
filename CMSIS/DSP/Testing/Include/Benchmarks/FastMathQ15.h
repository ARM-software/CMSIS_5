#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions.h"

class FastMathQ15:public Client::Suite
    {
        public:
            FastMathQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ15_decl.h"
            Client::Pattern<q15_t> samples;

            Client::LocalPattern<q15_t> output;
            
            int nbSamples;

            q15_t *pSrc;
            q15_t *pDst;
            
            
    };
