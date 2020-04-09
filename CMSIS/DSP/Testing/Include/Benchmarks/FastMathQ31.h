#include "Test.h"
#include "Pattern.h"
class FastMathQ31:public Client::Suite
    {
        public:
            FastMathQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ31_decl.h"
            Client::Pattern<q31_t> samples;

            Client::LocalPattern<q31_t> output;
            
            int nbSamples;

            q31_t *pSrc;
            q31_t *pDst;
            
            
    };
