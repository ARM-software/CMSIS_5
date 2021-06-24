#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions.h"

class StatsQ31:public Client::Suite
    {
        public:
            StatsQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsQ31_decl.h"
            
            Client::Pattern<q31_t> inputA;
            Client::Pattern<q31_t> inputB;

            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<q31_t> tmp;

            q31_t *inap;
            q31_t *inbp;
            q31_t *outp;
            q31_t *tmpp;

            int nb;
           

    };
