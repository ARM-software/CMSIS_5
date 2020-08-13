#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions.h"

class StatsQ7:public Client::Suite
    {
        public:
            StatsQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsQ7_decl.h"
            
            Client::Pattern<q7_t> inputA;
            Client::Pattern<q7_t> inputB;

            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<q7_t> tmp;

            q7_t *inap;
            q7_t *inbp;
            q7_t *outp;
            q7_t *tmpp;

            int nb;
           

    };
