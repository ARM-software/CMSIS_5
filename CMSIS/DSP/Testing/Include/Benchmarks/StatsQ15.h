#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions.h"

class StatsQ15:public Client::Suite
    {
        public:
            StatsQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsQ15_decl.h"
            
            Client::Pattern<q15_t> inputA;
            Client::Pattern<q15_t> inputB;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<q15_t> tmp;

            q15_t *inap;
            q15_t *inbp;
            q15_t *outp;
            q15_t *tmpp;

            int nb;
           

    };
