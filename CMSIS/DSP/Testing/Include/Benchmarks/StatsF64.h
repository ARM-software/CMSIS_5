#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions.h"

class StatsF64:public Client::Suite
    {
        public:
            StatsF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsF64_decl.h"
            
            Client::Pattern<float64_t> inputA;
            Client::Pattern<float64_t> inputB;

            Client::LocalPattern<float64_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<float64_t> tmp;

            float64_t *inap;
            float64_t *inbp;
            float64_t *outp;
            float64_t *tmpp;

            int nb;
           

    };
