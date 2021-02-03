#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions_f16.h"

class StatsF16:public Client::Suite
    {
        public:
            StatsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsF16_decl.h"
            
            Client::Pattern<float16_t> inputA;
            Client::Pattern<float16_t> inputB;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<float16_t> tmp;

            float16_t *inap;
            float16_t *inbp;
            float16_t *outp;
            float16_t *tmpp;

            int nb;
           

    };
