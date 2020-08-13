#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions.h"

class StatsF32:public Client::Suite
    {
        public:
            StatsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsF32_decl.h"
            
            Client::Pattern<float32_t> inputA;
            Client::Pattern<float32_t> inputB;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<float32_t> tmp;

            float32_t *inap;
            float32_t *inbp;
            float32_t *outp;
            float32_t *tmpp;

            int nb;
           

    };
