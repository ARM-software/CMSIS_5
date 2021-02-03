#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions_f16.h"

class StatsTestsF16:public Client::Suite
    {
        public:
            StatsTestsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsTestsF16_decl.h"
            
            Client::Pattern<float16_t> inputA;
            Client::Pattern<float16_t> inputB;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<float16_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;
            Client::Pattern<int16_t> maxIndexes;
            Client::Pattern<int16_t> minIndexes;

            int nbPatterns;
            int vecDim;

            int refOffset;

           

    };
