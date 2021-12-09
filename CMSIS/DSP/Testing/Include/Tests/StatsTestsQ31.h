#include "Test.h"
#include "Pattern.h"

#include "dsp/statistics_functions.h"

class StatsTestsQ31:public Client::Suite
    {
        public:
            StatsTestsQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsTestsQ31_decl.h"
            
            Client::Pattern<q31_t> inputA;
            Client::Pattern<q31_t> inputB;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q63_t> outputPower;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<q31_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;
            Client::RefPattern<q63_t> refPower;
            Client::Pattern<int16_t> maxIndexes;
            Client::Pattern<int16_t> minIndexes;

            int nbPatterns;
            int vecDim;

            int refOffset;

           

    };
