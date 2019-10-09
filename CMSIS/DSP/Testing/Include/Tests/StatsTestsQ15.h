#include "Test.h"
#include "Pattern.h"
class StatsTestsQ15:public Client::Suite
    {
        public:
            StatsTestsQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsTestsQ15_decl.h"
            
            Client::Pattern<q15_t> inputA;
            Client::Pattern<q15_t> inputB;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q63_t> outputPower;
            Client::LocalPattern<int16_t> index;
            Client::LocalPattern<q15_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;
            Client::RefPattern<q63_t> refPower;
            Client::Pattern<int16_t> maxIndexes;
            Client::Pattern<int16_t> minIndexes;

            int nbPatterns;
            int vecDim;

            int refOffset;

           

    };
