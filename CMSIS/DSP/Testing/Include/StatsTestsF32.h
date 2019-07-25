#include "Test.h"
#include "Pattern.h"
class StatsTestsF32:public Client::Suite
    {
        public:
            StatsTestsF32(Testing::testID_t id);
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "StatsTestsF32_decl.h"
            
            Client::Pattern<float32_t> inputA;
            Client::Pattern<float32_t> inputB;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;

            int nbPatterns;
            int vecDim;

           

    };