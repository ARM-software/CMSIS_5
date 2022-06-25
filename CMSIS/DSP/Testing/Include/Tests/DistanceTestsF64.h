#include "Test.h"
#include "Pattern.h"

#include "dsp/distance_functions.h"

class DistanceTestsF64:public Client::Suite
    {
        public:
            DistanceTestsF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DistanceTestsF64_decl.h"
            
            Client::Pattern<float64_t> inputA;
            Client::Pattern<float64_t> inputB;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float64_t> output;
            Client::LocalPattern<float64_t> tmpA;
            Client::LocalPattern<float64_t> tmpB;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float64_t> ref;

            int vecDim;
            int nbPatterns;


    };
