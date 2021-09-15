#include "Test.h"
#include "Pattern.h"

#include "dsp/basic_math_functions.h"

class BasicTestsF64:public Client::Suite
    {
        public:
            BasicTestsF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicTestsF64_decl.h"
            
            Client::Pattern<float64_t> input1;
            Client::Pattern<float64_t> input2;
            Client::LocalPattern<float64_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float64_t> ref;

            float64_t min,max;
    };
