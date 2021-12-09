#include "Test.h"
#include "Pattern.h"

#include "dsp/complex_math_functions.h"


class ComplexTestsF64:public Client::Suite
    {
        public:
            ComplexTestsF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexTestsF64_decl.h"
            
            Client::Pattern<float64_t> input1;
            Client::Pattern<float64_t> input2;
            Client::LocalPattern<float64_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float64_t> ref;
    };
