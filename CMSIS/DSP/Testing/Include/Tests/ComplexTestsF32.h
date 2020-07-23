#include "Test.h"
#include "Pattern.h"

#include "dsp/complex_math_functions.h"


class ComplexTestsF32:public Client::Suite
    {
        public:
            ComplexTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexTestsF32_decl.h"
            
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::LocalPattern<float32_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;
    };
