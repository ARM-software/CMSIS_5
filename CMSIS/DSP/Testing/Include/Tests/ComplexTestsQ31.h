#include "Test.h"
#include "Pattern.h"

#include "dsp/complex_math_functions.h"

class ComplexTestsQ31:public Client::Suite
    {
        public:
            ComplexTestsQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexTestsQ31_decl.h"
            
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::LocalPattern<q31_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;

            Client::LocalPattern<q63_t> dotOutput;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q63_t> dotRef;
    };
