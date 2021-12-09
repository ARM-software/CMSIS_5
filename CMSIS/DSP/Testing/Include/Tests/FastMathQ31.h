#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions.h"

class FastMathQ31:public Client::Suite
    {
        public:
            FastMathQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ31_decl.h"
            
            Client::Pattern<q31_t> input;


            Client::Pattern<q31_t> numerator;
            Client::Pattern<q31_t> denominator;


            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<int16_t> shift;


            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;
            Client::RefPattern<int16_t> refShift;


           
    };
