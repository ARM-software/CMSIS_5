#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions.h"

class FastMathQ15:public Client::Suite
    {
        public:
            FastMathQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ15_decl.h"
            
            Client::Pattern<q15_t> input;

            Client::Pattern<q15_t> numerator;
            Client::Pattern<q15_t> denominator;


            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<int16_t> shift;


            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;
            Client::RefPattern<int16_t> refShift;

           
    };
