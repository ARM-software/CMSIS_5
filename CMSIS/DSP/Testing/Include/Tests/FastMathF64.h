#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions.h"

class FastMathF64:public Client::Suite
    {
        public:
            FastMathF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathF64_decl.h"
            
            Client::Pattern<float64_t> input;

            Client::LocalPattern<float64_t> output;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float64_t> ref;

           
    };
