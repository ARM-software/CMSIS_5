#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions.h"

class FastMathQ63:public Client::Suite
    {
        public:
            FastMathQ63(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ63_decl.h"
            
            Client::Pattern<q63_t> input;
            Client::Pattern<uint64_t> inputU64;
            Client::Pattern<int64_t> inputS64;
            Client::Pattern<int32_t> inputS32;


            Client::LocalPattern<int32_t> outputVals;
            Client::LocalPattern<int16_t> outputNorms;


            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<int32_t> refVal;
            Client::RefPattern<int16_t> refNorm;


           
    };
