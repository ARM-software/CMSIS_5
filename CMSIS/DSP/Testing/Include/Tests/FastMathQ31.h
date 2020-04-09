#include "Test.h"
#include "Pattern.h"
class FastMathQ31:public Client::Suite
    {
        public:
            FastMathQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ31_decl.h"
            
            Client::Pattern<q31_t> input;

            Client::LocalPattern<q31_t> output;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;

           
    };
