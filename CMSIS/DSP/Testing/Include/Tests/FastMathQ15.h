#include "Test.h"
#include "Pattern.h"
class FastMathQ15:public Client::Suite
    {
        public:
            FastMathQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathQ15_decl.h"
            
            Client::Pattern<q15_t> input;

            Client::LocalPattern<q15_t> output;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;

           
    };
