#include "Test.h"
#include "Pattern.h"
class BasicTestsF16:public Client::Suite
    {
        public:
            BasicTestsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicTestsF16_decl.h"
            
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::LocalPattern<float16_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;
    };
