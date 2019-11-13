#include "Test.h"
#include "Pattern.h"
class ExampleCategoryQ7:public Client::Suite
    {
        public:
            ExampleCategoryQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ExampleCategoryQ7_decl.h"
            
            Client::Pattern<q7_t> input1;
            Client::Pattern<q7_t> input2;
            Client::LocalPattern<q7_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q7_t> ref;
    };
