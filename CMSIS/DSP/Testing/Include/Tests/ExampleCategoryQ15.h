#include "Test.h"
#include "Pattern.h"

#include "dsp/basic_math_functions.h"

class ExampleCategoryQ15:public Client::Suite
    {
        public:
            ExampleCategoryQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ExampleCategoryQ15_decl.h"
            
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            Client::LocalPattern<q15_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;
    };
