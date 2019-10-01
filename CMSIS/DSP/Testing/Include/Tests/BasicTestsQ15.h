#include "Test.h"
#include "Pattern.h"
class BasicTestsQ15:public Client::Suite
    {
        public:
            BasicTestsQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicTestsQ15_decl.h"
            
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q63_t> dotOutput;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;

            Client::RefPattern<q63_t> dotRef;

            /* Offset or scale value */
            q15_t scalar;
    };
