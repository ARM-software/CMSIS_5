#include "Test.h"
#include "Pattern.h"
class BasicTestsQ31:public Client::Suite
    {
        public:
            BasicTestsQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicTestsQ31_decl.h"
            
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::Pattern<uint32_t> inputLogical1;
            Client::Pattern<uint32_t> inputLogical2;

            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q63_t> dotOutput;
            Client::LocalPattern<uint32_t> outputLogical;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;
            Client::RefPattern<q63_t> dotRef;
            Client::RefPattern<uint32_t> refLogical;

            /* Offset or scale value */
            q31_t scalar;
    };
