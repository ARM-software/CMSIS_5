#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions.h"

class SupportTestsQ7:public Client::Suite
    {
        public:
            SupportTestsQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportTestsQ7_decl.h"
            
            Client::Pattern<q7_t> inputQ7;

            Client::LocalPattern<float32_t> outputF32;
            Client::LocalPattern<q15_t> outputQ15;
            Client::LocalPattern<q31_t> outputQ31;
            Client::LocalPattern<q7_t> outputQ7;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> refF32;
            Client::RefPattern<q15_t> refQ15;
            Client::RefPattern<q31_t> refQ31;
            Client::RefPattern<q7_t> refQ7;

            int nbSamples;

    };
