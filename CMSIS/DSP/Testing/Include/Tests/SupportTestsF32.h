#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions.h"

class SupportTestsF32:public Client::Suite
    {
        public:
            SupportTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportTestsF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> coefs;
            Client::LocalPattern<float32_t> buffer;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<q15_t> outputQ15;
            Client::LocalPattern<q31_t> outputQ31;
            Client::LocalPattern<q7_t> outputQ7;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;
            Client::RefPattern<q15_t> refQ15;
            Client::RefPattern<q31_t> refQ31;
            Client::RefPattern<q7_t> refQ7;

            int nbSamples;
            int offset;

    };
