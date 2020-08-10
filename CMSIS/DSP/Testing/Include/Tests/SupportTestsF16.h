#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions_f16.h"

class SupportTestsF16:public Client::Suite
    {
        public:
            SupportTestsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportTestsF16_decl.h"
            
            Client::Pattern<float16_t> input;
            Client::Pattern<q15_t> inputQ15;
            Client::Pattern<float32_t> inputF32;
            Client::Pattern<float16_t> coefs;
            Client::LocalPattern<float16_t> buffer;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<q15_t> outputQ15;
            Client::LocalPattern<float32_t> outputF32;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;
            Client::RefPattern<q15_t> refQ15;
            Client::RefPattern<float32_t> refF32;

            int nbSamples;
            int offset;

    };
