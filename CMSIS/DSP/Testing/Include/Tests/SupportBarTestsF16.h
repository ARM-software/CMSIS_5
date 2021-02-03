#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions_f16.h"

class SupportBarTestsF16:public Client::Suite
    {
        public:
            SupportBarTestsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportBarTestsF16_decl.h"
            Client::Pattern<float16_t> input;
            Client::Pattern<float16_t> coefs;
            Client::Pattern<float16_t> ref;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float16_t> output;

            int nbTests;


    };
