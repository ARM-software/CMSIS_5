#include "Test.h"
#include "Pattern.h"
class SupportTestsF32:public Client::Suite
    {
        public:
            SupportTestsF32(Testing::testID_t id);
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportTestsF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> coefs;

            Client::LocalPattern<float32_t> output;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;

            int nbSamples;

    };