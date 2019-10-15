#include "Test.h"
#include "Pattern.h"
class SupportBarTestsF32:public Client::Suite
    {
        public:
            SupportBarTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportBarTestsF32_decl.h"
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> coefs;
            Client::Pattern<float32_t> ref;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float32_t> output;

            int nbTests;


    };
