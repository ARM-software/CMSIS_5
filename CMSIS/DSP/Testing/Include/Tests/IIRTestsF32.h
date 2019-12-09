#include "Test.h"
#include "Pattern.h"
class IIRTestsF32:public Client::Suite
    {
        public:
            IIRTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "IIRTestsF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> coefs;
            Client::LocalPattern<float32_t> state;
            Client::LocalPattern<float32_t> output;
            Client::RefPattern<float32_t> ref;
    };
