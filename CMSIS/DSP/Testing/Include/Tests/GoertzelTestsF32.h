#include "Test.h"
#include "Pattern.h"
class GoertzelTestsF32:public Client::Suite
    {
        public:
            GoertzelTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "GoertzelTestsF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::LocalPattern<float32_t> output;
            Client::RefPattern<float32_t> ref;
            Client::Pattern<float32_t> buffer;
    };
