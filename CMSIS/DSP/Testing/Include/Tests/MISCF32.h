#include "Test.h"
#include "Pattern.h"
class MISCF32:public Client::Suite
    {
        public:
            MISCF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCF32_decl.h"
            
            Client::Pattern<float32_t> inputA;
            Client::Pattern<float32_t> inputB;

            Client::LocalPattern<float32_t> output;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;

            int nba,nbb;

           
    };
