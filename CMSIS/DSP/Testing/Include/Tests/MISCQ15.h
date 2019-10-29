#include "Test.h"
#include "Pattern.h"
class MISCQ15:public Client::Suite
    {
        public:
            MISCQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCQ15_decl.h"
            
            Client::Pattern<q15_t> inputA;
            Client::Pattern<q15_t> inputB;

            Client::LocalPattern<q15_t> output;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;

            int nba,nbb;

           
    };
