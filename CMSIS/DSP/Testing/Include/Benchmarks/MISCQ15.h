#include "Test.h"
#include "Pattern.h"
class MISCQ15:public Client::Suite
    {
        public:
            MISCQ15(Testing::testID_t id);
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            Client::LocalPattern<q15_t> output;

            int nba;
            int nbb;
            
    };