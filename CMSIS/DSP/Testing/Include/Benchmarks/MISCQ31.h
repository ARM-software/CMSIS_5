#include "Test.h"
#include "Pattern.h"
class MISCQ31:public Client::Suite
    {
        public:
            MISCQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCQ31_decl.h"
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::LocalPattern<q31_t> output;

            int nba;
            int nbb;
            
            const q31_t *inp1;
            const q31_t *inp2;
            q31_t *outp;
    };
