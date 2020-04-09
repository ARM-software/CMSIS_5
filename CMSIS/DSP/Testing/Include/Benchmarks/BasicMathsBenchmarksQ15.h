#include "Test.h"
#include "Pattern.h"
class BasicMathsBenchmarksQ15:public Client::Suite
    {
        public:
            BasicMathsBenchmarksQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicMathsBenchmarksQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            Client::LocalPattern<q15_t> output;

            int nb;

            q15_t *inp1;
            q15_t *inp2;
            q15_t *outp;
            
    };
