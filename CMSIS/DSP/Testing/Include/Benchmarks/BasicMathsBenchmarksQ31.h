#include "Test.h"
#include "Pattern.h"
class BasicMathsBenchmarksQ31:public Client::Suite
    {
        public:
            BasicMathsBenchmarksQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicMathsBenchmarksQ31_decl.h"
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::LocalPattern<q31_t> output;

            int nb;
            
            q31_t *inp1;
            q31_t *inp2;
            q31_t *outp;
    };
