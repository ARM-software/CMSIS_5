#include "Test.h"
#include "Pattern.h"
class ComplexMathsBenchmarksQ31:public Client::Suite
    {
        public:
            ComplexMathsBenchmarksQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexMathsBenchmarksQ31_decl.h"
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            // REal input
            Client::Pattern<q31_t> input3;
            Client::LocalPattern<q31_t> output;

            int nb;

            const q31_t *inp1;
            const q31_t *inp2;
            const q31_t *inp3;
            q31_t *outp;
            
    };
