#include "Test.h"
#include "Pattern.h"
class ComplexMathsBenchmarksQ15:public Client::Suite
    {
        public:
            ComplexMathsBenchmarksQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexMathsBenchmarksQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            // REal input
            Client::Pattern<q15_t> input3;
            Client::LocalPattern<q15_t> output;

            int nb;

            const q15_t *inp1;
            const q15_t *inp2;
            const q15_t *inp3;
            q15_t *outp;
            
    };
