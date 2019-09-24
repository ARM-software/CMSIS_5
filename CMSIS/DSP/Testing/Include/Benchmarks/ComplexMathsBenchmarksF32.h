#include "Test.h"
#include "Pattern.h"
class ComplexMathsBenchmarksF32:public Client::Suite
    {
        public:
            ComplexMathsBenchmarksF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexMathsBenchmarksF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            // REal input
            Client::Pattern<float32_t> input3;
            Client::LocalPattern<float32_t> output;

            int nb;

            const float32_t *inp1;
            const float32_t *inp2;
            const float32_t *inp3;
            float32_t *outp;
            
    };
