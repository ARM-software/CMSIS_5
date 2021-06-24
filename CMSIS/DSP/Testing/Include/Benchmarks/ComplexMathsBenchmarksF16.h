#include "Test.h"
#include "Pattern.h"

#include "dsp/complex_math_functions_f16.h"

class ComplexMathsBenchmarksF16:public Client::Suite
    {
        public:
            ComplexMathsBenchmarksF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ComplexMathsBenchmarksF16_decl.h"
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            // REal input
            Client::Pattern<float16_t> input3;
            Client::LocalPattern<float16_t> output;

            int nb;

            const float16_t *inp1;
            const float16_t *inp2;
            const float16_t *inp3;
            float16_t *outp;
            
    };
