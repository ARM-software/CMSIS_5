#include "Test.h"
#include "Pattern.h"

#include "dsp/basic_math_functions_f16.h"

class BasicMathsBenchmarksF16:public Client::Suite
    {
        public:
            BasicMathsBenchmarksF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicMathsBenchmarksF16_decl.h"
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::LocalPattern<float16_t> output;

            Client::RefPattern<float16_t> ref;


            int nb;

            float16_t *inp1;
            float16_t *inp2;
            float16_t *outp;

            float16_t *refp;
            
    };
