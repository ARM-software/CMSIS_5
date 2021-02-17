#include "Test.h"
#include "Pattern.h"

#include "dsp/quaternion_math_functions.h"

class QuaternionMathsBenchmarksF32:public Client::Suite
    {
        public:
            QuaternionMathsBenchmarksF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "QuaternionMathsBenchmarksF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::LocalPattern<float32_t> output;

            Client::RefPattern<float32_t> ref;


            int nb;

            float32_t *inp1;
            float32_t *inp2;
            float32_t *outp;

            float32_t *refp;
            
    };
