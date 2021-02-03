#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions_f16.h"

class UnaryF16:public Client::Suite
    {
        public:
            UnaryF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryF16_decl.h"
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::Pattern<float16_t> vec;

            Client::LocalPattern<float16_t> a;
            Client::LocalPattern<float16_t> b;
            Client::LocalPattern<float16_t> output;

            int nbr;
            int nbc;

            float16_t *vecp;
            float16_t *outp;
            arm_matrix_instance_f16 in1;
            arm_matrix_instance_f16 in2;
            arm_matrix_instance_f16 out;
    };
