#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions_f16.h"

class UnaryTestsF16:public Client::Suite
    {
        public:
            UnaryTestsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryTestsF16_decl.h"
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::Pattern<float16_t> ref;
            Client::Pattern<int16_t> dims;
            Client::LocalPattern<float16_t> output;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<float16_t> a;
            Client::LocalPattern<float16_t> b;

            int nbr;
            int nbc;

            arm_matrix_instance_f16 in1;
            arm_matrix_instance_f16 in2;
            arm_matrix_instance_f16 out;
    };
