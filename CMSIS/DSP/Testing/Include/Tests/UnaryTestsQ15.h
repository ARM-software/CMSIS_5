#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class UnaryTestsQ15:public Client::Suite
    {
        public:
            UnaryTestsQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryTestsQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            Client::Pattern<q15_t> ref;
            Client::Pattern<int16_t> dims;
            Client::LocalPattern<q15_t> output;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<q15_t> a;
            Client::LocalPattern<q15_t> b;

            int nbr;
            int nbc;

            arm_matrix_instance_q15 in1;
            arm_matrix_instance_q15 in2;
            arm_matrix_instance_q15 out;
    };
