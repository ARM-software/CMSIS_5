#include "Test.h"
#include "Pattern.h"
class UnaryTestsQ31:public Client::Suite
    {
        public:
            UnaryTestsQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryTestsQ31_decl.h"
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::Pattern<q31_t> ref;
            Client::Pattern<int16_t> dims;
            Client::LocalPattern<q31_t> output;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<q31_t> a;
            Client::LocalPattern<q31_t> b;

            int nbr;
            int nbc;

            arm_matrix_instance_q31 in1;
            arm_matrix_instance_q31 in2;
            arm_matrix_instance_q31 out;
    };
