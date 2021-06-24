#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class BinaryTestsQ7:public Client::Suite
    {
        public:
            BinaryTestsQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryTestsQ7_decl.h"
            Client::Pattern<q7_t> input1;
            Client::Pattern<q7_t> input2;
            Client::Pattern<q7_t> ref;
            Client::Pattern<int16_t> dims;
            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q7_t> tmp;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<q7_t> a;
            Client::LocalPattern<q7_t> b;

            int nbr;
            int nbc;

            arm_matrix_instance_q7 in1;
            arm_matrix_instance_q7 in2;
            arm_matrix_instance_q7 out;
    };
