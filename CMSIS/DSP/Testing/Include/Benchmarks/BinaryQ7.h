#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class BinaryQ7:public Client::Suite
    {
        public:
            BinaryQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryQ7_decl.h"
            Client::Pattern<q7_t> input1;
            Client::Pattern<q7_t> input2;
            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q7_t> state;

            int nbr;
            int nbi;
            int nbc;

            arm_matrix_instance_q7 in1;
            arm_matrix_instance_q7 in2;
            arm_matrix_instance_q7 out;

            q7_t *pState;
            
    };
