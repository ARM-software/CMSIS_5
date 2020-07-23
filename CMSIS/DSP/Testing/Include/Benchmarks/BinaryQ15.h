#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class BinaryQ15:public Client::Suite
    {
        public:
            BinaryQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> state;

            int nbr;
            int nbi;
            int nbc;

            arm_matrix_instance_q15 in1;
            arm_matrix_instance_q15 in2;
            arm_matrix_instance_q15 out;

            q15_t *pState;
            
    };
