#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class BinaryQ31:public Client::Suite
    {
        public:
            BinaryQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryQ31_decl.h"
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> tmp;

            int nbr;
            int nbi;
            int nbc;

            arm_matrix_instance_q31 in1;
            arm_matrix_instance_q31 in2;
            arm_matrix_instance_q31 out;
            q31_t *tmpPtr;
            
    };
