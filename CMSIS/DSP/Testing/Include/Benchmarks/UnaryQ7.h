#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class UnaryQ7:public Client::Suite
    {
        public:
            UnaryQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryQ7_decl.h"
            Client::Pattern<q7_t> input1;
            Client::Pattern<q7_t> vec;
            Client::LocalPattern<q7_t> output;

            int nbr;
            int nbc;

            q7_t *vecp;
            q7_t *outp;
            arm_matrix_instance_q7 in1;
            arm_matrix_instance_q7 out;
    };
