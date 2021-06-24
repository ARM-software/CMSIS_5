#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class UnaryQ15:public Client::Suite
    {
        public:
            UnaryQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> vec;
            Client::LocalPattern<q15_t> output;

            int nbr;
            int nbc;

            q15_t *vecp;
            q15_t *outp;
            arm_matrix_instance_q15 in1;
            arm_matrix_instance_q15 out;
    };
