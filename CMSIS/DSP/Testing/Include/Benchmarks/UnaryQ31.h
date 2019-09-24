#include "Test.h"
#include "Pattern.h"
class UnaryQ31:public Client::Suite
    {
        public:
            UnaryQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryQ31_decl.h"
            Client::Pattern<q31_t> input1;
            Client::LocalPattern<q31_t> output;

            int nbr;
            int nbc;

            arm_matrix_instance_q31 in1;
            arm_matrix_instance_q31 out;
    };
