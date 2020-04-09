#include "Test.h"
#include "Pattern.h"
class UnaryF64:public Client::Suite
    {
        public:
            UnaryF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryF64_decl.h"
            Client::Pattern<float64_t> input1;
            Client::LocalPattern<float64_t> output;

            int nbr;
            int nbc;

            arm_matrix_instance_f64 in1;
            arm_matrix_instance_f64 out;
    };
