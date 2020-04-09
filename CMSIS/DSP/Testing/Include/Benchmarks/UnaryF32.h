#include "Test.h"
#include "Pattern.h"
class UnaryF32:public Client::Suite
    {
        public:
            UnaryF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::LocalPattern<float32_t> output;

            int nbr;
            int nbc;

            arm_matrix_instance_f32 in1;
            arm_matrix_instance_f32 out;
    };
