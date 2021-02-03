#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class BinaryF32:public Client::Suite
    {
        public:
            BinaryF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::LocalPattern<float32_t> output;

            int nbr;
            int nbi;
            int nbc;

            arm_matrix_instance_f32 in1;
            arm_matrix_instance_f32 in2;
            arm_matrix_instance_f32 out;
            
    };
