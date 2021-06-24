#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions_f16.h"

class BinaryF16:public Client::Suite
    {
        public:
            BinaryF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryF16_decl.h"
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::LocalPattern<float16_t> output;

            int nbr;
            int nbi;
            int nbc;

            arm_matrix_instance_f16 in1;
            arm_matrix_instance_f16 in2;
            arm_matrix_instance_f16 out;
            
    };
