#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class UnaryF32:public Client::Suite
    {
        public:
            UnaryF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;

            Client::Pattern<float32_t> vec;

            Client::LocalPattern<float32_t> a;
            Client::LocalPattern<float32_t> b;
            Client::LocalPattern<float32_t> output;

            Client::LocalPattern<float32_t> outputll;
            Client::LocalPattern<float32_t> outputd;
            Client::LocalPattern<int16_t> outputp;

            arm_matrix_instance_f32 in1;
            arm_matrix_instance_f32 in2;
            arm_matrix_instance_f32 out;

            arm_matrix_instance_f32 outll;
            arm_matrix_instance_f32 outd;
            
            int nbr;
            int nbc;

            float32_t *vecp;
            float32_t *outp;

            int16_t *outpp;

            
    };
