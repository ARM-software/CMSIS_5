#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class BinaryTestsF32:public Client::Suite
    {
        public:
            BinaryTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BinaryTestsF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::Pattern<float32_t> ref;
            Client::Pattern<int16_t> dims;
            Client::LocalPattern<float32_t> output;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<float32_t> a;
            Client::LocalPattern<float32_t> b;

            int nbr;
            int nbc;

            arm_matrix_instance_f32 in1;
            arm_matrix_instance_f32 in2;
            arm_matrix_instance_f32 out;
    };
