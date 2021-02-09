#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class UnaryTestsF32:public Client::Suite
    {
        public:
            UnaryTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryTestsF32_decl.h"

            void compute_ldlt_error(const int n,const int16_t *outpp);

            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::Pattern<float32_t> ref;

            Client::Pattern<float32_t> refll;
            Client::Pattern<float32_t> refd;
            Client::Pattern<int16_t> refp;

            Client::Pattern<int16_t> dims;
            Client::LocalPattern<float32_t> output;

            Client::LocalPattern<float32_t> outputll;
            Client::LocalPattern<float32_t> outputd;
            Client::LocalPattern<int16_t> outputp;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<float32_t> a;
            Client::LocalPattern<float32_t> b;
            Client::LocalPattern<float32_t> c;
            Client::LocalPattern<float32_t> d;

            Client::LocalPattern<float64_t> tmpapat;
            Client::LocalPattern<float64_t> tmpbpat;
            Client::LocalPattern<float64_t> tmpcpat;
            Client::LocalPattern<float64_t> outputa;
            Client::LocalPattern<float64_t> outputb;

            int nbr;
            int nbc;

            arm_matrix_instance_f32 in1;
            arm_matrix_instance_f32 in2;
            
            arm_matrix_instance_f32 out;

            arm_matrix_instance_f32 outll;
            arm_matrix_instance_f32 outd;

            float64_t *outa;   
            float64_t *outb;

            float32_t snrRel,snrAbs;

    };
