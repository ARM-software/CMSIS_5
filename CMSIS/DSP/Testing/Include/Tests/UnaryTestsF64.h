#include "Test.h"
#include "Pattern.h"

#include "dsp/matrix_functions.h"

class UnaryTestsF64:public Client::Suite
    {
        public:
            UnaryTestsF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "UnaryTestsF64_decl.h"

            void compute_ldlt_error(const int n,const int16_t *outpp);
            
            Client::Pattern<float64_t> input1;
            Client::Pattern<float64_t> input2;
            Client::Pattern<float64_t> ref;

            Client::Pattern<float64_t> refll;
            Client::Pattern<float64_t> refd;
            Client::Pattern<int16_t> refp;

            Client::Pattern<int16_t> dims;
            Client::LocalPattern<float64_t> output;

            Client::LocalPattern<float64_t> outputll;
            Client::LocalPattern<float64_t> outputd;
            Client::LocalPattern<int16_t> outputp;

            /* Local copies of inputs since matrix instance in CMSIS-DSP are not using
               pointers to const.
            */
            Client::LocalPattern<float64_t> a;
            Client::LocalPattern<float64_t> b;
            Client::LocalPattern<float64_t> c;
            Client::LocalPattern<float64_t> d;

            Client::LocalPattern<float64_t> tmpapat;
            Client::LocalPattern<float64_t> tmpbpat;
            Client::LocalPattern<float64_t> tmpcpat;
            Client::LocalPattern<float64_t> outputa;
            Client::LocalPattern<float64_t> outputb;

            int nbr;
            int nbc;

            arm_matrix_instance_f64 in1;
            arm_matrix_instance_f64 in2;
            arm_matrix_instance_f64 out;

            arm_matrix_instance_f64 outll;
            arm_matrix_instance_f64 outd;

            float64_t *outa;   
            float64_t *outb;
    };
