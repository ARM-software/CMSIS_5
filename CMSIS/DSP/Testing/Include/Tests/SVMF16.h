#include "Test.h"
#include "Pattern.h"

#include "dsp/svm_functions_f16.h"


class SVMF16:public Client::Suite
    {
        public:
            SVMF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SVMF16_decl.h"
            Client::Pattern<float16_t> samples;
            Client::Pattern<int16_t> dims;
            Client::Pattern<float16_t> params;

            Client::RefPattern<int32_t> ref;
            Client::LocalPattern<int32_t> output;

            arm_svm_linear_instance_f16 linear;
            arm_svm_polynomial_instance_f16 poly;
            arm_svm_rbf_instance_f16 rbf;
            arm_svm_sigmoid_instance_f16 sigmoid;

            int vecDim,nbSupportVectors,nbTestSamples,degree;
            int32_t classes[2]={0,0};
            float16_t intercept;
            const float16_t *supportVectors;
            const float16_t *dualCoefs;
            float16_t coef0, gamma;

            enum {
                LINEAR=1,
                POLY=2,
                RBF=3,
                SIGMOID=4
            } kind;

            
    };
