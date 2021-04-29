#include "Test.h"
#include "Pattern.h"

#include "dsp/bayes_functions.h"

class BayesF32:public Client::Suite
    {
        public:
            BayesF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BayesF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> params;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float32_t> outputProbas;
            Client::LocalPattern<float32_t> temp;
            Client::LocalPattern<int16_t> outputPredicts;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<int16_t> predicts;

            int classNb,vecDim;
            int nb=0;
            const float32_t *theta;
            const float32_t *sigma;
            const float32_t *classPrior;
            float32_t epsilon;

            arm_gaussian_naive_bayes_instance_f32 bayes;

            const float32_t *inp;
            float32_t *bufp,*tempp;

    };
