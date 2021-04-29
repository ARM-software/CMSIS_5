#include "Test.h"
#include "Pattern.h"

#include "dsp/bayes_functions_f16.h"

class BayesF16:public Client::Suite
    {
        public:
            BayesF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BayesF16_decl.h"
            
            Client::Pattern<float16_t> input;
            Client::Pattern<float16_t> params;
            Client::Pattern<int16_t> dims;

            Client::LocalPattern<float16_t> outputProbas;
            Client::LocalPattern<float16_t> temp;
            Client::LocalPattern<int16_t> outputPredicts;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> probas;
            Client::RefPattern<int16_t> predicts;

            int nbPatterns,classNb,vecDim;
            const float16_t *theta;
            const float16_t *sigma;
            const float16_t *classPrior;
            float16_t epsilon;

            arm_gaussian_naive_bayes_instance_f16 bayes;

    };
