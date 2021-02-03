#include "BayesF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define REL_ERROR ((float16_t)3e-3)

    void BayesF16::test_gaussian_naive_bayes_predict_f16()
    {
       const float16_t *inp = input.ptr();

       float16_t *bufp = outputProbas.ptr();
       int16_t *p = outputPredicts.ptr();

       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *p = arm_gaussian_naive_bayes_predict_f16(&bayes, 
                inp, 
                bufp);

          inp += this->vecDim;
          bufp += this->classNb;
          p++;
       }

        ASSERT_REL_ERROR(outputProbas,probas,REL_ERROR);
        ASSERT_EQ(outputPredicts,predicts);
    } 

  
    void BayesF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       (void)paramsArgs;

       switch(id)
       {
          case BayesF16::TEST_GAUSSIAN_NAIVE_BAYES_PREDICT_F16_1:
            

            input.reload(BayesF16::INPUTS1_F16_ID,mgr);
            params.reload(BayesF16::PARAMS1_F16_ID,mgr);
            dims.reload(BayesF16::DIMS1_S16_ID,mgr);

            const int16_t *dimsp=dims.ptr();
            const float16_t *paramsp = params.ptr();

            this->nbPatterns=dimsp[0];
            this->classNb=dimsp[1];
            this->vecDim=dimsp[2];

            this->theta=paramsp;
            this->sigma=paramsp + (this->classNb * this->vecDim);
            this->classPrior=paramsp + 2*(this->classNb * this->vecDim);
            this->epsilon=paramsp[this->classNb + 2*(this->classNb * this->vecDim)];
            //printf("%f %f %f\n",this->theta[0],this->sigma[0],this->classPrior[0]);

            // Reference patterns are not loaded when we are in dump mode
            probas.reload(BayesF16::PROBAS1_F16_ID,mgr);
            predicts.reload(BayesF16::PREDICTS1_S16_ID,mgr);

            outputProbas.create(this->nbPatterns*this->classNb,BayesF16::OUT_PROBA_F16_ID,mgr);
            outputPredicts.create(this->nbPatterns,BayesF16::OUT_PREDICT_S16_ID,mgr);

            bayes.vectorDimension=this->vecDim;
            bayes.numberOfClasses=this->classNb;
            bayes.theta=this->theta;
            bayes.sigma=this->sigma;
            bayes.classPriors=this->classPrior;
            bayes.epsilon=this->epsilon; 

          break;

       }
       


    }

    void BayesF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        outputProbas.dump(mgr);
        outputPredicts.dump(mgr);
    }
