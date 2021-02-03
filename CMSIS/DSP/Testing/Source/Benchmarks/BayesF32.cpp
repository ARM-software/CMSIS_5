#include "BayesF32.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void BayesF32::test_gaussian_naive_bayes_predict_f32()
    {
       int16_t p;

       p = arm_gaussian_naive_bayes_predict_f32(&bayes, 
                inp, 
                bufp);

    } 

  
    void BayesF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
       this->vecDim = *it++;
       this->classNb = *it++;

       switch(id)
       {
          case BayesF32::TEST_GAUSSIAN_NAIVE_BAYES_PREDICT_F32_1:
          {

            int nbp,nbi;

            input.reload(BayesF32::INPUTS2_F32_ID,mgr);
            params.reload(BayesF32::PARAMS2_F32_ID,mgr);
            dims.reload(BayesF32::DIMS2_S16_ID,mgr);

            int16_t *dimsp=dims.ptr();

            nbi = dimsp[2*this->nb];
            nbp = dimsp[2*this->nb + 1];

            const float32_t *paramsp = params.ptr() + nbp;

            this->theta=paramsp ;
            this->sigma=paramsp + (this->classNb * this->vecDim);
            this->classPrior=paramsp + 2*(this->classNb * this->vecDim);
            this->epsilon=paramsp[this->classNb + 2*(this->classNb * this->vecDim)];
            //printf("%f %f %f\n",this->theta[0],this->sigma[0],this->classPrior[0]);

            // Reference patterns are not loaded when we are in dump mode
            predicts.reload(BayesF32::PREDICTS2_S16_ID,mgr);

            outputProbas.create(this->classNb,BayesF32::OUT_PROBA_F32_ID,mgr);

            bayes.vectorDimension=this->vecDim;
            bayes.numberOfClasses=this->classNb;
            bayes.theta=this->theta;
            bayes.sigma=this->sigma;
            bayes.classPriors=this->classPrior;
            bayes.epsilon=this->epsilon; 

            this->inp = input.ptr() + nbi;

            this->bufp = outputProbas.ptr();

          }
          break;

       }
       


    }

    void BayesF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        this->nb++;
    }
