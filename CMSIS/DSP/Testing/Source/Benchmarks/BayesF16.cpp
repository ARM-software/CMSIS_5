#include "BayesF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void BayesF16::test_gaussian_naive_bayes_predict_f16()
    {

       (void)arm_gaussian_naive_bayes_predict_f16(&bayes, 
                inp, 
                bufp,tempp);

    } 

  
    void BayesF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
       this->vecDim = *it++;
       this->classNb = *it++;

       switch(id)
       {
          case BayesF16::TEST_GAUSSIAN_NAIVE_BAYES_PREDICT_F16_1:
          {

            int nbp,nbi;

            input.reload(BayesF16::INPUTS2_F16_ID,mgr);
            params.reload(BayesF16::PARAMS2_F16_ID,mgr);
            dims.reload(BayesF16::DIMS2_S16_ID,mgr);

            int16_t *dimsp=dims.ptr();

            nbi = dimsp[2*this->nb];
            nbp = dimsp[2*this->nb + 1];

            const float16_t *paramsp = params.ptr() + nbp;

            this->theta=paramsp ;
            this->sigma=paramsp + (this->classNb * this->vecDim);
            this->classPrior=paramsp + 2*(this->classNb * this->vecDim);
            this->epsilon=paramsp[this->classNb + 2*(this->classNb * this->vecDim)];
            //printf("%f %f %f\n",this->theta[0],this->sigma[0],this->classPrior[0]);

            // Reference patterns are not loaded when we are in dump mode
            predicts.reload(BayesF16::PREDICTS2_S16_ID,mgr);

            outputProbas.create(this->classNb,BayesF16::OUT_PROBA_F16_ID,mgr);
            temp.create(this->classNb,BayesF16::OUT_PROBA_F16_ID,mgr);

            bayes.vectorDimension=this->vecDim;
            bayes.numberOfClasses=this->classNb;
            bayes.theta=this->theta;
            bayes.sigma=this->sigma;
            bayes.classPriors=this->classPrior;
            bayes.epsilon=this->epsilon; 

            this->inp = input.ptr() + nbi;

            this->bufp = outputProbas.ptr();
            this->tempp = temp.ptr();

          }
          break;

       }
       


    }

    void BayesF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        (void)mgr;
        this->nb++;
    }
