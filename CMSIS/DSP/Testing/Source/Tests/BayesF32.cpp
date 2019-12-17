#include "BayesF32.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"



    void BayesF32::test_gaussian_naive_bayes_predict_f32()
    {
       const float32_t *inp = input.ptr();

       float32_t *bufp = outputProbas.ptr();
       int16_t *p = outputPredicts.ptr();

       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *p = arm_gaussian_naive_bayes_predict_f32(&bayes, 
                inp, 
                bufp);

          inp += this->vecDim;
          bufp += this->classNb;
          p++;
       }

        ASSERT_REL_ERROR(outputProbas,probas,(float32_t)5e-6);
        ASSERT_EQ(outputPredicts,predicts);
    } 

  
    void BayesF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       

       switch(id)
       {
          case BayesF32::TEST_GAUSSIAN_NAIVE_BAYES_PREDICT_F32_1:
            

            input.reload(BayesF32::INPUTS1_F32_ID,mgr);
            params.reload(BayesF32::PARAMS1_F32_ID,mgr);
            dims.reload(BayesF32::DIMS1_S16_ID,mgr);

            const int16_t *dimsp=dims.ptr();
            const float32_t *paramsp = params.ptr();

            this->nbPatterns=dimsp[0];
            this->classNb=dimsp[1];
            this->vecDim=dimsp[2];

            this->theta=paramsp;
            this->sigma=paramsp + (this->classNb * this->vecDim);
            this->classPrior=paramsp + 2*(this->classNb * this->vecDim);
            this->epsilon=paramsp[this->classNb + 2*(this->classNb * this->vecDim)];
            //printf("%f %f %f\n",this->theta[0],this->sigma[0],this->classPrior[0]);

            // Reference patterns are not loaded when we are in dump mode
            probas.reload(BayesF32::PROBAS1_F32_ID,mgr);
            predicts.reload(BayesF32::PREDICTS1_S16_ID,mgr);

            outputProbas.create(this->nbPatterns*this->classNb,BayesF32::OUT_PROBA_F32_ID,mgr);
            outputPredicts.create(this->nbPatterns,BayesF32::OUT_PREDICT_S16_ID,mgr);

            bayes.vectorDimension=this->vecDim;
            bayes.numberOfClasses=this->classNb;
            bayes.theta=this->theta;
            bayes.sigma=this->sigma;
            bayes.classPriors=this->classPrior;
            bayes.epsilon=this->epsilon; 

          break;

       }
       


    }

    void BayesF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        outputProbas.dump(mgr);
        outputPredicts.dump(mgr);
    }
