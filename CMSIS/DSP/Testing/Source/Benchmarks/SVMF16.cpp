#include "SVMF16.h"
#include <stdio.h>
#include "Error.h"


void SVMF16::test_svm_linear_predict_f16()
{
      int32_t result;

      arm_svm_linear_predict_f16(&this->linear,inp,&result);
      

} 


void SVMF16::test_svm_polynomial_predict_f16()
{
      int32_t result;

      arm_svm_polynomial_predict_f16(&this->poly,inp,&result);
      

} 

void SVMF16::test_svm_rbf_predict_f16()
{
      int32_t result;

      arm_svm_rbf_predict_f16(&this->rbf,inp,&result);
     

} 

void SVMF16::test_svm_sigmoid_predict_f16()
{
      int32_t result;

      arm_svm_sigmoid_predict_f16(&this->sigmoid,inp,&result);
      

} 

void SVMF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& testparams,Client::PatternMgr *mgr)
{
      
      int kind;
      int nbp,nbi;
      const float16_t *paramsp;

      std::vector<Testing::param_t>::iterator it = testparams.begin();
      this->vecDim = *it++;
      this->nbSupportVectors = *it++;

      switch(id)
      {
          case SVMF16::TEST_SVM_LINEAR_PREDICT_F16_1:
          {

             samples.reload(SVMF16::INPUT_F16_ID,mgr,this->vecDim);
             params.reload(SVMF16::PARAMS_LINEAR_F16_ID,mgr);
             dims.reload(SVMF16::DIMS_LINEAR_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbLinear];
             nbp = dimsp[2*this->nbLinear + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF16::LINEAR;


          }
          break;

          case SVMF16::TEST_SVM_POLYNOMIAL_PREDICT_F16_2:
          {
             
             samples.reload(SVMF16::INPUT_F16_ID,mgr,this->vecDim);
             params.reload(SVMF16::PARAMS_POLY_F16_ID,mgr);
             dims.reload(SVMF16::DIMS_POLY_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbPoly];
             nbp = dimsp[2*this->nbPoly + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF16::POLY;
          }
          break;

          case SVMF16::TEST_SVM_RBF_PREDICT_F16_3:
          {
             
             samples.reload(SVMF16::INPUT_F16_ID,mgr,this->vecDim);
             params.reload(SVMF16::PARAMS_RBF_F16_ID,mgr);
             dims.reload(SVMF16::DIMS_RBF_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbRBF];
             nbp = dimsp[2*this->nbRBF + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF16::RBF;
          }
          break;

          case SVMF16::TEST_SVM_SIGMOID_PREDICT_F16_4:
          {
             samples.reload(SVMF16::INPUT_F16_ID,mgr,this->vecDim);
             params.reload(SVMF16::PARAMS_SIGMOID_F16_ID,mgr);
             dims.reload(SVMF16::DIMS_SIGMOID_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbSigmoid];
             nbp = dimsp[2*this->nbSigmoid + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF16::SIGMOID;
          }
          break;


      }


      
      
      this->classes[0] = 0;
      this->classes[1] = 1;
      this->intercept=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors];
      this->supportVectors=paramsp;
      this->dualCoefs=paramsp + (this->vecDim*this->nbSupportVectors);

      switch(kind)
      {

        
         case SVMF16::POLY:
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
             this->degree=(int)paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 3];

         break;

         case SVMF16::RBF:
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1];
         break;

         case SVMF16::SIGMOID:
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
         break;
      }

       
       switch(id)
       {
          case SVMF16::TEST_SVM_LINEAR_PREDICT_F16_1:
          {
             
             arm_svm_linear_init_f16(&linear, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes);
          }
          break;

          case SVMF16::TEST_SVM_POLYNOMIAL_PREDICT_F16_2:
          {
             
             arm_svm_polynomial_init_f16(&poly, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes,
                 this->degree,
                 this->coef0,
                 this->gamma
                 );
          }
          break;

          case SVMF16::TEST_SVM_RBF_PREDICT_F16_3:
          {
             
             arm_svm_rbf_init_f16(&rbf, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes,
                 this->gamma
                 );
          }
          break;

          case SVMF16::TEST_SVM_SIGMOID_PREDICT_F16_4:
          {
             
             arm_svm_sigmoid_init_f16(&sigmoid, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes,
                 this->coef0,
                 this->gamma
                 );
          }
          break;
       }


    
}

void SVMF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
{
        (void)mgr;
        switch(id)
        {
             case SVMF16::TEST_SVM_LINEAR_PREDICT_F16_1:
              nbLinear++;
             break;

             case SVMF16::TEST_SVM_POLYNOMIAL_PREDICT_F16_2:
              nbPoly++;
             break;

             case SVMF16::TEST_SVM_RBF_PREDICT_F16_3:
              nbRBF++;
             break;

             case SVMF16::TEST_SVM_SIGMOID_PREDICT_F16_4:
              nbSigmoid++;
             break;
        }
}



