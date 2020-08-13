#include "SVMF32.h"
#include <stdio.h>
#include "Error.h"


void SVMF32::test_svm_linear_predict_f32()
{
      int32_t result;

      arm_svm_linear_predict_f32(&this->linear,inp,&result);
      

} 


void SVMF32::test_svm_polynomial_predict_f32()
{
      int32_t result;

      arm_svm_polynomial_predict_f32(&this->poly,inp,&result);
      

} 

void SVMF32::test_svm_rbf_predict_f32()
{
      int32_t result;

      arm_svm_rbf_predict_f32(&this->rbf,inp,&result);
     

} 

void SVMF32::test_svm_sigmoid_predict_f32()
{
      int32_t result;

      arm_svm_sigmoid_predict_f32(&this->sigmoid,inp,&result);
      

} 

void SVMF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& testparams,Client::PatternMgr *mgr)
{
      
      int kind;
      int nbp,nbi;
      const float32_t *paramsp;

      std::vector<Testing::param_t>::iterator it = testparams.begin();
      this->vecDim = *it++;
      this->nbSupportVectors = *it++;

      switch(id)
      {
          case SVMF32::TEST_SVM_LINEAR_PREDICT_F32_1:
          {

             samples.reload(SVMF32::INPUT_F32_ID,mgr,this->vecDim);
             params.reload(SVMF32::PARAMS_LINEAR_F32_ID,mgr);
             dims.reload(SVMF32::DIMS_LINEAR_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbLinear];
             nbp = dimsp[2*this->nbLinear + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF32::LINEAR;


          }
          break;

          case SVMF32::TEST_SVM_POLYNOMIAL_PREDICT_F32_2:
          {
             
             samples.reload(SVMF32::INPUT_F32_ID,mgr,this->vecDim);
             params.reload(SVMF32::PARAMS_POLY_F32_ID,mgr);
             dims.reload(SVMF32::DIMS_POLY_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbPoly];
             nbp = dimsp[2*this->nbPoly + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF32::POLY;
          }
          break;

          case SVMF32::TEST_SVM_RBF_PREDICT_F32_3:
          {
             
             samples.reload(SVMF32::INPUT_F32_ID,mgr,this->vecDim);
             params.reload(SVMF32::PARAMS_RBF_F32_ID,mgr);
             dims.reload(SVMF32::DIMS_RBF_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbRBF];
             nbp = dimsp[2*this->nbRBF + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF32::RBF;
          }
          break;

          case SVMF32::TEST_SVM_SIGMOID_PREDICT_F32_4:
          {
             samples.reload(SVMF32::INPUT_F32_ID,mgr,this->vecDim);
             params.reload(SVMF32::PARAMS_SIGMOID_F32_ID,mgr);
             dims.reload(SVMF32::DIMS_SIGMOID_S16_ID,mgr);

             int16_t *dimsp=dims.ptr();

             nbi = dimsp[2*this->nbSigmoid];
             nbp = dimsp[2*this->nbSigmoid + 1];

             paramsp = params.ptr() + nbp;

             inp=samples.ptr() + nbi;

             kind = SVMF32::SIGMOID;
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

        
         case SVMF32::POLY:
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
             this->degree=(int)paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 3];

         break;

         case SVMF32::RBF:
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1];
         break;

         case SVMF32::SIGMOID:
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
         break;
      }

       
       switch(id)
       {
          case SVMF32::TEST_SVM_LINEAR_PREDICT_F32_1:
          {
             
             arm_svm_linear_init_f32(&linear, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes);
          }
          break;

          case SVMF32::TEST_SVM_POLYNOMIAL_PREDICT_F32_2:
          {
             
             arm_svm_polynomial_init_f32(&poly, 
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

          case SVMF32::TEST_SVM_RBF_PREDICT_F32_3:
          {
             
             arm_svm_rbf_init_f32(&rbf, 
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

          case SVMF32::TEST_SVM_SIGMOID_PREDICT_F32_4:
          {
             
             arm_svm_sigmoid_init_f32(&sigmoid, 
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

void SVMF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
{
        switch(id)
        {
             case SVMF32::TEST_SVM_LINEAR_PREDICT_F32_1:
              nbLinear++;
             break;

             case SVMF32::TEST_SVM_POLYNOMIAL_PREDICT_F32_2:
              nbPoly++;
             break;

             case SVMF32::TEST_SVM_RBF_PREDICT_F32_3:
              nbRBF++;
             break;

             case SVMF32::TEST_SVM_SIGMOID_PREDICT_F32_4:
              nbSigmoid++;
             break;
        }
}



