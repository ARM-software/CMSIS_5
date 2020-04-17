#ifndef _SVM_H_
#define _SVM_H_

#define nbSupportVectors 155
#define vectorDimensions 256

#define degree 3
#define coef0 1.000000
#define gamma 0.003906
#define intercept 0.849596

extern const float dualCoefs[nbSupportVectors];
extern const float supportVectors[nbSupportVectors*vectorDimensions];

#endif
