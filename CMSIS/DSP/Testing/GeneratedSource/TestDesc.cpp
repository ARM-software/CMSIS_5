#include "Test.h"

#include "BayesF32.h"
    BayesF32::BayesF32(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&BayesF32::test_gaussian_naive_bayes_predict_f32);

    }
