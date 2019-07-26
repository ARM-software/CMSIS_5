#include "Test.h"

#include "Pooling.h"
    Pooling::Pooling(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(2,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(3,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(4,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(5,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(6,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(7,(Client::test)&Pooling::test_avgpool_s8);
this->addTest(8,(Client::test)&Pooling::test_avgpool_s8);

    }
