#include "Test.h"
#include "Pattern.h"
#include "Pooling.h"
class NNTests : public Client::Group
{
   public:
     NNTests(Testing::testID_t id):Client::Group(id)
 ,PoolingVar(2)

     { 
        this->addContainer(NULL);this->addContainer(&PoolingVar);

     }
    private:
        Pooling PoolingVar;
;
};
class Root : public Client::Group
{
   public:
     Root(Testing::testID_t id):Client::Group(id)
 ,NNTestsVar(3)

     { 
        this->addContainer(NULL);this->addContainer(NULL);this->addContainer(&NNTestsVar);
this->addContainer(NULL);
     }
    private:
        NNTests NNTestsVar;
;
};
