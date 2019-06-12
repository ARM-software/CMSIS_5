#include "Test.h"
#include "Pattern.h"
#include "FullyConnected.h"
class NNTests : public Client::Group
{
   public:
     NNTests(Testing::testID_t id):Client::Group(id)
 ,FullyConnectedVar(1)

     { 
        this->addContainer(&FullyConnectedVar);

     }
    private:
        FullyConnected FullyConnectedVar;
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
