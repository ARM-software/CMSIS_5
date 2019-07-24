#include "Test.h"
#include "Pattern.h"
#include "BayesF32.h"
class BayesTests : public Client::Group
{
   public:
     BayesTests(Testing::testID_t id):Client::Group(id)
 ,BayesF32Var(1)

     { 
        this->addContainer(&BayesF32Var);

     }
    private:
        BayesF32 BayesF32Var;
;
};
class DSPTests : public Client::Group
{
   public:
     DSPTests(Testing::testID_t id):Client::Group(id)
 ,BayesTestsVar(3)

     { 
        this->addContainer(NULL);this->addContainer(NULL);this->addContainer(&BayesTestsVar);

     }
    private:
        BayesTests BayesTestsVar;
;
};
class Root : public Client::Group
{
   public:
     Root(Testing::testID_t id):Client::Group(id)
 ,DSPTestsVar(1)

     { 
        this->addContainer(&DSPTestsVar);
this->addContainer(NULL);this->addContainer(NULL);this->addContainer(NULL);
     }
    private:
        DSPTests DSPTestsVar;
;
};
