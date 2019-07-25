#include "Test.h"
#include "Pattern.h"
#include "SupportTestsF32.h"
class SupportTests : public Client::Group
{
   public:
     SupportTests(Testing::testID_t id):Client::Group(id)
 ,SupportTestsF32Var(1)

     { 
        this->addContainer(&SupportTestsF32Var);

     }
    private:
        SupportTestsF32 SupportTestsF32Var;
;
};
class DSPTests : public Client::Group
{
   public:
     DSPTests(Testing::testID_t id):Client::Group(id)
 ,SupportTestsVar(2)

     { 
        this->addContainer(NULL);this->addContainer(&SupportTestsVar);
this->addContainer(NULL);this->addContainer(NULL);this->addContainer(NULL);
     }
    private:
        SupportTests SupportTestsVar;
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
