#include "Test.h"
#include "Pattern.h"
#include "BasicTestsF32.h"
#include "SVMF32.h"
#include "BasicMathsBenchmarksF32.h"
#include "BasicMathsBenchmarksQ31.h"
#include "BasicMathsBenchmarksQ15.h"
#include "BasicMathsBenchmarksQ7.h"
#include "FullyConnected.h"
#include "FullyConnectedBench.h"
class BasicTests : public Client::Group
{
   public:
     BasicTests(Testing::testID_t id):Client::Group(id)
 ,BasicTestsF32Var(1)

     { 
        this->addContainer(&BasicTestsF32Var);

     }
    private:
        BasicTestsF32 BasicTestsF32Var;
;
};
class SVMTests : public Client::Group
{
   public:
     SVMTests(Testing::testID_t id):Client::Group(id)
 ,SVMF32Var(1)

     { 
        this->addContainer(&SVMF32Var);

     }
    private:
        SVMF32 SVMF32Var;
;
};
class DSPTests : public Client::Group
{
   public:
     DSPTests(Testing::testID_t id):Client::Group(id)
 ,BasicTestsVar(1)
,SVMTestsVar(2)

     { 
        this->addContainer(&BasicTestsVar);
this->addContainer(&SVMTestsVar);

     }
    private:
        BasicTests BasicTestsVar;
SVMTests SVMTestsVar;
;
};
class BasicBenchmarks : public Client::Group
{
   public:
     BasicBenchmarks(Testing::testID_t id):Client::Group(id)
 ,BasicMathsBenchmarksF32Var(1)
,BasicMathsBenchmarksQ31Var(2)
,BasicMathsBenchmarksQ15Var(3)
,BasicMathsBenchmarksQ7Var(4)

     { 
        this->addContainer(&BasicMathsBenchmarksF32Var);
this->addContainer(&BasicMathsBenchmarksQ31Var);
this->addContainer(&BasicMathsBenchmarksQ15Var);
this->addContainer(&BasicMathsBenchmarksQ7Var);

     }
    private:
        BasicMathsBenchmarksF32 BasicMathsBenchmarksF32Var;
BasicMathsBenchmarksQ31 BasicMathsBenchmarksQ31Var;
BasicMathsBenchmarksQ15 BasicMathsBenchmarksQ15Var;
BasicMathsBenchmarksQ7 BasicMathsBenchmarksQ7Var;
;
};
class DSPBenchmarks : public Client::Group
{
   public:
     DSPBenchmarks(Testing::testID_t id):Client::Group(id)
 ,BasicBenchmarksVar(1)

     { 
        this->addContainer(&BasicBenchmarksVar);

     }
    private:
        BasicBenchmarks BasicBenchmarksVar;
;
};
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
class NNBenchmarks : public Client::Group
{
   public:
     NNBenchmarks(Testing::testID_t id):Client::Group(id)
 ,FullyConnectedBenchVar(1)

     { 
        this->addContainer(&FullyConnectedBenchVar);

     }
    private:
        FullyConnectedBench FullyConnectedBenchVar;
;
};
class Root : public Client::Group
{
   public:
     Root(Testing::testID_t id):Client::Group(id)
 ,DSPTestsVar(1)
,DSPBenchmarksVar(2)
,NNTestsVar(3)
,NNBenchmarksVar(4)

     { 
        this->addContainer(&DSPTestsVar);
this->addContainer(&DSPBenchmarksVar);
this->addContainer(&NNTestsVar);
this->addContainer(&NNBenchmarksVar);

     }
    private:
        DSPTests DSPTestsVar;
DSPBenchmarks DSPBenchmarksVar;
NNTests NNTestsVar;
NNBenchmarks NNBenchmarksVar;
;
};
