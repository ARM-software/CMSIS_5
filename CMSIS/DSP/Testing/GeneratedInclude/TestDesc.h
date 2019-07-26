#include "Test.h"
#include "Pattern.h"
#include "StatsTestsF32.h"
#include "SupportTestsF32.h"
#include "BasicTestsF32.h"
#include "SVMF32.h"
#include "BayesF32.h"
#include "DistanceTestsF32.h"
#include "DistanceTestsU32.h"
#include "BasicMathsBenchmarksF32.h"
#include "BasicMathsBenchmarksQ31.h"
#include "BasicMathsBenchmarksQ15.h"
#include "BasicMathsBenchmarksQ7.h"
#include "FullyConnected.h"
#include "FullyConnectedBench.h"
class StatsTests : public Client::Group
{
   public:
     StatsTests(Testing::testID_t id):Client::Group(id)
 ,StatsTestsF32Var(1)

     { 
        this->addContainer(&StatsTestsF32Var);

     }
    private:
        StatsTestsF32 StatsTestsF32Var;
;
};
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
class DistanceTests : public Client::Group
{
   public:
     DistanceTests(Testing::testID_t id):Client::Group(id)
 ,DistanceTestsF32Var(1)
,DistanceTestsU32Var(2)

     { 
        this->addContainer(&DistanceTestsF32Var);
this->addContainer(&DistanceTestsU32Var);

     }
    private:
        DistanceTestsF32 DistanceTestsF32Var;
DistanceTestsU32 DistanceTestsU32Var;
;
};
class DSPTests : public Client::Group
{
   public:
     DSPTests(Testing::testID_t id):Client::Group(id)
 ,StatsTestsVar(1)
,SupportTestsVar(2)
,BasicTestsVar(3)
,SVMTestsVar(4)
,BayesTestsVar(5)
,DistanceTestsVar(6)

     { 
        this->addContainer(&StatsTestsVar);
this->addContainer(&SupportTestsVar);
this->addContainer(&BasicTestsVar);
this->addContainer(&SVMTestsVar);
this->addContainer(&BayesTestsVar);
this->addContainer(&DistanceTestsVar);

     }
    private:
        StatsTests StatsTestsVar;
SupportTests SupportTestsVar;
BasicTests BasicTestsVar;
SVMTests SVMTestsVar;
BayesTests BayesTestsVar;
DistanceTests DistanceTestsVar;
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
