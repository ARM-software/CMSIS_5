#include "Test.h"
#include "Pattern.h"
class MicroBenchmarksQ7:public Client::Suite
    {
        public:
            MicroBenchmarksQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MicroBenchmarksQ7_decl.h"

            Client::Pattern<q7_t> input1;
            Client::Pattern<q7_t> input2;
            Client::LocalPattern<q7_t> output;

            
            int nbSamples;

            q7_t *inp1;
            q7_t *inp2;
            q7_t *outp;
    };
