#include "Test.h"
#include "Pattern.h"
class MicroBenchmarksQ31:public Client::Suite
    {
        public:
            MicroBenchmarksQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MicroBenchmarksQ31_decl.h"

            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::LocalPattern<q31_t> output;

            
            int nbSamples;

            q31_t *inp1;
            q31_t *inp2;
            q31_t *outp;
    };
