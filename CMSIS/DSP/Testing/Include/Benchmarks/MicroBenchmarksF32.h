#include "Test.h"
#include "Pattern.h"
class MicroBenchmarksF32:public Client::Suite
    {
        public:
            MicroBenchmarksF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MicroBenchmarksF32_decl.h"

            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::LocalPattern<float32_t> output;

            
            int nbSamples;

            float32_t *inp1;
            float32_t *inp2;
            float32_t *outp;
    };
