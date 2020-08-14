#include "Test.h"
#include "Pattern.h"
class MicroBenchmarksF16:public Client::Suite
    {
        public:
            MicroBenchmarksF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MicroBenchmarksF16_decl.h"

            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::LocalPattern<float16_t> output;

            
            int nbSamples;

            float16_t *inp1;
            float16_t *inp2;
            float16_t *outp;
    };
