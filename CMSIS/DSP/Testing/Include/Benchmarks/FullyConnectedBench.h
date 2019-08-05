#include "Test.h"
#include "Pattern.h"
class FullyConnectedBench:public Client::Suite
    {
        public:
            FullyConnectedBench(Testing::testID_t id);
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FullyConnectedBench_decl.h"
            
            int repeatNb;
            Client::Pattern<q7_t> input;
            Client::Pattern<q7_t> bias;
            Client::Pattern<q7_t> weight;
            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q15_t> temp;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q7_t> ref;
    };