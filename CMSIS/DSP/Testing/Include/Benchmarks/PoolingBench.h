#include "Test.h"
#include "Pattern.h"
class PoolingBench:public Client::Suite
    {
        public:
            PoolingBench(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "PoolingBench_decl.h"
            
            Client::Pattern<q7_t> input;

            Client::LocalPattern<q7_t> tmpInput;
            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q15_t> temp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q7_t> ref;

            int DIM_IN_X;
            int DIM_IN_Y;
            int DIM_OUT_X;
            int DIM_OUT_Y;
            int IN_CHANNEL;
            int DIM_FILTER_X;
            int DIM_FILTER_Y;
            int PAD_WIDTH;
            int PAD_HEIGHT;
            int STRIDE_X;
            int STRIDE_Y;
            int ACT_MIN;
            int ACT_MAX;

            int repeatNb;

            q7_t *tmpin;
            q7_t *outp;
            q15_t *tempp;

    };
