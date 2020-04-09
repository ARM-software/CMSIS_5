#include "Test.h"
#include "Pattern.h"
class FullyConnected:public Client::Suite
    {
        public:
            FullyConnected(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FullyConnected_decl.h"

            int32_t output_mult = 1073741824;
            int16_t output_shift = -1;
            int32_t filter_offset = 1;
            int32_t input_offset = 1;
            int32_t output_offset = -1;
            int32_t act_min =-128;
            int32_t act_max= 127;
            int32_t nb_batches=1;
            int32_t rowDim;
            int32_t colDim;
            
            Client::Pattern<q7_t> input;
            Client::Pattern<q31_t> bias;
            Client::Pattern<q7_t> weight;
            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q15_t> temp;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q7_t> ref;

           




    };
