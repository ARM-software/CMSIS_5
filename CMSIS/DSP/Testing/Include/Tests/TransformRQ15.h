#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class TransformRQ15:public Client::Suite
    {
        public:
            TransformRQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformRQ15_decl.h"
            
            Client::Pattern<q15_t> input;
            Client::LocalPattern<q15_t> outputfft;
            Client::LocalPattern<q15_t> overheadoutputfft;
            Client::LocalPattern<q15_t> inputchanged;

            Client::RefPattern<q15_t> ref;

            arm_rfft_instance_q15 instRfftQ15;

            int ifft;
            int scaling;
            
    };
