#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class TransformRQ31:public Client::Suite
    {
        public:
            TransformRQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformRQ31_decl.h"
            
            Client::Pattern<q31_t> input;
            Client::LocalPattern<q31_t> outputfft;
            Client::LocalPattern<q31_t> overheadoutputfft;
            Client::LocalPattern<q31_t> inputchanged;

            Client::RefPattern<q31_t> ref;

            arm_rfft_instance_q31 instRfftQ31;

            int ifft;
            int scaling;
            
    };
