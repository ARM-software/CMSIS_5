#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class MFCCQ31:public Client::Suite
    {
        public:
            MFCCQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MFCCQ31_decl.h"
            
            Client::Pattern<q31_t> input1;
            Client::Pattern<q31_t> input2;
            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> tmp;
            Client::LocalPattern<q31_t> tmpin;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;

            arm_mfcc_instance_q31 mfcc;

            uint32_t fftLen;

    };
