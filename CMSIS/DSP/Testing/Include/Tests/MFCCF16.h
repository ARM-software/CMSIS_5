#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions_f16.h"

class MFCCF16:public Client::Suite
    {
        public:
            MFCCF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MFCCF16_decl.h"
            
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<float16_t> tmp;
            Client::LocalPattern<float16_t> tmpin;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;

            arm_mfcc_instance_f16 mfcc;

            uint16_t fftLen;

    };
