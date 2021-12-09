#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class MFCCF32:public Client::Suite
    {
        public:
            MFCCF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MFCCF32_decl.h"
            
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> tmp;
            Client::LocalPattern<float32_t> tmpin;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;

            arm_mfcc_instance_f32 mfcc;

            uint32_t fftLen;

    };
