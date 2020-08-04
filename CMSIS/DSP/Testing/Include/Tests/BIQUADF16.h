#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions_f16.h"

class BIQUADF16:public Client::Suite
    {
        public:
            BIQUADF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BIQUADF16_decl.h"
            
            Client::Pattern<float16_t> coefs;
            Client::Pattern<float16_t> inputs;
            Client::Pattern<int16_t> configs;
            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<float16_t> state;
            Client::LocalPattern<float16_t> debugstate;
            Client::LocalPattern<float16_t> vecCoefs;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;


            arm_biquad_casd_df1_inst_f16 Sdf1;
            arm_biquad_cascade_df2T_instance_f16 Sdf2T;
            arm_biquad_cascade_stereo_df2T_instance_f16 SStereodf2T;

    };
