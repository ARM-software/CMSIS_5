#include "Test.h"
#include "Pattern.h"
class BIQUADF32:public Client::Suite
    {
        public:
            BIQUADF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BIQUADF32_decl.h"
            
            Client::Pattern<float32_t> coefs;
            Client::Pattern<float32_t> inputs;
            Client::Pattern<int16_t> configs;
            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> state;
            Client::LocalPattern<float32_t> debugstate;
            Client::LocalPattern<float32_t> vecCoefs;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;


            arm_biquad_casd_df1_inst_f32 Sdf1;
            arm_biquad_cascade_df2T_instance_f32 Sdf2T;
            arm_biquad_cascade_stereo_df2T_instance_f32 SStereodf2T;

    };
