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
            Client::Pattern<float32_t> samples;
            Client::Pattern<float32_t> coefs;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> state;
            Client::LocalPattern<float32_t> neonCoefs;

            arm_biquad_casd_df1_inst_f32 instBiquadDf1;
            arm_biquad_cascade_df2T_instance_f32 instBiquadDf2T;
            arm_biquad_cascade_stereo_df2T_instance_f32 instStereo;

            int nbSamples;
            int numStages;    

            const float32_t *pSrc;
            float32_t *pDst;     
            
    };
