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
            Client::Pattern<float16_t> samples;
            Client::Pattern<float16_t> coefs;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<float16_t> state;
            Client::LocalPattern<float16_t> neonCoefs;

            arm_biquad_casd_df1_inst_f16 instBiquadDf1;
            arm_biquad_cascade_df2T_instance_f16 instBiquadDf2T;
            arm_biquad_cascade_stereo_df2T_instance_f16 instStereo;

            int nbSamples;
            int numStages;    

            const float16_t *pSrc;
            float16_t *pDst;     
            
    };
