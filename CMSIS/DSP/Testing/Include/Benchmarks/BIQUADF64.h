#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class BIQUADF64:public Client::Suite
    {
        public:
            BIQUADF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BIQUADF64_decl.h"
            Client::Pattern<float64_t> samples;
            Client::Pattern<float64_t> coefs;

            Client::LocalPattern<float64_t> output;
            Client::LocalPattern<float64_t> state;

            arm_biquad_cascade_df2T_instance_f64 instBiquadDf2T;

            int nbSamples;
            int numStages;   

            const float64_t *pSrc;
            float64_t *pDst;     
            
    };
