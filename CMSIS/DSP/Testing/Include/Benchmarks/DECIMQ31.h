#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class DECIMQ31:public Client::Suite
    {
        public:
            DECIMQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DECIMQ31_decl.h"
            Client::Pattern<q31_t> coefs;
            Client::Pattern<q31_t> samples;

            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> state;

            int nbTaps;
            int nbSamples;
            int decimationFactor;
            int interpolationFactor;

            arm_fir_decimate_instance_q31  instDecim;
            arm_fir_interpolate_instance_q31 instInterpol;
            
            const q31_t *pSrc;
            q31_t *pDst;
            
    };
