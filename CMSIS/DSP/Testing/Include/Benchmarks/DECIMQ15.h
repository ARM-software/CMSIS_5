#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class DECIMQ15:public Client::Suite
    {
        public:
            DECIMQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DECIMQ15_decl.h"
            Client::Pattern<q15_t> coefs;
            Client::Pattern<q15_t> samples;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> state;

            int nbTaps;
            int nbSamples;
            int decimationFactor;
            int interpolationFactor;

            arm_fir_decimate_instance_q15  instDecim;
            arm_fir_interpolate_instance_q15 instInterpol;
            
            const q15_t *pSrc;
            q15_t *pDst;
            
    };
