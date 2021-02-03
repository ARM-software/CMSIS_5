#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class FIRQ7:public Client::Suite
    {
        public:
            FIRQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRQ7_decl.h"
            Client::Pattern<q7_t> coefs;
            Client::Pattern<q7_t> samples;
            Client::Pattern<q7_t> refs;

            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q7_t> error;
            Client::LocalPattern<q7_t> state;

            int nbTaps;
            int nbSamples;

            arm_fir_instance_q7  instFir;

            const q7_t *pSrc;
            const q7_t *pCoefs;
            q7_t *pDst;
            const q7_t *pRef;
            q7_t *pErr;
            
    };
