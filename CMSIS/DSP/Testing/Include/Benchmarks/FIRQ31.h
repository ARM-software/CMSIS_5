#include "Test.h"
#include "Pattern.h"
class FIRQ31:public Client::Suite
    {
        public:
            FIRQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRQ31_decl.h"
            Client::Pattern<q31_t> coefs;
            Client::Pattern<q31_t> samples;
            Client::Pattern<q31_t> refs;

            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> error;
            Client::LocalPattern<q31_t> state;

            int nbTaps;
            int nbSamples;

            arm_fir_instance_q31  instFir;
            arm_lms_instance_q31  instLms;
            arm_lms_norm_instance_q31 instLmsNorm;
            
            const q31_t *pSrc;
            const q31_t *pCoefs;
            q31_t *pDst;
            const q31_t *pRef;
            q31_t *pErr;
    };
