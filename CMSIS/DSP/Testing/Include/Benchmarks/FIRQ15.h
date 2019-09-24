#include "Test.h"
#include "Pattern.h"
class FIRQ15:public Client::Suite
    {
        public:
            FIRQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRQ15_decl.h"
            Client::Pattern<q15_t> coefs;
            Client::Pattern<q15_t> samples;
            Client::Pattern<q15_t> refs;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> error;
            Client::LocalPattern<q15_t> state;

            int nbTaps;
            int nbSamples;

            arm_fir_instance_q15  instFir;
            arm_lms_instance_q15  instLms;
            arm_lms_norm_instance_q15 instLmsNorm;

            const q15_t *pSrc;
            const q15_t *pCoefs;
            q15_t *pDst;
            const q15_t *pRef;
            q15_t *pErr;
            
    };
