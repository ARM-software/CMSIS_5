#include "Test.h"
#include "Pattern.h"

#include "dsp/transform_functions.h"

class TransformQ15:public Client::Suite
    {
        public:
            TransformQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformQ15_decl.h"
            Client::Pattern<q15_t> samples;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> state;
            
            int nbSamples;
            int ifft;
            int bitRev;

            q15_t *pSrc;
            q15_t *pDst;
            q15_t *pState;

            arm_cfft_instance_q15 cfftInstance;

            arm_dct4_instance_q15 dct4Instance;
            arm_rfft_instance_q15 rfftInstance;
            arm_cfft_radix4_instance_q15 cfftRadix4Instance;
            arm_cfft_radix2_instance_q15 cfftRadix2Instance;
            
    };
