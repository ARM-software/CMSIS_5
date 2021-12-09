#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions.h"

class SupportBarF32:public Client::Suite
    {
        public:
            SupportBarF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportBarF32_decl.h"
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> coefs;

            Client::LocalPattern<float32_t> output;

            int vecDim;
            int nbVectors;

            const float32_t *inp;
            const float32_t *coefsp;

            float32_t *outp;

            
    };
