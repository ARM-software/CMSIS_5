#include "Test.h"
#include "Pattern.h"

#include "dsp/support_functions_f16.h"

class SupportBarF16:public Client::Suite
    {
        public:
            SupportBarF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "SupportBarF16_decl.h"
            Client::Pattern<float16_t> input;
            Client::Pattern<float16_t> coefs;

            Client::LocalPattern<float16_t> output;

            int vecDim;
            int nbVectors;

            const float16_t *inp;
            const float16_t *coefsp;

            float16_t *outp;

            
    };
