#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions_f16.h"

class FastMathF16:public Client::Suite
    {
        public:
            FastMathF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathF16_decl.h"
            Client::Pattern<float16_t> samples;

            Client::LocalPattern<float16_t> output;
            
            int nbSamples;

            float16_t *pSrc;
            float16_t *pDst;
            
            
    };
