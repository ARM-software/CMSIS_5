#include "Test.h"
#include "Pattern.h"

#include "dsp/fast_math_functions.h"

class FastMathF32:public Client::Suite
    {
        public:
            FastMathF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FastMathF32_decl.h"
            Client::Pattern<float32_t> samples;

            Client::LocalPattern<float32_t> output;
            
            int nbSamples;

            float32_t *pSrc;
            float32_t *pDst;
            
            
    };
