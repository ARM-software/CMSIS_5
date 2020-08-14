#include "Test.h"
#include "Pattern.h"

#include "dsp/distance_functions_f16.h"

class DistanceF16:public Client::Suite
    {
        public:
            DistanceF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DistanceF16_decl.h"
            
            Client::Pattern<float16_t> inputA;
            Client::Pattern<float16_t> inputB;

            Client::LocalPattern<float16_t> tmpA;
            Client::LocalPattern<float16_t> tmpB;

            int vecDim;

            const float16_t *inpA;
            const float16_t *inpB;

            float16_t *tmpAp;
            float16_t *tmpBp;


    };
