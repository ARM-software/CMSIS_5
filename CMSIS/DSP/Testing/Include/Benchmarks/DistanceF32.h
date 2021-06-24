#include "Test.h"
#include "Pattern.h"

#include "dsp/distance_functions.h"

class DistanceF32:public Client::Suite
    {
        public:
            DistanceF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DistanceF32_decl.h"
            
            Client::Pattern<float32_t> inputA;
            Client::Pattern<float32_t> inputB;

            Client::LocalPattern<float32_t> tmpA;
            Client::LocalPattern<float32_t> tmpB;

            int vecDim;

            const float32_t *inpA;
            const float32_t *inpB;

            float32_t *tmpAp;
            float32_t *tmpBp;


    };
