#include "Test.h"
#include "Pattern.h"

#include "dsp/distance_functions.h"

class DistanceU32:public Client::Suite
    {
        public:
            DistanceU32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DistanceU32_decl.h"
            
            Client::Pattern<uint32_t> inputA;
            Client::Pattern<uint32_t> inputB;

            Client::LocalPattern<uint32_t> tmpA;
            Client::LocalPattern<uint32_t> tmpB;

            int vecDim;

            const uint32_t *inpA;
            const uint32_t *inpB;

            uint32_t *tmpAp;
            uint32_t *tmpBp;


    };
