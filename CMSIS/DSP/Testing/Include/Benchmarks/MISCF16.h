#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions_f16.h"

class MISCF16:public Client::Suite
    {
        public:
            MISCF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCF16_decl.h"
            Client::Pattern<float16_t> input1;
            Client::Pattern<float16_t> input2;
            Client::LocalPattern<float16_t> output;

            int nba;
            int nbb;

            const float16_t *inp1;
            const float16_t *inp2;
            float16_t *outp;
            
    };
