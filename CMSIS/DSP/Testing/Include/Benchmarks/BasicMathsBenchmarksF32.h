#include "Test.h"
#include "Pattern.h"
class BasicMathsBenchmarksF32:public Client::Suite
    {
        public:
            BasicMathsBenchmarksF32(Testing::testID_t id);
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BasicMathsBenchmarksF32_decl.h"
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;
            Client::LocalPattern<float32_t> output;

            int nb;

            float32_t *inp1;
            float32_t *inp2;
            float32_t *outp;
            
    };