#include "dsp/filtering_functions_f16.h"

#include "Test.h"
#include "Pattern.h"

class MISCF16:public Client::Suite
    {
        public:
            MISCF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCF16_decl.h"
            
            Client::Pattern<float16_t> inputA;
            Client::Pattern<float16_t> inputB;

            Client::LocalPattern<float16_t> output;
            Client::LocalPattern<float16_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;

            int nba,nbb,errOffset,first;

           
    };
