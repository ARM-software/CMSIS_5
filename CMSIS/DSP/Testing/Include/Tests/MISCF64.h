#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class MISCF64:public Client::Suite
    {
        public:
            MISCF64(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCF64_decl.h"
            
            Client::Pattern<float64_t> inputA;
            Client::Pattern<float64_t> inputB;

            Client::LocalPattern<float64_t> output;
            Client::LocalPattern<float64_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float64_t> ref;

            int nba,nbb,errOffset,first;

           
    };
