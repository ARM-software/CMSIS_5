#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class MISCQ7:public Client::Suite
    {
        public:
            MISCQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCQ7_decl.h"
            
            Client::Pattern<q7_t> inputA;
            Client::Pattern<q7_t> inputB;

            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q7_t> tmp;

            Client::LocalPattern<q15_t> scratchA,scratchB;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q7_t> ref;

            int nba,nbb,first;

           
    };
