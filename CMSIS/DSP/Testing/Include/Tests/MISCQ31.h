#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class MISCQ31:public Client::Suite
    {
        public:
            MISCQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCQ31_decl.h"
            
            Client::Pattern<q31_t> inputA;
            Client::Pattern<q31_t> inputB;

            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> tmp;

            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;

            int nba,nbb,errOffset,first;

           
    };
