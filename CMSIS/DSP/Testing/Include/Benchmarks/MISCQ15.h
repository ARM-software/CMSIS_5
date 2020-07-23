#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class MISCQ15:public Client::Suite
    {
        public:
            MISCQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "MISCQ15_decl.h"
            Client::Pattern<q15_t> input1;
            Client::Pattern<q15_t> input2;
            Client::LocalPattern<q15_t> output;

            int nba;
            int nbb;
            
            const q15_t *inp1;
            const q15_t *inp2;
            q15_t *outp;
    };
