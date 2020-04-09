#include "Test.h"
#include "Pattern.h"
class BIQUADQ15:public Client::Suite
    {
        public:
            BIQUADQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BIQUADQ15_decl.h"
            
            Client::Pattern<q15_t> coefs;
            Client::Pattern<q15_t> inputs;
            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> state;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;


            arm_biquad_casd_df1_inst_q15 S;

    };
