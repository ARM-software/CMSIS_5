#include "Test.h"
#include "Pattern.h"
class BIQUADQ31:public Client::Suite
    {
        public:
            BIQUADQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "BIQUADQ31_decl.h"
            
            Client::Pattern<q31_t> coefs;
            Client::Pattern<q31_t> inputs;
            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> state;
            Client::LocalPattern<q63_t> state64;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;


            arm_biquad_casd_df1_inst_q31 S;
            arm_biquad_cas_df1_32x64_ins_q31 S32x64;

    };
