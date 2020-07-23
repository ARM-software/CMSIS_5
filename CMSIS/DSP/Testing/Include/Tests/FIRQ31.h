#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class FIRQ31:public Client::Suite
    {
        public:
            FIRQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRQ31_decl.h"
            
            Client::Pattern<q31_t> coefs;
            Client::Pattern<q31_t> inputs;
            Client::RefPattern<int16_t> configs;
            Client::LocalPattern<q31_t> output;
            Client::LocalPattern<q31_t> state;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;


            arm_fir_instance_q31 S;
    };
