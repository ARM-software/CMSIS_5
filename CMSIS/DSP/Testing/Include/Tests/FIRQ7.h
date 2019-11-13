#include "Test.h"
#include "Pattern.h"
class FIRQ7:public Client::Suite
    {
        public:
            FIRQ7(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRQ7_decl.h"
            
            Client::Pattern<q7_t> coefs;
            Client::Pattern<q7_t> inputs;
            Client::RefPattern<int16_t> configs;
            Client::LocalPattern<q7_t> output;
            Client::LocalPattern<q7_t> state;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q7_t> ref;


            arm_fir_instance_q7 S;
    };
