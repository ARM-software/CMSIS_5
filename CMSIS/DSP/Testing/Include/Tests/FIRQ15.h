#include "Test.h"
#include "Pattern.h"
class FIRQ15:public Client::Suite
    {
        public:
            FIRQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRQ15_decl.h"
            
            Client::Pattern<q15_t> coefs;
            Client::Pattern<q15_t> inputs;
            Client::RefPattern<int16_t> configs;
            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> state;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;


            arm_fir_instance_q15 S;
    };
