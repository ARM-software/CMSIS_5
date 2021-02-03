#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class FIRF32:public Client::Suite
    {
        public:
            FIRF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "FIRF32_decl.h"
            
            Client::Pattern<float32_t> coefs;
            Client::Pattern<float32_t> inputs;
            Client::RefPattern<int16_t> configs;

            Client::LocalPattern<float32_t> output;
            Client::LocalPattern<float32_t> state;
            Client::LocalPattern<float32_t> tmp;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;


            arm_fir_instance_f32 S;

    };
