#include "Test.h"
#include "Pattern.h"
class NNSupport:public Client::Suite
    {
        public:
            NNSupport(Testing::testID_t id);
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "NNSupport_decl.h"
            
           

    };