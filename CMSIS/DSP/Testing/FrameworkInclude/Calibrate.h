#include "Test.h"
#include "Pattern.h"
class Calibrate:public Client::Suite
    {
        public:
            Calibrate(Testing::testID_t id);
            void empty();
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
    };
