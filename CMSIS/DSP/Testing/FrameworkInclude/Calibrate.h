#include "Test.h"
#include "Pattern.h"
class Calibrate:public Client::Suite
    {
        public:
            Calibrate(Testing::testID_t id);
            void empty();
            void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
    };
