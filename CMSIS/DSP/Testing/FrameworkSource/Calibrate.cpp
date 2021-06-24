#include "Calibrate.h"

Calibrate::Calibrate(Testing::testID_t id):Client::Suite(id)
{
}
    
void Calibrate::empty()
{
}
       
void Calibrate::setUp(Testing::testID_t,std::vector<Testing::param_t>& ,Client::PatternMgr *)
{
}
       
void Calibrate::tearDown(Testing::testID_t,Client::PatternMgr *)
{

}
