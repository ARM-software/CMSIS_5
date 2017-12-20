#include "Driver_Flash.h"
  
// NAND driver instance
extern ARM_DRIVER_FLASH ARM_Driver_Flash_(0);
extern ARM_DRIVER_FLASH * flashDev = &(ARM_Driver_Flash_(0));
 
 
int main (void)
{
  /* Query drivers capabilities */
  const ARM_FLASH_CAPABILITIES capabilities = flashDev->GetCapabilities();
 
  /* Initialize NAND device */
  flashDev->Initialize (NULL);
  
  /* Power-on NAND device */
  flashDev->PowerControl (ARM_POWER_FULL);
  
  /* Read data taking data_width into account */
  uint8_t buf[256];
  flashDev->ReadData (0x1000U, buf, sizeof(buf)>>capabilities.data_width);
  
  /* Switch off gracefully */
  flashDev->PowerControl (ARM_POWER_OFF);
  flashDev->Uninitialize ();
}
