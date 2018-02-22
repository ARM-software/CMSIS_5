#include "Driver_NAND.h"
  
// NAND driver instance
extern ARM_DRIVER_NAND Driver_NAND0;
extern ARM_DRIVER_NAND * nandDev = &Driver_NAND0;
 
static void SendAddress (ARM_DRIVER_NAND *drv, uint32_t dev_num, uint32_t addr, uint32_t cycles)
{
  while (cycles--) {
    drv->SendAddress (dev_num, (uint8_t)addr);
    addr >>= 8U;
  }
}
 
int main (void)
{
  /* Query drivers capabilities */
  const ARM_NAND_CAPABILITIES capabilities = nandDev->GetCapabilities();
 
  /* Initialize NAND device */
  nandDev->Initialize (NULL);
  
  /* Power-on NAND device */
  nandDev->PowerControl (ARM_POWER_FULL);
  
  /* Turn ON device power */
  uint32_t volt = 0U;
  if (capabilities.vcc)      { volt |= ARM_NAND_POWER_VCC_3V3;  }
  if (capabilities.vcc_1v8)  { volt |= ARM_NAND_POWER_VCC_1V8;  }
  if (capabilities.vccq)     { volt |= ARM_NAND_POWER_VCCQ_3V3; }
  if (capabilities.vccq_1v8) { volt |= ARM_NAND_POWER_VCCQ_1V8; }
  
  if (volt != 0U) {  
    nandDev->DevicePower (volt);
  }
  
  /* Setting bus mode */
  nandDev->Control (0U, ARM_NAND_BUS_MODE, ARM_NAND_BUS_SDR);
  
  /* Setting bus data width */
  nandDev->Control (0U, ARM_NAND_BUS_DATA_WIDTH, ARM_NAND_BUS_DATA_WIDTH_8);

  /* Enable chip manually if needed */
  if (capabilities.ce_manual) {
    nandDev->ChipEnable (0U, true);
  }
  
  /* Send ONFI Read command */
  nandDev->SendCommand (0U, 0x00U);
  
  /* Send address, LSB first */
  SendAddress (nandDev, 0U, 0x0100U, 2U);
  SendAddress (nandDev, 0U, 0x0047U, 2U);
  
  /* Read some data */
  uint8_t buf[256];
  nandDev->ReadData (0U, buf, sizeof(buf)/sizeof(buf[0]), 0U);
  
  /* Disable chip manually if needed */
  if (capabilities.ce_manual) {
    nandDev->ChipEnable (0U, false);
  }
  
  /* Switch off gracefully */
  volt = 0U;
  if (capabilities.vcc)  { volt |= ARM_NAND_POWER_VCC_OFF;  }
  if (capabilities.vccq) { volt |= ARM_NAND_POWER_VCCQ_OFF; }
  if (volt) {
    nandDev->DevicePower (volt);
  }
  nandDev->PowerControl (ARM_POWER_OFF);
  nandDev->Uninitialize ();
}
