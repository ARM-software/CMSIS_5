#include "Driver_I2C.h"
 
// I2C driver instance
extern ARM_DRIVER_I2C Driver_I2C0; 
static ARM_DRIVER_I2C *i2cDev = &Driver_I2C0;
 
static volatile uint32_t event = 0;
 
static void I2C_DrvEvent (uint32_t e) {
    event |= e;
}
 
int main (void) {
    uint8_t cnt = 0;
 
    /* Initialize I2C peripheral */
    i2cDev->Initialize(I2C_DrvEvent);
 
    /* Power-on SPI peripheral */
    i2cDev->PowerControl(ARM_POWER_FULL);
 
    /* Configure USART bus*/
    i2cDev->Control(ARM_I2C_OWN_ADDRESS, 0x78);
 
    while (1) {
		/* Receive chuck */
        i2cDev->SlaveReceive(&cnt, 1);
        while ((event & ARM_event_TRANSFER_DONE) == 0);
        event &= ~ARM_event_TRANSFER_DONE;
 
		/* Transmit chunk back */
        i2cDev->SlaveTransmit(&cnt, 1);
        while ((event & ARM_event_TRANSFER_DONE) == 0);
        event &= ~ARM_event_TRANSFER_DONE;
    }
}
