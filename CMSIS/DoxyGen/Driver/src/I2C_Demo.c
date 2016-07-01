#include "Driver_I2C.h"
#include "cmsis_os.h"                   // ARM::CMSIS:RTOS:Keil RTX
#include <string.h>
 
/* I2C Driver */
extern ARM_DRIVER_I2C Driver_I2C0;
static ARM_DRIVER_I2C * I2Cdrv = &Driver_I2C0;
 
 
#ifndef EEPROM_I2C_PORT
#define EEPROM_I2C_PORT       0         /* I2C Port number                    */
#endif
 
#define EEPROM_I2C_ADDR       0x51      /* 24LC128 EEPROM I2C address         */
 
#define EEPROM_MAX_ADDR       16384     /* Max memory locations available     */
#define EEPROM_MAX_WRITE      16        /* Max bytes to write in one step     */
 
#define A_WR                  0         /* Master will write to the I2C       */
#define A_RD                  1         /* Master will read from the I2C      */
 
static uint8_t DeviceAddr;
static uint8_t wr_buf[EEPROM_MAX_WRITE + 2];
 
int32_t EEPROM_WriteBuf (uint16_t addr, const uint8_t *buf, uint32_t len) {
 
  wr_buf[0] = (uint8_t)(addr >> 8);
  wr_buf[1] = (uint8_t)(addr & 0xFF);
 
  memcpy (&wr_buf[2], &buf[0], len);
 
  I2Cdrv->MasterTransmit (DeviceAddr, wr_buf, len + 2, false);
  while (I2Cdrv->GetStatus().busy);
  if (I2Cdrv->GetDataCount () != (len + 2)) return -1;
  /* Acknowledge polling */
 
  do {
    I2Cdrv->MasterReceive (DeviceAddr, &wr_buf[0], 1, false);
    while (I2Cdrv->GetStatus().busy);
  } while (I2Cdrv->GetDataCount () < 0);
 
  return 0;
}
 
int32_t EEPROM_ReadBuf (uint16_t addr, uint8_t *buf, uint32_t len) {
  uint8_t a[2];
 
  a[0] = (uint8_t)(addr >> 8);
  a[1] = (uint8_t)(addr & 0xFF);
 
  I2Cdrv->MasterTransmit (DeviceAddr, a, 2, true);
  while (I2Cdrv->GetStatus().busy);
  I2Cdrv->MasterReceive (DeviceAddr, buf, len, false);
  while (I2Cdrv->GetStatus().busy);
  if (I2Cdrv->GetDataCount () != len) return -1;
 
  return 0;
}
 
int32_t EEPROM_Initialize (void) {
  uint8_t val;
 
  I2Cdrv->Initialize   (NULL);
  I2Cdrv->PowerControl (ARM_POWER_FULL);
  I2Cdrv->Control      (ARM_I2C_BUS_SPEED, ARM_I2C_BUS_SPEED_FAST);
  I2Cdrv->Control      (ARM_I2C_BUS_CLEAR, 0);
 
  /* Init 24LC128 EEPROM device */
  DeviceAddr = EEPROM_I2C_ADDR;
 
  /* Read min and max address */
  if (EEPROM_ReadBuf (0x00, &val, 1) == 0) {
    return (EEPROM_ReadBuf (EEPROM_MAX_ADDR-1, &val, 1));
  }
  return -1;
}
  
uint32_t EEPROM_GetSize (void) {
  return EEPROM_MAX_ADDR;
}
