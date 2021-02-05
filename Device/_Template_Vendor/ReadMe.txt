/*****************************************************************************
 * @file     ReadMe.txt
 * @brief    Explanation how to use the Device folder and template files 
 * @version  V3.0.4
 * @date     20. January 2021
 *****************************************************************************/

Following directory structure and template files are given:

  - <Vendor>
      |
      +-- <Device>
            |
            +-- Include
            |     +-- Template                            only Armv8-M/v8.1-M TrustZone
            |     |     +- partition_<Device>.h           Secure/Non-Secure configuration
            |     +- <Device>.h                           header file 
            |     +- system_<Device>.h                    system include file 
            +-- Source
                  |
                  +- startup_<Device>.c                   C startup file file 
                  +- system_<Device>.c                    system source file 
                  |
                  +-- ARM                                 Arm ARMCLang toolchain
                  |    +- startup_<Device>.s              ASM startup file for ARMCC    (deprecated)
                  |    +- startup_<Device>.S              ASM startup file for ARMCLang (deprecated)
                  |    +- <Device>.sct                    Scatter file
                  |
                  +-- GCC                                 Arm GNU toolchain
                  |    +- startup_<Device>.S              ASM startup file              (deprecated)
                  |    +- <Device>.ld                     Linker description file
                  |
                  +-- IAR                                 IAR toolchain
                       +- startup_<Device>.s              ASM startup file


Copy the complete folder including files and replace:
  - folder name 'Vendor' with the abbreviation for the device vendor  e.g.: NXP. 
  - folder name 'Device' with your specific device name e.g.: LPC17xx.
  - in the filenames 'Device' with your specific device name e.g.: LPC17xx. 


The template files contain comments starting with 'ToDo: '
There it is described what you need to do.


The template files contain following placeholder:

  <Device>
  <Device> should be replaced with your specific device name.
   e.g.: LPC17xx
  
  <DeviceInterrupt>
  <DeviceInterrupt> should be replaced with a specific device interrupt name.
  e.g.: TIM1 for Timer#1 interrupt.

  <DeviceAbbreviation>
  <DeviceAbbreviation> should be replaced with a dedicated device family
  abbreviation (e.g.: LPC for LPC17xx device family)

  Cortex-M#
  Cortex-M# can be replaced with the specific Cortex-M number
  e.g.: Cortex-M3



Note:
  Template files (i.e. startup_Device.s, system_Device.c) are application
  specific and therefore expected to be copied into the application project
  folder prior to use!
  