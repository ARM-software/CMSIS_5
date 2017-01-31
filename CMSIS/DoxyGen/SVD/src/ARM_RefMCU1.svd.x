<?xml version="1.0" encoding="utf-8"?>
<!--  File naming: <vendor>_<part/series name>_svd.xml  -->
<!-- 
  Copyright (C) 2016 ARM Limited. All rights reserved.

  Purpose: System Viewer SVD Description for ARM Reference
           This is a reference example.
  -->
<device schemaVersion="1.3" xmlns:xs="http://www.w3.org/2001/XMLSchema-instance" xs:noNamespaceSchemaLocation="CMSIS-SVD.xsd">
  <vendor>ARM Ltd.</vendor>                                       <!--  Name for DoxyGroup  -->
  <vendorID>ARM</vendorID>                                        <!--  Vendor ID  -->
  <name>ARM_RefMCU</name>                                         <!--  name of part or part series  -->
  <series>ARM_Ref</series>                                        <!--  series   -->
  <version>1.0</version>                                          <!--  version of this description  -->

  <description>SVD Reference Description MCU V1.0, \n 
  with ARM 32-bit Cortex-M3 Microcontroller, CPU clock up to 80MHz</description>

  <licenseText>THIS SOFTWARE IS PROVIDED "AS IS". \n 
  NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, \n
  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. \n
  ARM SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, \n
  FOR ANY REASON WHATSOEVER. </licenseText>

  <cpu>
    <name>SC300</name>
    <revision>r2p0</revision>                                    <!--  CPU Revision r2p0 = 0x2000  -->
    <endian>little</endian>                                      <!--  little, big, configurable (headerfile: little, big, configurable (#ifdef compilerflag)    -->
    <mpuPresent>true</mpuPresent>                                <!--  Does the CPU has a MPU? {0|1}  -->
    <fpuPresent>true</fpuPresent>                                <!--  Does the CPU has a FPU? {0|1}  -->
    <nvicPrioBits>4</nvicPrioBits>                               <!--  Number of NVIC Priority Bits {8..2}  -->
    <vendorSystickConfig>0</vendorSystickConfig>                 <!--  Does the Vendor has his own Systick Configuration Function? See CMSIS: core_cm3.h  -->
	<fpuDP>0</fpuDP>
	<icachePresent>false</icachePresent>
	<dcachePresent>false</dcachePresent>
	<itcmPresent>false</itcmPresent>
	<dtcmPresent>0</dtcmPresent>
  </cpu>
  
  <headerSystemFilename>system_ARM</headerSystemFilename>        <!--  System Header File overwrite  -->
  <headerDefinitionsPrefix>ARM_</headerDefinitionsPrefix>        <!--  Prefix for all structs and #defines  -->
  
  <addressUnitBits>8</addressUnitBits>                           <!--  byte addressable memory  -->
  <width>32</width>                                              <!--  bus width is 32 bits  -->
  <size>32</size>                                                <!--  this is the default size (number of bits) of all peripherals  -->
                                                                 <!--  and register that do not define "size" themselves            -->
  <resetValue>0x00000000</resetValue>                            <!--  by default all bits of the registers are initialized to 0 on reset  -->
  <resetMask>0xFFFFFFFF</resetMask>                              <!--  by default all 32Bits of the registers are used  -->
    

  <peripherals>

    <!--  Timer 0  -->
    <peripheral>
      <name>TIMER0</name>
      <version>1.0</version>
      <description>32 Timer / Counter, counting up or down from different sources</description>
      <groupName>TIMER</groupName>
      <headerStructName>TIMER</headerStructName>
      <baseAddress>0x40010000</baseAddress>
      <size>32</size>
      <access>read-write</access>

      <addressBlock>
        <offset>0</offset>
        <size>0x2000</size>
        <usage>registers</usage>
      </addressBlock>

      <interrupt>
        <name>TIMER0</name>
		<description>Timer 0 Interrupt</description>
        <value>0</value>
      </interrupt>

      <registers>
      <!--  CR: Control Register  -->
      
        <!-- register type="TypeA">
          <offset>0x20</offset>
        </register -->
                
        <register>
          <name>CR</name>
          <description>Control Register \n Second description Line of CR with Register- \n description</description>
          <addressOffset>0x00</addressOffset>
          <size>32</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0x1337F7F</resetMask>

          <fields>
            <!--  EN: Enable  -->
            <field>
              <name>EN</name>
              <description>Enable \n enables the Timer</description>
              <bitRange>[0:0]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
			    <name>EN_Values</name>
                <enumeratedValue>
                  <name>Disable</name>
                  <description>Timer is disabled \n and does not operate</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enable</name>
                  <description>Timer is enabled and can operate</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            
            
            <!--  RST: Reset  -->
            <field>
              <name>RST</name>
              <description>Reset Timer</description>
              <bitRange>[1:1]</bitRange>
              <access>write-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Reserved</name>
                  <description>Write as ZERO if necessary</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reset Timer</name>
                  <description>Reset the Timer</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  CNT: Counting Direction  -->
            <field>
              <name>CNT</name>
              <description>Counting direction</description>
              <bitRange>[3:2]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Count UP</name>
                  <description>Timer Counts UO and wraps, if no STOP condition is set</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Count DOWN</name>
                  <description>Timer Counts DOWN and wraps, if no STOP condition is set</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Toggle</name>
                  <description>Timer Counts up to MAX, then DOWN to ZERO, if no STOP condition is set</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  MODE: Operation Mode  -->
            <field>
              <name>MODE</name>
              <description>Operation Mode</description>
              <bitRange>[6:4]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Continous</name>
                  <description>Timer runs continously</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Single, ZERO/MAX</name>
                  <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT) and stops</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Single, MATCH</name>
                  <description>Timer counts to the Value of MATCH Register and stops</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reload, ZERO/MAX</name>
                  <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT), loads the RELOAD Value and continues</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reload, MATCH</name>
                  <description>Timer counts to the Value of MATCH Register, loads the RELOAD Value and continues</description>
                  <value>4</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  PSC: Use Prescaler  -->
            <field>
              <name>PSC</name>
              <description>Use Prescaler</description>
              <bitRange>[7:7]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Disabled</name>
                  <description>Prescaler is not used</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enabled</name>
                  <description>Prescaler is used as divider</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  CNTSRC: Timer / Counter Soruce Divider  -->
            <field>
              <name>CNTSRC</name>
              <description>Timer / Counter Source Divider</description>
              <bitRange>[11:8]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>CAP SRC</name>
                  <description>Capture Source is used directly</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 2</name>
                  <description>Capture Source is divided by 2</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 4</name>
                  <description>Capture Source is divided by 4</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 8</name>
                  <description>Capture Source is divided by 8</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 16</name>
                  <description>Capture Source is divided by 16</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 32</name>
                  <description>Capture Source is divided by 32</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 64</name>
                  <description>Capture Source is divided by 64</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 128</name>
                  <description>Capture Source is divided by 128</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 256</name>
                  <description>Capture Source is divided by 256</description>
                  <value>8</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  CAPSRC: Timer / COunter Capture Source  -->
            <field>
              <name>CAPSRC</name>
              <description>Timer / Counter Capture Source</description>
              <bitRange>[15:12]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>CClk</name>
                  <description>Core Clock</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_0</name>
                  <description>GPIO A, PIN 0</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_1</name>
                  <description>GPIO A, PIN 1</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_2</name>
                  <description>GPIO A, PIN 2</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_3</name>
                  <description>GPIO A, PIN 3</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_4</name>
                  <description>GPIO A, PIN 4</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_5</name>
                  <description>GPIO A, PIN 5</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_6</name>
                  <description>GPIO A, PIN 6</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_7</name>
                  <description>GPIO A, PIN 7</description>
                  <value>8</value>
                </enumeratedValue>

                <enumeratedValue>
                  <name>GPIOB_0</name>
                  <description>GPIO B, PIN 0</description>
                  <value>9</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_1</name>
                  <description>GPIO B, PIN 1</description>
                  <value>10</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_2</name>
                  <description>GPIO B, PIN 2</description>
                  <value>11</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_3</name>
                  <description>GPIO B, PIN 3</description>
                  <value>12</value>
                </enumeratedValue>

                <enumeratedValue>
                  <name>GPIOC_0</name>
                  <description>GPIO C, PIN 0</description>
                  <value>13</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOC_5</name>
                  <description>GPIO C, PIN 1</description>
                  <value>14</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOC_6</name>
                  <description>GPIO C, PIN 2</description>
                  <value>15</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  CAPEDGE: Capture Edge  -->
            <field>
              <name>CAPEDGE</name>
              <description>Capture Edge, select which Edge should result in a counter increment or decrement</description>
              <bitRange>[17:16]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RISING</name>
                  <description>Only rising edges result in a counter increment or decrement</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>FALLING</name>
                  <description>Only falling edges  result in a counter increment or decrement</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>BOTH</name>
                  <description>Rising and falling edges result in a counter increment or decrement</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  TRGEXT: Triggers an other Peripheral  -->
            <field>
              <name>TRGEXT</name>
              <description>Triggers an other Peripheral</description>
              <bitRange>[21:20]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>NONE</name>
                  <description>No Trigger is emitted</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DMA1</name>
                  <description>DMA Controller 1 is triggered, dependant on MODE</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DMA2</name>
                  <description>DMA Controller 2 is triggered, dependant on MODE</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>UART</name>
                  <description>UART is triggered, dependant on MODE</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  Reload: Selects Reload Register n  -->
            <field>
              <name>RELOAD</name>
              <description>Select RELOAD Register n to reload Timer on condition</description>
              <bitRange>[25:24]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RELOAD0</name>
                  <description>Selects Reload Register number 0</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD1</name>
                  <description>Selects Reload Register number 1</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD2</name>
                  <description>Selects Reload Register number 2</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD3</name>
                  <description>Selects Reload Register number 3</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  IDR: Inc or dec Reload Register Selection  -->
            <field>
              <name>IDR</name>
              <description>Selects, if Reload Register number is incremented, decremented or not modified</description>
              <bitRange>[27:26]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>KEEP</name>
                  <description>Reload Register number does not change automatically</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>INCREMENT</name>
                  <description>Reload Register number is incremented on each match</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DECREMENT</name>
                  <description>Reload Register number is decremented on each match</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            
            <field> 
              <name>reSeRveD</name>
              <description>Reserved Register</description>
              <bitRange>[29:28]</bitRange>
            </field>
            
            <!-- field>
              <name>EN_FOO</name>
              <description>Enable Foo</description>
              <bitRange>[30:30]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <usage>read</usage>
                <enumeratedValue>
                  <name>enabled</name>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>disabled</name>
                  <value>0</value>
                </enumeratedValue>
              </enumeratedValues>
              <enumeratedValues>
                <usage>write</usage>
                <enumeratedValue>
                  <name>enable interrupt</name>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>ignored</name>
                  <value>0</value>
                </enumeratedValue>
              </enumeratedValues>
            </field -->

            <!--  START: Starts / Stops the Timer/Counter  -->
            <field>
              <name>START</name>
              <description>Starts and Stops the Timer / Counter</description>
              <bitRange>[31:31]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>STOP</name>
                  <description>Timer / Counter is stopped</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>START</name>
                  <description>Timer / Counter is started</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
        </register>

        <!--  SR: Status Register  -->
        <register>
          <name>SR</name>
          <description>Status Register</description>
          <addressOffset>0x04</addressOffset>
          <size>16</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xD701</resetMask>

          <fields>
            <!--  RUN: Shows if Timer is running  -->
            <field>
              <name>RUN</name>
              <description>Shows if Timer is running or not</description>
              <bitRange>[0:0]</bitRange>
              <access>read-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Not Running</name>
                  <description>Timer is not running</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Running</name>
                  <description>Timer is running</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  MATCH: Shows if a Match was hit  -->
            <field>
              <name>MATCH</name>
              <description>Shows if the MATCH was hit</description>
              <bitRange>[8:8]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>No Match</name>
                  <description>The MATCH condition was not hit</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Match Hit</name>
                  <description>The MATCH condition was hit</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  UN: Shows if an underflow occured  -->
            <field>
              <name>UN</name>
              <description>Shows if an underflow occured. This flag is sticky</description>
              <bitRange>[9:9]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>No Underflow</name>
                  <description>No underflow occured since last clear</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Underflow occured</name>
                  <description>A minimum of one underflow occured since last clear</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  OV: Shows if an overflow occured  -->
            <field>
              <name>OV</name>
              <description>Shows if an overflow occured. This flag is sticky</description>
              <bitRange>[10:10]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>No Overflow</name>
                  <description>No overflow occured since last clear</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Overflow occured</name>
                  <description>A minimum of one overflow occured since last clear</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  RST: Shows if Timer is in RESET state  -->
            <field>
              <name>RST</name>
              <description>Shows if Timer is in RESET state</description>
              <bitRange>[12:12]</bitRange>
              <access>read-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Timer can operate</name>
                  <description>Timer is not in RESET state and can operate</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reset State</name>
                  <description>Timer is in RESET state and can not operate</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  RELOAD: Shows the currently active Reload Register  -->
            <field>
              <name>RELOAD</name>
              <description>Shows the currently active RELOAD Register</description>
              <bitRange>[15:14]</bitRange>
              <access>read-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RELOAD0</name>
                  <description>Reload Register number 0 is active</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD1</name>
                  <description>Reload Register number 1 is active</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD2</name>
                  <description>Reload Register number 2 is active</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD3</name>
                  <description>Reload Register number 3 is active</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
        </register>

        <!--  INT: Interrupt Register  -->
        <register>
          <name>INT</name>
          <description>Interrupt Register</description>
          <addressOffset>0x10</addressOffset>
          <size>16</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0x0771</resetMask>

          <fields>
            <!--  EN: Interrupt Enable  -->
            <field>
              <name>EN</name>
              <description>Interrupt Enable</description>
              <bitRange>[0:0]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Disabled</name>
                  <description>Timer does not generate Interrupts</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enable</name>
                  <description>Timer triggers the TIMERn Interrupt</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  MODE: Interrupt Mode  -->
            <field>
              <name>MODE</name>
              <description>Interrupt Mode, selects on which condition the Timer should generate an Interrupt</description>
              <bitRange>[6:4]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Match</name>
                  <description>Timer generates an Interrupt when the MATCH condition is hit</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Underflow</name>
                  <description>Timer generates an Interrupt when it underflows</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Overflow</name>
                  <description>Timer generates an Interrupt when it overflows</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
        </register>

        <!--  COUNT: Counter Register  -->
        <register>
          <name>COUNT</name>
          <description>The Counter Register reflects the actual Value of the Timer/Counter</description>
          <addressOffset>0x20</addressOffset>
          <size>32</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>

        <register>
          <name>COUNT_SIGNED</name>
          <description>The Prescale Register stores the Value for the prescaler. The cont event gets divided by this value</description>
          <alternateRegister>COUNT</alternateRegister>
          <addressOffset>0x20</addressOffset>
          <size>32</size>          
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <dataType>int32_t</dataType>
        </register>

        <!--  MATCH: Match Register  -->
        <register>
          <name>MATCH</name>
          <description>The Match Register stores the compare Value for the MATCH condition</description>
          <addressOffset>0x24</addressOffset>
          <size>32</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>
        
        <!--  MATCH: Match Register  -->
        <register>
          <name>MATCH2</name>
          <description>The Match Register stores the compare Value for the MATCH condition</description>
          <alternateRegister>MATCH</alternateRegister>
          <addressOffset>0x24</addressOffset>
          <size>32</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>

        <!--  PRESCALE: Prescale Read Register  -->
        <register>
          <name>PRESCALE_RD</name>
          <description>The Prescale Register stores the Value for the prescaler. The cont event gets divided by this value</description>
          <addressOffset>0x28</addressOffset>
          <size>32</size>
          <access>read-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>
        
        <!--  PRESCALE: Prescale Write Register  -->
        <register>
          <name>PRESCALE_WR</name>
          <description>The Prescale Register stores the Value for the prescaler. The cont event gets divided by this value</description>
          <addressOffset>0x28</addressOffset>
          <size>32</size>
          <access>write-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>
		
		
        <!--  EnumDeriveTest  -->
        <register>
          <name>EnumContDeriveTest</name>
          <description>Test for Enumerates Values Container deriveFrom</description>
          <addressOffset>0x30</addressOffset>
          <size>32</size>
          <access>write-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>Foo</name>
              <description>Foo Register</description>
              <bitRange>[0:0]</bitRange>
              <access>read-write</access>
			  <enumeratedValues derivedFrom="TIMER0.CR.EN.EN_Values">
			  </enumeratedValues>
            </field>
		  </fields>
        </register>

        <!--  RELOAD: Array of Reload Register with 4 elements -->
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0,1,2,3</dimIndex>
          <name>RELOAD[%s]</name>
          <description>The Reload Register stores the Value the COUNT Register gets reloaded on a when a condition was met.</description>
          <addressOffset>0x50</addressOffset>
          <size>32</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          
          <fields>
            <!--  Data  -->
            <field>
              <name>DATA</name>
              <description>Data Register</description>
              <bitRange>[23:0]</bitRange>
              <access>read-write</access>
            </field>
            <field>
              <name>CONF</name>
              <description>Data Conf</description>
              <bitRange>[30:27]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>1</value>
                </enumeratedValue>
                  <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>8</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>9</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>10</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>11</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>12</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>13</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>14</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Value Size</name>
                  <description>Size of the Reload Value</description>
                  <value>15</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            <field>
              <name>ACTIVE</name>
              <description>Data Active</description>
              <bitRange>[31:31]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Inavtive</name>
                  <description>Reload is inactive</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Avtive</name>
                  <description>Reload is active</description>
                  <value>1</value>
                </enumeratedValue>                
              </enumeratedValues>
            </field>
          </fields>
          
        </register>
        
        <!--  RELOAD2: Array of Reload Register with 4 elements -->
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <!-- dimIndex>0,1,2,3</dimIndex -->
          <name>RELOAD_NO_DIMINDEX[%s]</name>
          <description>The Reload2 Register stores the Value the COUNT Register gets reloaded on a when a condition was met.</description>
          <alternateRegister>RELOAD[%s]</alternateRegister>          
          <addressOffset>0x50</addressOffset>
          <size>32</size>
          <access>read-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>


        <!--  This Registers will create a union  -->
        <!--  VALUE MODEA: Value Mode A Register  -->
        <register>
          <name>VALUE_MODE_8</name>
          <displayName>VALUE Mode 8</displayName>
          <description>Value in Timer/Counter Mode 8</description>
          <addressOffset>0x60</addressOffset>
          <size>8</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>Value</name>
              <description>Value in 8Bit Mode</description>
              <bitRange>[7:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <!--  VALUE MODEA: Value Mode B Register  -->
        <register>
          <name>VALUE_MODE_16</name>
          <displayName>VALUE Mode 16</displayName>
          <description>Value in Timer/Counter Mode 16</description>
          <alternateRegister>VALUE_MODE_8</alternateRegister>
          <addressOffset>0x60</addressOffset>          
          <size>16</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>Value</name>
              <description>Value in 16Bit Mode</description>
              <bitRange>[15:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <!--  VALUE MODEA: Value Mode C Register  -->
        <register>
          <name>VALUE_MODE_24</name>
          <displayName>VALUE Mode 24</displayName>
          <description>Value in Timer/Counter Mode 24</description>
          <alternateRegister>VALUE_MODE_8</alternateRegister>
          <addressOffset>0x60</addressOffset>          
          <size>24</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>Value</name>
              <description>Value in 24Bit Mode</description>
              <bitRange>[23:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <!--  VALUE MODEA: Value Mode D Register  -->
        <register>
          <name>VALUE_MODE_32</name>
          <displayName>VALUE Mode 32</displayName>
          <description>Value in Timer/Counter Mode 32</description>
          <alternateRegister>VALUE_MODE_8</alternateRegister>
          <addressOffset>0x60</addressOffset>          
          <size>32</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>Value</name>
              <description>Value in 32Bit Mode</description>
              <bitRange>[31:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>
    
        <!--  Read Action Test  -->
		<register>
          <name>NoReadAction</name>
          <description>This Register has no Read action</description>
          <addressOffset>0x1800</addressOffset>
          <size>32</size>
          <access>write-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
        </register>
        <register>
          <name>ReadAction</name>
          <description>This Register has a Read action</description>
          <addressOffset>0x1804</addressOffset>
          <size>32</size>
          <access>write-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
		  <readAction>modify</readAction>
        </register>
        <register>
          <name>ReadActionField</name>
          <description>This Register has a Read action</description>
          <addressOffset>0x1808</addressOffset>
          <size>32</size>
          <access>write-only</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
		  <fields>
            <field>
              <name>NoReadAction</name>
              <description>Field has noi Read Action</description>
              <bitRange>[7:0]</bitRange>
              <access>read-write</access>
            </field>
			<field>
              <name>ReadAction</name>
              <description>Field has Read Action</description>
              <bitRange>[15:8]</bitRange>
              <access>read-write</access>
			  <readAction>modify</readAction>
            </field>
          </fields>		  
        </register>		
   
    <!--  --------------------   Cluster Test  ----------------------------------------  -->
    <!--  --------------------   Cluster Test  ----------------------------------------  -->
    <!--  --------------------   Cluster Test  ----------------------------------------  -->
    <!--  --------------------   Cluster Test  ----------------------------------------  -->
    <cluster>
      <dim>4</dim>
      <dimIncrement>0x20</dimIncrement>
      <name>Cluster_1[%s]</name>
      <headerStructName>MyClust</headerStructName>
      <description>Test Cluster Instance 1</description>
      <addressOffset>0x70</addressOffset>
      <register>
        <dim>4</dim>
        <dimIncrement>0x04</dimIncrement>
        <name>CLUSTER_REG_A[%s]</name>
        <description>Cluster Test Register A</description>
        <addressOffset>0x00</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
        <fields>
            <field>
              <name>EN</name>
              <description>Enable</description>
              <bitRange>[0:0]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Disable</name>
                  <description>Timer is disabled and does not operate</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enable</name>
                  <description>Timer is enabled and can operate</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            
            
            <field>
              <name>RST</name>
              <description>Reset Timer</description>
              <bitRange>[1:1]</bitRange>
              <access>write-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Reserved</name>
                  <description>Write as ZERO if necessary</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reset Timer</name>
                  <description>Reset the Timer</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CNT</name>
              <description>Counting direction</description>
              <bitRange>[3:2]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Count UP</name>
                  <description>Timer Counts UO and wraps, if no STOP condition is set</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Count DOWN</name>
                  <description>Timer Counts DOWN and wraps, if no STOP condition is set</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Toggle</name>
                  <description>Timer Counts up to MAX, then DOWN to ZERO, if no STOP condition is set</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>MODE</name>
              <description>Operation Mode</description>
              <bitRange>[6:4]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Continous</name>
                  <description>Timer runs continously</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Single, ZERO/MAX</name>
                  <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT) and stops</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Single, MATCH</name>
                  <description>Timer counts to the Value of MATCH Register and stops</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reload, ZERO/MAX</name>
                  <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT), loads the RELOAD Value and continues</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reload, MATCH</name>
                  <description>Timer counts to the Value of MATCH Register, loads the RELOAD Value and continues</description>
                  <value>4</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>PSC</name>
              <description>Use Prescaler</description>
              <bitRange>[7:7]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Disabled</name>
                  <description>Prescaler is not used</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enabled</name>
                  <description>Prescaler is used as divider</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CNTSRC</name>
              <description>Timer / Counter Source Divider</description>
              <bitRange>[11:8]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>CAP SRC</name>
                  <description>Capture Source is used directly</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 2</name>
                  <description>Capture Source is divided by 2</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 4</name>
                  <description>Capture Source is divided by 4</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 8</name>
                  <description>Capture Source is divided by 8</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 16</name>
                  <description>Capture Source is divided by 16</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 32</name>
                  <description>Capture Source is divided by 32</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 64</name>
                  <description>Capture Source is divided by 64</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 128</name>
                  <description>Capture Source is divided by 128</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 256</name>
                  <description>Capture Source is divided by 256</description>
                  <value>8</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CAPSRC</name>
              <description>Timer / Counter Capture Source</description>
              <bitRange>[15:12]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>CClk</name>
                  <description>Core Clock</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_0</name>
                  <description>GPIO A, PIN 0</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_1</name>
                  <description>GPIO A, PIN 1</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_2</name>
                  <description>GPIO A, PIN 2</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_3</name>
                  <description>GPIO A, PIN 3</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_4</name>
                  <description>GPIO A, PIN 4</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_5</name>
                  <description>GPIO A, PIN 5</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_6</name>
                  <description>GPIO A, PIN 6</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_7</name>
                  <description>GPIO A, PIN 7</description>
                  <value>8</value>
                </enumeratedValue>

                <enumeratedValue>
                  <name>GPIOB_0</name>
                  <description>GPIO B, PIN 0</description>
                  <value>9</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_1</name>
                  <description>GPIO B, PIN 1</description>
                  <value>10</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_2</name>
                  <description>GPIO B, PIN 2</description>
                  <value>11</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_3</name>
                  <description>GPIO B, PIN 3</description>
                  <value>12</value>
                </enumeratedValue>

                <enumeratedValue>
                  <name>GPIOC_0</name>
                  <description>GPIO C, PIN 0</description>
                  <value>13</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOC_5</name>
                  <description>GPIO C, PIN 1</description>
                  <value>14</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOC_6</name>
                  <description>GPIO C, PIN 2</description>
                  <value>15</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CAPEDGE</name>
              <description>Capture Edge, select which Edge should result in a counter increment or decrement</description>
              <bitRange>[17:16]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RISING</name>
                  <description>Only rising edges result in a counter increment or decrement</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>FALLING</name>
                  <description>Only falling edges  result in a counter increment or decrement</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>BOTH</name>
                  <description>Rising and falling edges result in a counter increment or decrement</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>TRGEXT</name>
              <description>Triggers an other Peripheral</description>
              <bitRange>[21:20]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>NONE</name>
                  <description>No Trigger is emitted</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DMA1</name>
                  <description>DMA Controller 1 is triggered, dependant on MODE</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DMA2</name>
                  <description>DMA Controller 2 is triggered, dependant on MODE</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>UART</name>
                  <description>UART is triggered, dependant on MODE</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>RELOAD</name>
              <description>Select RELOAD Register n to reload Timer on condition</description>
              <bitRange>[25:24]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RELOAD0</name>
                  <description>Selects Reload Register number 0</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD1</name>
                  <description>Selects Reload Register number 1</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD2</name>
                  <description>Selects Reload Register number 2</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD3</name>
                  <description>Selects Reload Register number 3</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>IDR</name>
              <description>Selects, if Reload Register number is incremented, decremented or not modified</description>
              <bitRange>[27:26]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>KEEP</name>
                  <description>Reload Register number does not change automatically</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>INCREMENT</name>
                  <description>Reload Register number is incremented on each match</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DECREMENT</name>
                  <description>Reload Register number is decremented on each match</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            
            <field> 
              <name>reSeRveD</name>
              <description>Reserved Register</description>
              <bitRange>[29:28]</bitRange>
            </field>

            <field>
              <name>START</name>
              <description>Starts and Stops the Timer / Counter</description>
              <bitRange>[31:31]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>STOP</name>
                  <description>Timer / Counter is stopped</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>START</name>
                  <description>Timer / Counter is started</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
      </register>
      <register>            
        <name>CLUSTER_REG_B</name>
        <description>Cluster Test Register B</description>
        <addressOffset>0x00</addressOffset>
        <alternateRegister>CLUSTER_REG_A0</alternateRegister>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      <register>            
        <name>CLUSTER_REG_C</name>
        <description>Cluster Test Register C</description>
        <addressOffset>0x00</addressOffset>
        <alternateRegister>CLUSTER_REG_A0</alternateRegister>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
        
        <fields>
            <field>
              <name>EN</name>
              <description>Enable</description>
              <bitRange>[0:0]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Disable</name>
                  <description>Timer is disabled and does not operate</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enable</name>
                  <description>Timer is enabled and can operate</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            
            
            <field>
              <name>RST</name>
              <description>Reset Timer</description>
              <bitRange>[1:1]</bitRange>
              <access>write-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Reserved</name>
                  <description>Write as ZERO if necessary</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reset Timer</name>
                  <description>Reset the Timer</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CNT</name>
              <description>Counting direction</description>
              <bitRange>[3:2]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Count UP</name>
                  <description>Timer Counts UO and wraps, if no STOP condition is set</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Count DOWN</name>
                  <description>Timer Counts DOWN and wraps, if no STOP condition is set</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Toggle</name>
                  <description>Timer Counts up to MAX, then DOWN to ZERO, if no STOP condition is set</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>MODE</name>
              <description>Operation Mode</description>
              <bitRange>[6:4]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Continous</name>
                  <description>Timer runs continously</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Single, ZERO/MAX</name>
                  <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT) and stops</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Single, MATCH</name>
                  <description>Timer counts to the Value of MATCH Register and stops</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reload, ZERO/MAX</name>
                  <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT), loads the RELOAD Value and continues</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reload, MATCH</name>
                  <description>Timer counts to the Value of MATCH Register, loads the RELOAD Value and continues</description>
                  <value>4</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>PSC</name>
              <description>Use Prescaler</description>
              <bitRange>[7:7]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Disabled</name>
                  <description>Prescaler is not used</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Enabled</name>
                  <description>Prescaler is used as divider</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CNTSRC</name>
              <description>Timer / Counter Source Divider</description>
              <bitRange>[11:8]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>CAP SRC</name>
                  <description>Capture Source is used directly</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 2</name>
                  <description>Capture Source is divided by 2</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 4</name>
                  <description>Capture Source is divided by 4</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 8</name>
                  <description>Capture Source is divided by 8</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 16</name>
                  <description>Capture Source is divided by 16</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 32</name>
                  <description>Capture Source is divided by 32</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 64</name>
                  <description>Capture Source is divided by 64</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 128</name>
                  <description>Capture Source is divided by 128</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>CAP SRC / 256</name>
                  <description>Capture Source is divided by 256</description>
                  <value>8</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CAPSRC</name>
              <description>Timer / Counter Capture Source</description>
              <bitRange>[15:12]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>CClk</name>
                  <description>Core Clock</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_0</name>
                  <description>GPIO A, PIN 0</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_1</name>
                  <description>GPIO A, PIN 1</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_2</name>
                  <description>GPIO A, PIN 2</description>
                  <value>3</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_3</name>
                  <description>GPIO A, PIN 3</description>
                  <value>4</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_4</name>
                  <description>GPIO A, PIN 4</description>
                  <value>5</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_5</name>
                  <description>GPIO A, PIN 5</description>
                  <value>6</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_6</name>
                  <description>GPIO A, PIN 6</description>
                  <value>7</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOA_7</name>
                  <description>GPIO A, PIN 7</description>
                  <value>8</value>
                </enumeratedValue>

                <enumeratedValue>
                  <name>GPIOB_0</name>
                  <description>GPIO B, PIN 0</description>
                  <value>9</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_1</name>
                  <description>GPIO B, PIN 1</description>
                  <value>10</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_2</name>
                  <description>GPIO B, PIN 2</description>
                  <value>11</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOB_3</name>
                  <description>GPIO B, PIN 3</description>
                  <value>12</value>
                </enumeratedValue>

                <enumeratedValue>
                  <name>GPIOC_0</name>
                  <description>GPIO C, PIN 0</description>
                  <value>13</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOC_5</name>
                  <description>GPIO C, PIN 1</description>
                  <value>14</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>GPIOC_6</name>
                  <description>GPIO C, PIN 2</description>
                  <value>15</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>CAPEDGE</name>
              <description>Capture Edge, select which Edge should result in a counter increment or decrement</description>
              <bitRange>[17:16]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RISING</name>
                  <description>Only rising edges result in a counter increment or decrement</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>FALLING</name>
                  <description>Only falling edges  result in a counter increment or decrement</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>BOTH</name>
                  <description>Rising and falling edges result in a counter increment or decrement</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>TRGEXT</name>
              <description>Triggers an other Peripheral</description>
              <bitRange>[21:20]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>NONE</name>
                  <description>No Trigger is emitted</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DMA1</name>
                  <description>DMA Controller 1 is triggered, dependant on MODE</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DMA2</name>
                  <description>DMA Controller 2 is triggered, dependant on MODE</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>UART</name>
                  <description>UART is triggered, dependant on MODE</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>RELOAD</name>
              <description>Select RELOAD Register n to reload Timer on condition</description>
              <bitRange>[25:24]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>RELOAD0</name>
                  <description>Selects Reload Register number 0</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD1</name>
                  <description>Selects Reload Register number 1</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD2</name>
                  <description>Selects Reload Register number 2</description>
                  <value>2</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>RELOAD3</name>
                  <description>Selects Reload Register number 3</description>
                  <value>3</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>IDR</name>
              <description>Selects, if Reload Register number is incremented, decremented or not modified</description>
              <bitRange>[27:26]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>KEEP</name>
                  <description>Reload Register number does not change automatically</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>INCREMENT</name>
                  <description>Reload Register number is incremented on each match</description>
                  <value>1</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>DECREMENT</name>
                  <description>Reload Register number is decremented on each match</description>
                  <value>2</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            
            <field> 
              <name>reSeRveD</name>
              <description>Reserved Register</description>
              <bitRange>[29:28]</bitRange>
            </field>

            <field>
              <name>START</name>
              <description>Starts and Stops the Timer / Counter</description>
              <bitRange>[31:31]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>STOP-1</name>
                  <description>Timer / Counter is stopped</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>START</name>
                  <description>Timer / Counter is started</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
      </register>
      <register>            
        <dim>4</dim>
        <dimIncrement>4</dimIncrement>
        <name>CLUSTER_REG_D[%s]</name>
        <description>Cluster Test Register D</description>
        <addressOffset>0x00</addressOffset>
        <alternateRegister>CLUSTER_REG_A[%s]</alternateRegister>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
    </cluster>
    
      <register>
        <dim>4</dim>
        <dimIncrement>0x04</dimIncrement>
        <name>REG_UNION_CLUSTER[%s]</name>
        <description>Register in union with Cluster</description>
        <alternateRegister>Cluster_1.CLUSTER_REG_A[%s]</alternateRegister>
        <addressOffset>0x70</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      
      <register>
        <dim>4</dim>
        <dimIncrement>0x04</dimIncrement>
        <name>REG_UNION_CLUSTERy[%s]</name>
        <description>Register in union with Cluster</description>
        <alternateRegister>REG_UNION_CLUSTER[%s]</alternateRegister>
        <addressOffset>0x70</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      
    <cluster>
      <dim>4</dim>
      <dimIncrement>0x100</dimIncrement>
      <dimIndex>A,B,C,D</dimIndex>
      <name>Cluster_2%s</name>
      <description>Test Cluster Instance 1</description>
      <addressOffset>0x70</addressOffset>
      <alternateCluster>Cluster_1</alternateCluster>
      <register>
        <name>CLUSTER_REG_A</name>
        <description>Cluster Test Register A</description>
        <addressOffset>0x00</addressOffset>
        <alternateRegister>Cluster_1</alternateRegister>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      <register>            
        <name>CLUSTER_REG_B</name>
        <description>Cluster Test Register B</description>
        <addressOffset>0x04</addressOffset>
        <alternateRegister>Cluster_1</alternateRegister>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      <register>            
        <name>CLUSTER_REG_C</name>
        <description>Cluster Test Register C</description>
        <addressOffset>0x08</addressOffset>
        <alternateRegister>Cluster_1</alternateRegister>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      <register>
        <dim>4</dim>
        <dimIncrement>4</dimIncrement>
        <name>CLUSTER_REG_D[%s]</name>
        <description>Cluster Test Register D</description>
        <addressOffset>0x0c</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
    </cluster>
    
    <cluster>
      <dim>4</dim>
      <dimIncrement>0x04</dimIncrement>
      <name>Cluster_3[%s]</name>
      <description>Test Cluster Instance 3</description>
      <addressOffset>0x70</addressOffset>
      <alternateCluster>Cluster_1</alternateCluster>
      
      <register>
        <dim>4</dim>
        <dimIncrement>4</dimIncrement>
        <dimIndex>_a,_b,_c,_d</dimIndex>
        <name>CLUSTER_REG_A%s</name>
        <description>Cluster Test Register A</description>
        <addressOffset>0x00</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
      
      <register>
        <dim>4</dim>
        <dimIncrement>4</dimIncrement>
        <dimIndex>_Alt_a,_Alt_b,_Alt_c,_Alt_d</dimIndex>
        <name>CLUSTER_REG_B%s</name>      
        <description>Cluster Test Register B</description>
        <addressOffset>0x00</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
    </cluster>
    
    <cluster derivedFrom="Cluster_3[%s]">
      <name>Cluster_4[%s]</name>
      <description>Test Cluster Instance 4</description>
      <addressOffset>0x70</addressOffset>
      <alternateCluster>Cluster_2</alternateCluster>
    </cluster>

      </registers>
    </peripheral>

    <!--  Timer 1  -->
    <peripheral derivedFrom="TIMER0">
      <name>TIMER1</name>
      <baseAddress>0x40010100</baseAddress>
      <interrupt>
        <name>TIMER1</name>
        <value>5</value>
      </interrupt>
    </peripheral>

    <!--  Timer 2  -->
    <peripheral derivedFrom="TIMER0">
      <name>TIMER2</name>
      <baseAddress>0x40010200</baseAddress>
      <interrupt>
        <name>TIMER2</name>
        <value>6</value>
      </interrupt>
    </peripheral>

    <!--  Timer 3  -->
    <peripheral derivedFrom="TIMER0">
      <name>TIMER3</name>
      <baseAddress>0x40010300</baseAddress>
      <interrupt>
        <name>TIMER3</name>
        <value>7</value>
      </interrupt>

      <registers>
        <register derivedFrom="CR">
          <name>CR2</name>
          <description>Control Register for the second group.</description>
          <addressOffset>0x40</addressOffset>
        </register>

        <register derivedFrom="COUNT">
          <dim>4</dim>
          <dimIncrement>0x04</dimIncrement>
          <dimIndex>CA,CB,CC,CD</dimIndex>
          <name>COUNT_%s</name>
          <displayName>COUNT%s</displayName>
          <description>Set of count Registers</description>
          <addressOffset>0x500</addressOffset>

          <fields>
            <field>
              <name>ACTIVE</name>
              <description>Active Flag</description>
              <bitRange>[0:0]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Inactive</name>
                  <description>The Count Register is inactive</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Active</name>
                  <description>The Count Register is active</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <field>
              <name>COUNT</name>
              <description>Alternative Count Register 30Bit</description>
              <bitRange>[30:1]</bitRange>
              <access>read-write</access>
            </field>

            <field derivedFrom="ACTIVE">
              <name>ACTIVE2</name>
              <description>Alternate Active Flag</description>
              <bitRange>[31:31]</bitRange>
            </field>
          </fields>
        </register>
        
        <cluster>
          <dim>4</dim>
          <dimIncrement>0x20</dimIncrement>
          <name>Cluster_5[%s]</name>
          <headerStructName>Cluster5</headerStructName>
          <description>Test Cluster Instance 1</description>
          <addressOffset>0x170</addressOffset>
          <register>
            <dim>4</dim>
            <dimIncrement>0x04</dimIncrement>
            <name>CLUSTER_REG_A[%s]</name>
            <description>Cluster Test Register A</description>
            <addressOffset>0x00</addressOffset>
            <size>32</size>
            <access>read-write</access>
            <resetValue>0x00000000</resetValue>
            <resetMask>0xFFFFFFFF</resetMask>
          </register>
        </cluster>

      </registers>
    </peripheral>


    <!--  UART 0  -->
    <peripheral>
      <name>UART0</name>
      <version>1.0</version>
      <description>UART peripheral</description>
      <groupName>UART</groupName>
      <baseAddress>0x40020000</baseAddress>      
      <size>32</size>
      <access>read-write</access>

      <addressBlock>
        <offset>0</offset>
        <size>0x100</size>
        <usage>registers</usage>
      </addressBlock>

      <interrupt>
        <name>UART0</name>
        <value>10</value>
      </interrupt>

      <registers>
      <!--  DR: Data Register  -->
        <register>
          <name>DR</name>
          <description>Data Register</description>
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>

          <fields>
            <!--  Data  -->
            <field>
              <name>DATA</name>
              <description>Data Register</description>
              <bitRange>[7:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>

      <!--  DR: Data Register: ASCII Mode  -->
        <register>
          <name>DR</name>
          <description>Data Register: ASCII Mode</description>
          <alternateGroup>ASCII</alternateGroup>          
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>

          <fields>
            <!--  Data  -->
            <field>
              <name>DATA</name>
              <description>Data Register</description>
              <bitRange>[7:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>

        <!--  DR: Data Register: I2C Mode  -->
        <register>
          <name>DR</name>
          <description>Data Register: I2C Mode</description>
          <alternateGroup>I2C</alternateGroup>          
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>

          <fields>
            <!--  Data  -->
            <field>
              <name>DATA</name>
              <description>Data Register</description>
              <bitRange>[7:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>

        <!--  DR: Data Register: SPI Mode  -->
        <register>
          <name>DR</name>
          <description>Data Register: SPI Mode</description>
          <alternateGroup>SPI</alternateGroup>          
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>

          <fields>
            <!--  Data  -->
            <field>
              <name>DATA</name>
              <description>Data Register</description>
              <bitRange>[7:0]</bitRange>
              <access>read-write</access>
            </field>
          </fields>
        </register>

      </registers>
    </peripheral>
    

    <peripheral>
      <name>UART1</name>
      <version>1.0</version>
      <description>UART peripheral</description>
      <groupName>UART</groupName>
      <baseAddress>0x40021000</baseAddress>      
      <size>32</size>
      <access>read-write</access>

      <addressBlock>
        <offset>0</offset>
        <size>0x100</size>
        <usage>registers</usage>
      </addressBlock>

      <interrupt>
        <name>UART1</name>
        <value>11</value>
      </interrupt>
      
      <registers>
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0,1,2,3</dimIndex>
          <name>DR_A[%s]</name>
          <description>Data Register A</description>
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0,1,2,3</dimIndex>
          <name>DR_B[%s]</name>
          <description>Data Register B</description>
          <alternateRegister>DR_A[%s]</alternateRegister>          
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
        <register>
          <name>DR_C</name>
          <description>Data Register C</description>
          <alternateRegister>DR_A0</alternateRegister>          
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
      </registers>
  
    </peripheral>
    
 
    <!--  UART 3  -->
    <peripheral>
      <name>UART3</name>
      <version>1.0</version>
      <description>UART3 peripheral</description>
      <groupName>UART</groupName>
      <baseAddress>0x40030000</baseAddress>
      <size>32</size>
      <access>read-write</access>

      <addressBlock>
        <offset>0</offset>
        <size>0x1000</size>
        <usage>registers</usage>
      </addressBlock>

      <interrupt>
        <name>FOO</name>
        <value>20</value>
      </interrupt>
      

      <registers>
      <!--  DR: Data Register  -->
        <!--  -----  -->
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <!-- dimIndex>0,1,2,3</dimIndex -->
          <name>DR_A[%s]</name>
          <description>Data Register A</description>
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0,1,2,3</dimIndex>
          <name>DR_B[%s]</name>
          <description>Data Register B</description>
          <alternateRegister>DR_A[%s]</alternateRegister>          
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
        <!--  -----  -->
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>a,b,c,d</dimIndex>
          <name>DR_C%s</name>
          <description>Data Register C</description>
          <addressOffset>0x50</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>w,x,y,z</dimIndex>
          <name>DR_D%s</name>
          <description>Data Register D</description>
          <alternateRegister>DR_C%s</alternateRegister>          
          <addressOffset>0x50</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
        <!--  -----  -->
        <register>          
          <name>DR_E_A0</name>
          <description>Data Register E to A0</description>
          <alternateRegister>DR_A0</alternateRegister>
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        
      </registers>
    </peripheral>

    <!--  UART 4  -->
    <peripheral derivedFrom="UART3">
      <name>UART4</name>
      <version>1.0</version>
      <description>UART4 peripheral</description>
      <alternatePeripheral>UART3</alternatePeripheral>
      <groupName>UART</groupName>
      <baseAddress>0x40030000</baseAddress>
      <size>32</size>
      <access>read-write</access>

      <addressBlock>
        <offset>0</offset>
        <size>0x1000</size>
        <usage>registers</usage>
      </addressBlock>

      <interrupt>
        <name>BAR</name>
        <value>21</value>
      </interrupt>
      
    </peripheral>
	
	
	<peripheral>
      <name>TIMER5</name>
      <version>1.0</version>
      <description>32 Timer / Counter, counting up or down from different sources</description>
      <groupName>TIMER</groupName>
      <headerStructName>TIM5</headerStructName>
      <baseAddress>0x40060000</baseAddress>
      <size>32</size>
      <access>read-write</access>

      <addressBlock>
        <offset>0</offset>
        <size>0x400</size>
        <usage>registers</usage>
      </addressBlock>

      <interrupt>
        <name>TIMER5</name>
        <value>17</value>
      </interrupt>

      <registers>
	    <register>
		  <dim>4</dim>
		  <dimIncrement>0x04</dimIncrement>
		  <name>DR_M[%s]</name>
		  <description>Multiple Definitions</description>
		  <addressOffset>0x160</addressOffset>
		</register>
		
	  <register>
		  <dim>4</dim>
		  <dimIncrement>0x04</dimIncrement>
		  <dimIndex>A,B,C,D</dimIndex>
		  <name>DR_M%s</name>
		  <description>Multiple Definitions</description>
		  <addressOffset>0x170</addressOffset>
		</register>
		
	  <register derivedFrom="DR_M0">
		  <dim>4</dim>
		  <dimIncrement>0x04</dimIncrement>
		  <name>DR_M1[%s]</name>
		  <displayName>DR_M1[%s]</displayName>
		  <description>Multiple Definitions derive</description>
		  <addressOffset>0x150</addressOffset>
		</register>
		
		
    <cluster>
      <name>Cluster_A</name>
      <headerStructName>TIM5Clust</headerStructName>
      <description>Cluster A</description>
      <addressOffset>0x0</addressOffset>
      <register>
        <dim>4</dim>
        <dimIncrement>0x04</dimIncrement>
        <name>CLUSTER_REG_A[%s]</name>
        <description>Cluster Test Register A</description>
        <addressOffset>0x00</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
    </cluster>

    <cluster derivedFrom="Cluster_A">
      <name>Cluster_B</name>
      <description>Cluster B</description>
      <addressOffset>0x100</addressOffset>
    </cluster>
		
		<cluster derivedFrom="Cluster_A">
		  <dim>4</dim>
		  <dimIncrement>0x10</dimIncrement>
          <name>Cluster_C[%s]</name>
          <description>Cluster C</description>
          <addressOffset>0x100</addressOffset>
        </cluster>
	    
		<register derivedFrom="UART3.DR_E_A0">
		  <dim>4</dim>
		  <dimIncrement>0x04</dimIncrement>
		  <name>DR_E_A[%s]</name>
		  <description>Derived from UART3</description>
		  <addressOffset>0x140</addressOffset>
		  <alternateRegister>DR_M[%s]</alternateRegister>
		</register>
				
		<register derivedFrom="UART3.DR_A0">
		  <dim>4</dim>
		  <dimIncrement>0x04</dimIncrement>
		  <name>DR_A[%s]</name>
		  <description>Derived from UART3</description>
		  <addressOffset>0x160</addressOffset>
      <alternateRegister>DR_M%s</alternateRegister>
		</register>

   	
		<cluster derivedFrom="TIMER3.Cluster_5[%s]">
		  <addressOffset>0x180</addressOffset>
		</cluster>
		
		<cluster derivedFrom="Cluster_5[%s]">
		  <name>Cluster_6[%s]</name>
		  <addressOffset>0x180</addressOffset>
		</cluster>
		
		
		<register derivedFrom="TIMER3.Cluster_1[%s].CLUSTER_REG_B">
		  <addressOffset>0x200</addressOffset>
		</register>
		
		
		<register derivedFrom="Cluster_5[%s].CLUSTER_REG_A[%s]">
		  <addressOffset>0x210</addressOffset>
		</register>
		
		<register derivedFrom="CLUSTER_REG_A[%s]">
		  <name>RegC[%s]</name>
		  <addressOffset>0x220</addressOffset>
		</register>
		
		
      </registers>
	  	  
    </peripheral>
	
	
	
	<peripheral derivedFrom="TIMER5">
    <name>TIMER6</name>
    <version>1.0</version>
    <description>TIMER6 peripheral</description>
    <groupName>TIMER</groupName>
    <baseAddress>0x40031000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <interrupt>
      <name>TIMER6</name>
      <value>31</value>
    </interrupt>
	  
	  <registers>
	    <register derivedFrom="CLUSTER_REG_A[%s]">
		  <name>RegD[%s]</name>
		  <addressOffset>0x230</addressOffset>
      </register>
		
      <cluster derivedFrom="Cluster_6[%s]">
        <name>Cluster_7[%s]</name>
        <addressOffset>0x180</addressOffset>
      </cluster>
	
    </registers>      
  </peripheral>
    
    
  <peripheral>
    <name>SPI0</name>
    <version>1.11</version>
    <description>SPI master</description>
    <baseAddress>0x40003000</baseAddress>
    <groupName>SPI</groupName>
    <size>32</size>
    <access>read-write</access>
    <headerStructName>SPI</headerStructName>

    <addressBlock>
      <offset>0</offset>
      <size>0x1000</size>
      <usage>registers</usage>
    </addressBlock>

    <interrupt>
      <name>SPI0_TWI0</name>
      <value>3</value>
    </interrupt>

    <registers>

      <register>
        <name>EVENTS_READY</name>
        <description>TXD byte sent and RXD byte received.</description>
        <addressOffset>0x108</addressOffset>
      </register>

      <register>
        <name>INTENSET</name>
        <description>usage test: Interrupt enable set register.</description>
        <addressOffset>0x304</addressOffset>
        <fields>
          <field>
            <name>READY</name>
            <description>Enable interrupt on READY event.</description>
            <lsb>2</lsb> <msb>2</msb>
            <enumeratedValues>
              <usage>read</usage>
              <enumeratedValue>
                <name>Disabled</name>
                <description>Interrupt disabled.</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Enabled</name>
                <description>Interrupt enabled.</description>
                <value>1</value>
              </enumeratedValue>
            </enumeratedValues>
            <enumeratedValues>
              <usage>write</usage>
              <enumeratedValue>
                <name>Set</name>
                <description>Enable interrupt on write.</description>
                <value>1</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>
        </fields>
      </register>
    </registers>
  </peripheral>
  
  <peripheral derivedFrom="SPI0">
    <name>SPI1</name>
    <version>1.11</version>
    <description>SPI 1</description>
    <baseAddress>0x40004000</baseAddress>
    <groupName>SPI</groupName>
    <size>32</size>
    <access>read-write</access>
    <interrupt>
      <name>SPI1_TWI1</name>
      <value>4</value>
    </interrupt>
  </peripheral>
  
  
  <peripheral>
    <name>SPI5</name>
    <description>SPI 5</description>
    <baseAddress>0x40005000</baseAddress>
    <groupName>SPI</groupName>
    <access>read-write</access>
    <size>8</size>
    
    <addressBlock>
      <offset>0</offset>
      <size>0x100</size>
      <usage>registers</usage>
    </addressBlock>
    
    <interrupt>
      <name>SPI5</name>
      <value>1</value>    <!--  test -1 here -->
      <description>SPI5 Interrupt</description>
    </interrupt>
    
    <registers>
      <register>
        <name>CR</name>
        <description>Control Register</description>
        <addressOffset>0x00</addressOffset>
      </register>
      
      <register>
        <name>SR</name>
        <description>Status Register</description>
        <addressOffset>0x01</addressOffset>
      </register>
      
      <register>
        <name>DR</name>
        <description>Data Register</description>
        <addressOffset>0x02</addressOffset>
      </register>
      
    </registers>
    
  </peripheral>
  
  
  
  <peripheral>
    <name>TIMER10</name>
    <version>1.0</version>
    <description>32 Timer / Counter, counting up or down from different sources</description>
    <groupName>TIMER</groupName>
    <headerStructName>TIMER10</headerStructName>
    <baseAddress>0x400A0000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <addressBlock>
      <offset>0</offset>
      <size>0x2000</size>
      <usage>registers</usage>
    </addressBlock>
    
    <registers>
      <register>
        <dim>4</dim>
        <dimIncrement>4</dimIncrement>
        <name>CR[%s]</name>
        <description>Control Register</description>
        <addressOffset>0x00</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0x1337F7F</resetMask>

        <fields>
          <!--  EN: Enable  -->
          <field>
            <name>EN</name>
            <description>Enable \n enables the Timer</description>
            <bitRange>[0:0]</bitRange>
            <access>read-write</access>
            <enumeratedValues>
              <enumeratedValue>
                <name>Disable</name>
                <description>Timer is disabled \n and does not operate</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Enable</name>
                <description>Timer is enabled and can operate</description>
                <value>1</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>
          
          
          <!--  RST: Reset  -->
          <field>
            <name>RST</name>
            <description>Reset Timer</description>
            <bitRange>[1:1]</bitRange>
            <access>write-only</access>
            <enumeratedValues>
              <enumeratedValue>
                <name>Reserved</name>
                <description>Write as ZERO if necessary</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Reset Timer</name>
                <description>Reset the Timer</description>
                <value>1</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>

          <!--  CNT: Counting Direction  -->
          <field>
            <name>CNT</name>
            <description>Counting direction</description>
            <bitRange>[3:2]</bitRange>
            <access>read-write</access>
            <enumeratedValues>
              <enumeratedValue>
                <name>Count UP</name>
                <description>Timer Counts UO and wraps, if no STOP condition is set</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Count DOWN</name>
                <description>Timer Counts DOWN and wraps, if no STOP condition is set</description>
                <value>1</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Toggle</name>
                <description>Timer Counts up to MAX, then DOWN to ZERO, if no STOP condition is set</description>
                <value>2</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>

          <!--  MODE: Operation Mode  -->
          <field>
            <name>MODE</name>
            <description>Operation Mode</description>
            <bitRange>[6:4]</bitRange>
            <access>read-write</access>
            <enumeratedValues>
              <enumeratedValue>
                <name>Continous</name>
                <description>Timer runs continously</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Single, ZERO/MAX</name>
                <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT) and stops</description>
                <value>1</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Single, MATCH</name>
                <description>Timer counts to the Value of MATCH Register and stops</description>
                <value>2</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Reload, ZERO/MAX</name>
                <description>Timer counts to 0x00 or 0xFFFFFFFF (depending on CNT), loads the RELOAD Value and continues</description>
                <value>3</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Reload, MATCH</name>
                <description>Timer counts to the Value of MATCH Register, loads the RELOAD Value and continues</description>
                <value>4</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>

          <!--  PSC: Use Prescaler  -->
          <field>
            <name>PSC</name>
            <description>Use Prescaler</description>
            <bitRange>[7:7]</bitRange>
            <access>read-write</access>
            <enumeratedValues>
              <enumeratedValue>
                <name>Disabled</name>
                <description>Prescaler is not used</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>Enabled</name>
                <description>Prescaler is used as divider</description>
                <value>1</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>

          <!--  CNTSRC: Timer / Counter Soruce Divider  -->
          <field>
            <name>CNTSRC</name>
            <description>Timer / Counter Source Divider</description>
            <bitRange>[11:8]</bitRange>
            <access>read-write</access>
            <enumeratedValues>
              <enumeratedValue>
                <name>CAP SRC</name>
                <description>Capture Source is used directly</description>
                <value>0</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 2</name>
                <description>Capture Source is divided by 2</description>
                <value>1</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 4</name>
                <description>Capture Source is divided by 4</description>
                <value>2</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 8</name>
                <description>Capture Source is divided by 8</description>
                <value>3</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 16</name>
                <description>Capture Source is divided by 16</description>
                <value>4</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 32</name>
                <description>Capture Source is divided by 32</description>
                <value>5</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 64</name>
                <description>Capture Source is divided by 64</description>
                <value>6</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 128</name>
                <description>Capture Source is divided by 128</description>
                <value>7</value>
              </enumeratedValue>
              <enumeratedValue>
                <name>CAP SRC / 256</name>
                <description>Capture Source is divided by 256</description>
                <value>8</value>
              </enumeratedValue>
            </enumeratedValues>
          </field>
        </fields>
      </register>
      
      <!--  SR: Status Register  -->
        <register>
          <name>SR</name>
          <description>Status Register</description>
          <addressOffset>0x10</addressOffset>
          <size>16</size>
          <access>read-write</access>
          <resetValue>0x00000000</resetValue>
          <resetMask>0xD701</resetMask>

          <fields>
            <!--  RUN: Shows if Timer is running  -->
            <field>
              <name>RUN</name>
              <description>Shows if Timer is running or not</description>
              <bitRange>[0:0]</bitRange>
              <access>read-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Not Running</name>
                  <description>Timer is not running</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Running</name>
                  <description>Timer is running</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  MATCH: Shows if a Match was hit  -->
            <field>
              <name>MATCH</name>
              <description>Shows if the MATCH was hit</description>
              <bitRange>[8:8]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>No Match</name>
                  <description>The MATCH condition was not hit</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Match Hit</name>
                  <description>The MATCH condition was hit</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  UN: Shows if an underflow occured  -->
            <field>
              <name>UN</name>
              <description>Shows if an underflow occured. This flag is sticky</description>
              <bitRange>[9:9]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>No Underflow</name>
                  <description>No underflow occured since last clear</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Underflow occured</name>
                  <description>A minimum of one underflow occured since last clear</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  OV: Shows if an overflow occured  -->
            <field>
              <name>OV</name>
              <description>Shows if an overflow occured. This flag is sticky</description>
              <bitRange>[10:10]</bitRange>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>No Overflow</name>
                  <description>No overflow occured since last clear</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Overflow occured</name>
                  <description>A minimum of one overflow occured since last clear</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>

            <!--  RST: Shows if Timer is in RESET state  -->
            <field>
              <name>RST</name>
              <description>Shows if Timer is in RESET state</description>
              <bitRange>[12:12]</bitRange>
              <access>read-only</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>Timer can operate</name>
                  <description>Timer is not in RESET state and can operate</description>
                  <value>0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>Reset State</name>
                  <description>Timer is in RESET state and can not operate</description>
                  <value>1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
        </register>
		
		<!--  dimIncrement test  -->
		<register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0-3</dimIndex>
          <name>DimInc_A%s</name>
          <description>DimIncrement test A</description>
          <addressOffset>0x20</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
		
		<register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>[0-3]</dimIndex>
          <name>DimInc_B%s</name>
          <description>DimIncrement test B</description>
          <addressOffset>0x30</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
		
		<register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0,1,2,3</dimIndex>
          <name>DimInc_C%s</name>
          <description>DimIncrement test C</description>
          <addressOffset>0x40</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
		
		<register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>AA,AB,AC,AD</dimIndex>
          <name>DimInc_D%s</name>
          <description>DimIncrement test D</description>
          <addressOffset>0x50</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
		
		<register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>[0-3]</dimIndex>
          <name>DimInc_E[%s]</name>
          <description>DimIncrement test E</description>
          <addressOffset>0x60</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
		
    </registers>    
  </peripheral>
  
  
  <peripheral>
    <name>IntPrefixTest</name>
    <version>1.0</version>
    <description>Interrupt prefix / Reserved test</description>
    <groupName>Interrupt</groupName>
    <baseAddress>0x60000000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <addressBlock>
      <offset>0</offset>
      <size>0x2000</size>
      <usage>registers</usage>
    </addressBlock>
	
	<interrupt>
	  <name>Interrupt_A</name>
	  <value>100</value>
	</interrupt>
    <interrupt>
	  <name>Interrupt_B</name>
	  <value>101</value>
	</interrupt>
    <interrupt>
	  <name>Interrupt_C</name>
	  <value>102</value>
	</interrupt>
    <interrupt>
	  <name>Interrupt_D</name>
	  <value>103</value>
	</interrupt>
	
	<interrupt>
	  <name>INT1</name>
	  <value>104</value>
	</interrupt>
    <interrupt>
	  <name>INT2</name>
	  <value>105</value>
	</interrupt>
    <interrupt>
	  <name>INT3</name>
	  <value>106</value>
	</interrupt>
    <interrupt>
	  <name>INT4</name>
	  <value>107</value>
	</interrupt>
	
	<interrupt>
	  <name>INT_Foo</name>
	  <value>108</value>
	</interrupt>
    <interrupt>
	  <name>INT_Bar</name>
	  <value>109</value>
	</interrupt>
    <interrupt>
	  <name>INT_Itchi</name>
	  <value>110</value>
	</interrupt>
    <interrupt>
	  <name>INT_Scratchi</name>
	  <value>111</value>
	</interrupt>
	<interrupt>
	  <name>INT_Dumb</name>
	  <value>112</value>
	</interrupt>
	
	<interrupt>
	  <name>SSP0</name>
	  <value>113</value>
	</interrupt>
    <interrupt>
	  <name>SSP1</name>
	  <value>114</value>
	</interrupt>
    <interrupt>
	  <name>SSP2</name>
	  <value>115</value>
	</interrupt>
    <interrupt>
	  <name>SSP3</name>
	  <value>116</value>
	</interrupt>
	<interrupt>
	  <name>SSP4</name>
	  <value>117</value>
	</interrupt>
	<interrupt>
	  <name>SSP5</name>
	  <value>118</value>
	</interrupt>
	<interrupt>
	  <name>SSP6</name>
	  <value>119</value>
	</interrupt>
	
	<!-- interrupt>
	  <name>reserved0</name>
	  <value>120</value>
	</interrupt -->
	<!-- interrupt>
	  <name>RESERVED1</name>
	  <value>121</value>
	</interrupt -->
	<!-- interrupt>
	  <name>ID_ReservedFoo3</name>
	  <value>122</value>
	</interrupt -->
	
	
    
	
    <registers>
      <register>
        <dim>4</dim>
        <dimIncrement>4</dimIncrement>
        <name>CR[%s]</name>
        <description>Control Register</description>
        <addressOffset>0x00</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0x1337F7F</resetMask>
      </register>		
    </registers>    
  </peripheral>

  
  <peripheral>
    <name>TimerInt_EnumValuesTest</name>
    <version>1.11</version>
    <description>Enumerated Values Test</description>
    <baseAddress>0x60004000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <addressBlock>
      <offset>0</offset>
      <size>0x1000</size>
      <usage>registers</usage>
    </addressBlock>

    <registers>

      <register>
        <name>INTENSET</name>
        <description>usage test: Interrupt enable set register.</description>
        <addressOffset>0x00</addressOffset>
        <fields>
          <field>
            <name>TimerIntSelect</name>
            <description>TimerIntSelect</description>
            <lsb>2</lsb> <msb>2</msb>
            <enumeratedValues>
              <name>TimerIntSelect</name>
              <usage>read</usage>   
			  
              <enumeratedValue>
                <name>disabled</name>
                <description>The clock source clk0 is turned off.</description>
                <value>0</value>
              </enumeratedValue>              
              </enumeratedValues>
              
              <enumeratedValues>              
                <name>TimerIntSelect</name>
                <usage>write</usage>
				
				<enumeratedValue>              
			      <name>disabled</name>
                  <description>The clock source clk0 is turned off.</description>
                  <value>0</value>
                </enumeratedValue>
              </enumeratedValues>
          </field>
        </fields>
      </register>
    </registers>
  </peripheral>
  
  
  
  <peripheral>
    <name>DuplicateReg</name>
    <version>1.11</version>
    <description>Duplicate Regs Test</description>
    <baseAddress>0x60005000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <addressBlock>
      <offset>0</offset>
      <size>0x1000</size>
      <usage>registers</usage>
    </addressBlock>

    <registers>

      <register>
        <name>DUPREG1</name>
        <description>Duplicate Register 1.</description>
        <addressOffset>0x00</addressOffset>
      </register>
      <register>
        <name>DUPREG1</name>
        <description>Duplicate Register 1.</description>
        <addressOffset>0x00</addressOffset>
      </register>
      <register>
        <name>DUPREG1</name>
        <description>Duplicate Register 1.</description>
        <addressOffset>0x04</addressOffset>
      </register>
      
      <register>
        <name>DUPREG2to1</name>
        <description>Duplicate Register 2 to 1.</description>
        <addressOffset>0x00</addressOffset>
      </register>
    </registers>
  </peripheral>
  
  
  <peripheral>
    <name>DIM_GT512Test</name>
    <description>DIM greater 512 Test, see Case 570966</description>
    <baseAddress>0x60006000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <addressBlock>
      <offset>0</offset>
      <size>0x1000</size>
      <usage>registers</usage>
    </addressBlock>

    <registers>    
      <register>
        <SVDConv>break</SVDConv>
        <dim>516</dim>
        <dimIncrement>4</dimIncrement>
        <name>g_ambient_buf[%s]</name>
        <description>The Reload Register stores the Value the COUNT Register gets reloaded on a when a 
        condition was met.</description>
        <addressOffset>0x60</addressOffset>
        <size>32</size>
        <access>read-write</access>
        <resetValue>0x00000000</resetValue>
        <resetMask>0xFFFFFFFF</resetMask>
      </register>
    </registers>
  </peripheral>
  
  
  <peripheral>
    <name>Microsemi_Test</name>
    <description>Microsemi Test Union</description>
    <baseAddress>0x60007000</baseAddress>
    <size>32</size>
    <access>read-write</access>

    <addressBlock>
      <offset>0</offset>
      <size>0x100</size>
      <usage>registers</usage>
    </addressBlock>

	<registers>
		<register>
		<name>USB_RX_COUNT_REG</name>
		<description>No description provided for this register</description>
		<addressOffset>0x00</addressOffset>
		<size>16</size>
		<fields>
			<field>
			<name>ENDPOINT_RX_COUNT</name>
			<description>No description provided for this field</description>
			<bitOffset>0</bitOffset>
			<bitWidth>14</bitWidth>
			<access>read-only</access>
			</field>
		</fields>
		</register>
		<register>
		<name>USB_COUNT0_REG</name>
		<description>No description provided for this register</description>
		<alternateRegister>USB_RX_COUNT_REG</alternateRegister>
		<addressOffset>0x00</addressOffset>
		<size>8</size>
		<fields>
			<field>
			<name>ENDPOINT_RX_COUNT</name>
			<description>No description provided for this field</description>
			<bitOffset>0</bitOffset>
			<bitWidth>7</bitWidth>
			<access>read-only</access>
			</field>
		</fields>
		</register>
	</registers>
  </peripheral>
  
          <peripheral>
            <name>ReneassTest_RSPI0</name>
            <description>Reneass Test Serial Peripheral Interface 0</description>
            <baseAddress>0x40072000</baseAddress>
            <addressBlock>
                <offset>0x00</offset>
                <size>8</size>
                <usage>registers</usage>
            </addressBlock>
            <addressBlock>
                <offset>0x04</offset>
                <size>2</size>
                <usage>registers</usage>
            </addressBlock>
            <addressBlock>
                <offset>0x08</offset>
                <size>24</size>
                <usage>registers</usage>
            </addressBlock>
            <registers>
                <register>
                    <name>SPCR</name>
                    <description>RSPI Control Register</description>
                    <addressOffset>0x00</addressOffset>
                    <size>8</size>
                    <access>read-write</access>
                    <resetValue>0x00</resetValue>
                    <resetMask>0xFF</resetMask>
                    <fields>
                        <field>
                            <name>SPRIE</name>
                            <description>RSPI Receive Buffer Full Interrupt Enable</description>
                            <lsb>7</lsb>
                            <msb>7</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Disables the generation of RSPI receive buffer full interrupt requests</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Enables the generation of RSPI receive buffer full interrupt requests</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPE</name>
                            <description>RSPI Function Enable</description>
                            <lsb>6</lsb>
                            <msb>6</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Disables the RSPI function</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Enables the RSPI function</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPTIE</name>
                            <description>Transmit Buffer Empty Interrupt Enable</description>
                            <lsb>5</lsb>
                            <msb>5</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Disables the generation of transmit buffer empty interrupt requests</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Enables the generation of transmit buffer empty interrupt requests</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPEIE</name>
                            <description>RSPI Error Interrupt Enable</description>
                            <lsb>4</lsb>
                            <msb>4</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Disables the generation of RSPI error interrupt requests</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Enables the generation of RSPI error interrupt requests</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>MSTR</name>
                            <description>RSPI Master/Slave Mode Select</description>
                            <lsb>3</lsb>
                            <msb>3</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Slave mode</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Master mode</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>MODFEN</name>
                            <description>Mode Fault Error Detection Enable</description>
                            <lsb>2</lsb>
                            <msb>2</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Disables the detection of mode fault error</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Enables the detection of mode fault error</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>TXMD</name>
                            <description>Communications Operating Mode Select</description>
                            <lsb>1</lsb>
                            <msb>1</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Full-duplex synchronous serial communications</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Serial communications consisting of only transmit operations</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPMS</name>
                            <description>RSPI Mode Select</description>
                            <lsb>0</lsb>
                            <msb>0</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>SPI operation (four-wire method)</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Clock synchronous operation (three-wire method)</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                    </fields>
                </register>
                <register>
                    <name>SSLP</name>
                    <description>RSPI Slave Select Polarity Register</description>
                    <addressOffset>0x01</addressOffset>
                    <size>8</size>
                    <access>read-write</access>
                    <resetValue>0x00</resetValue>
                    <resetMask>0xFF</resetMask>
                    <fields>
                        <field>
                            <name>SSL3P</name>
                            <description>SSL3 Signal Polarity Setting</description>
                            <lsb>3</lsb>
                            <msb>3</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>SSL3 signal is active low</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>SSL3 signal is active high</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SSL2P</name>
                            <description>SSL2 Signal Polarity Setting</description>
                            <lsb>2</lsb>
                            <msb>2</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>SSL2 signal is active low</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>SSL2 signal is active high</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SSL1P</name>
                            <description>SSL1 Signal Polarity Setting</description>
                            <lsb>1</lsb>
                            <msb>1</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>SSL1 signal is active low</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>SSL1 signal is active high</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SSL0P</name>
                            <description>SSL0 Signal Polarity Setting</description>
                            <lsb>0</lsb>
                            <msb>0</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>SSL0 signal is active low</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>SSL0 signal is active high</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                    </fields>
                </register>
                <register>
                    <name>SPPCR</name>
                    <description>RSPI Pin Control Register</description>
                    <addressOffset>0x02</addressOffset>
                    <size>8</size>
                    <access>read-write</access>
                    <resetValue>0x00</resetValue>
                    <resetMask>0xFF</resetMask>
                    <fields>
                        <field>
                            <name>MOIFE</name>
                            <description>MOSI Idle Value Fixing Enable</description>
                            <lsb>5</lsb>
                            <msb>5</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>MOSI output value equals final data from previous transfer</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>MOSI output value equals the value set in the MOIFV bit</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>MOIFV</name>
                            <description>MOSI Idle Fixed Value</description>
                            <lsb>4</lsb>
                            <msb>4</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>The level output on the MOSIA pin during MOSI idling corresponds to low.</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>The level output on the MOSIA pin during MOSI idling corresponds to high.</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPLP2</name>
                            <description>RSPI Loopback 2</description>
                            <lsb>1</lsb>
                            <msb>1</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Normal mode</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Loopback mode (data is not inverted for transmission)</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPLP</name>
                            <description>RSPI Loopback</description>
                            <lsb>0</lsb>
                            <msb>0</msb>
                            <access>read-write</access>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Normal mode</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Loopback mode (data is inverted for transmission)</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                    </fields>
                </register>
                <register>
                    <name>SPSR</name>
                    <description>RSPI Status Register</description>
                    <addressOffset>0x03</addressOffset>
                    <size>8</size>
                    <access>read-write</access>
                    <resetValue>0x20</resetValue>
                    <resetMask>0xFF</resetMask>
                    <fields>
                        <field>
                            <name>SPRF</name>
                            <description>RSPI Receive Buffer Full Flag</description>
                            <lsb>7</lsb>
                            <msb>7</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>No valid data in SPDR</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>Valid data found in SPDR</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>SPTEF</name>
                            <description>RSPI Transmit Buffer Empty Flag</description>
                            <lsb>5</lsb>
                            <msb>5</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Data found in the transmit buffer</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>No data in the transmit buffer</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>UDRF</name>
                            <description>Underrun Error Flag
(When MODF is 0,  This bit is invalid.)</description>
                            <lsb>4</lsb>
                            <msb>4</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>A mode fault error occurs (MODF=1)</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>An underrun error occurs (MODF=1)</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>PERF</name>
                            <description>Parity Error Flag</description>
                            <lsb>3</lsb>
                            <msb>3</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>No parity error occurs</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>A parity error occurs</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>MODF</name>
                            <description>Mode Fault Error Flag</description>
                            <lsb>2</lsb>
                            <msb>2</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>Neither mode fault error nor underrun error occurs</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>A mode fault error or an underrun error occurs.</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>IDLNF</name>
                            <description>RSPI Idle Flag</description>
                            <lsb>1</lsb>
                            <msb>1</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>RSPI is in the idle state</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>RSPI is in the transfer state</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                        <field>
                            <name>OVRF</name>
                            <description>Overrun Error Flag</description>
                            <lsb>0</lsb>
                            <msb>0</msb>
                            <access>read-write</access>
                            <modifiedWriteValues>zeroToClear</modifiedWriteValues>
                            <readAction>modify</readAction>
                            <enumeratedValues>
                                <enumeratedValue>
                                    <name>0</name>
                                    <description>No overrun error occurs</description>
                                    <value>#0</value>
                                </enumeratedValue>
                                <enumeratedValue>
                                    <name>1</name>
                                    <description>An overrun error occurs</description>
                                    <value>#1</value>
                                </enumeratedValue>
                            </enumeratedValues>
                        </field>
                    </fields>
                </register>
                <register>
                    <name>SPDR</name>
                    <description>RSPI Data Register</description>
                    <addressOffset>0x04</addressOffset>
                    <size>32</size>
                    <access>read-write</access>
                    <resetValue>0x00000000</resetValue>
                    <resetMask>0xFFFFFFFF</resetMask>
                </register>
                <register>
                    <name>SPDR_HA</name>
                    <description>RSPI Data Register ( halfword access )</description>
                    <alternateRegister>SPDR</alternateRegister>
                    <addressOffset>0x04</addressOffset>
                    <size>16</size>
                    <access>read-write</access>
                    <resetValue>0x0000</resetValue>
                    <resetMask>0xFFFF</resetMask>
                </register>
                <register>
                    <name>SPSCR</name>
                    <description>RSPI Sequence Control Register</description>
                    <addressOffset>0x8</addressOffset>
                    <size>8</size>
                    <access>read-write</access>
                    <resetValue>0x00</resetValue>
                    <resetMask>0xFF</resetMask>
                    <fields>
                        <field>
                            <name>SPSLN</name>
                            <description>RSPI Sequence Length Specification

The order in which the SPCMD0 to SPCMD07 registers are to be referenced is changed in accordance with the sequence length that is set in these bits. The relationship among the setting of these bits, sequence length, and SPCMD0 to SPCMD7 registers referenced by the RSPI is shown above. However, the RSPI in slave mode always references SPCMD0.</description>
                            <lsb>0</lsb>
                            <msb>2</msb>
                            <access>read-write</access>
                        </field>
                    </fields>
                </register>
            </registers>
        </peripheral>
        <peripheral derivedFrom="ReneassTest_RSPI0">
            <name>RSPI1</name>
            <description>Serial Peripheral Interface 1</description>
            <baseAddress>0x40072100</baseAddress>
        </peripheral>
	
	<peripheral>
      <name>CRC</name>
      <description>Cyclic Redundancy Check</description>
      <prependToName>CRC_</prependToName>
      <baseAddress>0x40032000</baseAddress>
      <addressBlock>
        <offset>0</offset>
        <size>0xC</size>
        <usage>registers</usage>
      </addressBlock>
      <registers>
        <register>
          <name>CRCL</name>
          <description>CRC_CRCL register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0</addressOffset>
          <size>16</size>
          <resetValue>0xFFFF</resetValue>
          <resetMask>0xFFFF</resetMask>
          <fields>
            <field>
              <name>CRCL</name>
              <description>CRCL stores the lower 16 bits of the 16/32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>16</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CRCLL</name>
          <description>CRC_CRCLL register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>CRCLL</name>
              <description>CRCLL stores the first 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CRC</name>
          <description>CRC Data Register</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0</addressOffset>
          <size>32</size>
          <resetValue>0xFFFFFFFF</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>LL</name>
              <description>CRC Low Lower Byte</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
            <field>
              <name>LU</name>
              <description>CRC Low Upper Byte</description>
              <bitOffset>8</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
            <field>
              <name>HL</name>
              <description>CRC High Lower Byte</description>
              <bitOffset>16</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
            <field>
              <name>HU</name>
              <description>CRC High Upper Byte</description>
              <bitOffset>24</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CRCLU</name>
          <description>CRC_CRCLU register.</description>
          <addressOffset>0x1</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>CRCLU</name>
              <description>CRCLL stores the second 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CRCHL</name>
          <description>CRC_CRCHL register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x2</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>CRCHL</name>
              <description>CRCHL stores the third 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CRCH</name>
          <description>CRC_CRCH register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x2</addressOffset>
          <size>16</size>
          <resetValue>0xFFFF</resetValue>
          <resetMask>0xFFFF</resetMask>
          <fields>
            <field>
              <name>CRCH</name>
              <description>CRCL stores the high 16 bits of the 16/32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>16</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CRCHU</name>
          <description>CRC_CRCHU register.</description>
          <addressOffset>0x3</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>CRCHU</name>
              <description>CRCHU stores the fourth 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLYLL</name>
          <description>CRC_GPOLYLL register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x4</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>GPOLYLL</name>
              <description>POLYLL stores the first 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLYL</name>
          <description>CRC_GPOLYL register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x4</addressOffset>
          <size>16</size>
          <resetValue>0xFFFF</resetValue>
          <resetMask>0xFFFF</resetMask>
          <fields>
            <field>
              <name>GPOLYL</name>
              <description>POLYL stores the lower 16 bits of the 16/32 bit CRC polynomial value</description>
              <bitOffset>0</bitOffset>
              <bitWidth>16</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLY</name>
          <description>CRC Polynomial Register</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x4</addressOffset>
          <size>32</size>
          <resetValue>0x1021</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>LOW</name>
              <description>Low polynominal half-word</description>
              <bitOffset>0</bitOffset>
              <bitWidth>16</bitWidth>
              <access>read-write</access>
            </field>
            <field>
              <name>HIGH</name>
              <description>High polynominal half-word</description>
              <bitOffset>16</bitOffset>
              <bitWidth>16</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLYLU</name>
          <description>CRC_GPOLYLU register.</description>
          <addressOffset>0x5</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>GPOLYLU</name>
              <description>POLYLL stores the second 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLYHL</name>
          <description>CRC_GPOLYHL register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x6</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>GPOLYHL</name>
              <description>POLYHL stores the third 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLYH</name>
          <description>CRC_GPOLYH register.</description>
          <alternateGroup>CRC</alternateGroup>
          <addressOffset>0x6</addressOffset>
          <size>16</size>
          <resetValue>0xFFFF</resetValue>
          <resetMask>0xFFFF</resetMask>
          <fields>
            <field>
              <name>GPOLYH</name>
              <description>POLYH stores the high 16 bits of the 16/32 bit CRC polynomial value</description>
              <bitOffset>0</bitOffset>
              <bitWidth>16</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>GPOLYHU</name>
          <description>CRC_GPOLYHU register.</description>
          <addressOffset>0x7</addressOffset>
          <size>8</size>
          <resetValue>0xFF</resetValue>
          <resetMask>0xFF</resetMask>
          <fields>
            <field>
              <name>GPOLYHU</name>
              <description>POLYHU stores the fourth 8 bits of the 32 bit CRC</description>
              <bitOffset>0</bitOffset>
              <bitWidth>8</bitWidth>
              <access>read-write</access>
            </field>
          </fields>
        </register>
        <register>
          <name>CTRL</name>
          <description>CRC Control Register</description>
          <addressOffset>0x8</addressOffset>
          <size>32</size>
          <resetValue>0</resetValue>
          <resetMask>0xFFFFFFFF</resetMask>
          <fields>
            <field>
              <name>RESERVED</name>
              <description>no description available</description>
              <bitOffset>0</bitOffset>
              <bitWidth>24</bitWidth>
              <access>read-only</access>
            </field>
            <field>
              <name>TCRC</name>
              <description>no description available</description>
              <bitOffset>24</bitOffset>
              <bitWidth>1</bitWidth>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>0</name>
                  <description>16-bit CRC protocol.</description>
                  <value>#0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>1</name>
                  <description>32-bit CRC protocol.</description>
                  <value>#1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            <field>
              <name>WAS</name>
              <description>Write CRC data register as seed</description>
              <bitOffset>25</bitOffset>
              <bitWidth>1</bitWidth>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>0</name>
                  <description>Writes to the CRC data register are data values.</description>
                  <value>#0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>1</name>
                  <description>Writes to the CRC data register are seed values.</description>
                  <value>#1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            <field>
              <name>FXOR</name>
              <description>Complement Read of CRC data register</description>
              <bitOffset>26</bitOffset>
              <bitWidth>1</bitWidth>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>0</name>
                  <description>No XOR on reading.</description>
                  <value>#0</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>1</name>
                  <description>Invert or complement the read value of the CRC data register.</description>
                  <value>#1</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            <field>
              <name>RESERVED</name>
              <description>no description available</description>
              <bitOffset>27</bitOffset>
              <bitWidth>1</bitWidth>
              <access>read-only</access>
            </field>
            <field>
              <name>TOTR</name>
              <description>Type of Transpose for Read</description>
              <bitOffset>28</bitOffset>
              <bitWidth>2</bitWidth>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>00</name>
                  <description>No transposition.</description>
                  <value>#00</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>01</name>
                  <description>Bits in bytes are transposed; bytes are not transposed.</description>
                  <value>#01</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>10</name>
                  <description>Both bits in bytes and bytes are transposed.</description>
                  <value>#10</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>11</name>
                  <description>Only bytes are transposed; no bits in a byte are transposed. </description>
                  <value>#11</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
            <field>
              <name>TOT</name>
              <description>Type of Transpose for Writes</description>
              <bitOffset>30</bitOffset>
              <bitWidth>2</bitWidth>
              <access>read-write</access>
              <enumeratedValues>
                <enumeratedValue>
                  <name>00</name>
                  <description>No transposition.</description>
                  <value>#00</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>01</name>
                  <description>Bits in bytes are transposed; bytes are not transposed.</description>
                  <value>#01</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>10</name>
                  <description>Both bits in bytes and bytes are transposed.</description>
                  <value>#10</value>
                </enumeratedValue>
                <enumeratedValue>
                  <name>11</name>
                  <description>Only bytes are transposed; no bits in a byte are transposed.</description>
                  <value>#11</value>
                </enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
        </register>
      </registers>
    </peripheral>

	<peripheral>
	<name>PERI</name>
	<description>test</description>
	<groupName>G0</groupName>
	<baseAddress>0x40050000</baseAddress>
	<addressBlock>
	  <offset>0</offset>
		<size>0x80</size>
		<usage>register</usage>
	</addressBlock>
	<registers>
	<!--  ================= REG0W Information =================  -->
	<register>
	<name>REG0W</name>
	<description>reg0</description>
	<addressOffset>0x00000010</addressOffset>
	<size>32</size>

	<fields>
		<!-- ============= [31:0] ============= -->
		<field>
		<name>BIT0W</name> <bitRange>[31:0]</bitRange> <access>read-write</access>
		<description>test</description>
		</field>
	</fields> 
	</register>
	<!--  ================= REG0W End =================  -->

	<!--  ================= REG0 Information =================  -->
	<register>
	<name>REG0</name>
	<alternateRegister>REG0W</alternateRegister>
	<description>reg0</description>
	<addressOffset>0x00000010</addressOffset>
	<size>8</size>

	<fields>
		<!-- ============= [7:0] ============= -->
		<field>
		<name>BIT0</name> <bitRange>[7:0]</bitRange> <access>read-write</access>
		<description>test</description>
		</field>
	</fields> 
	</register>
	<!--  ================= REG0 End =================  -->

	<!--  ================= REG1W Information =================  -->
	<register>
	<name>REG1W</name>
	<description>reg1</description>
	<addressOffset>0x00000040</addressOffset>
	<size>32</size>

	<fields>
		<!-- ============= [31:0] ============= -->
		<field>
		<name>BIT1W</name> <bitRange>[31:0]</bitRange> <access>read-write</access>
		<description>test</description>
		</field>
	</fields> 
	</register>
	<!--  ================= REG1W End =================  -->

	<!--  ================= REG1H Information =================  -->
	<register>
	<name>REG1H</name>
	<alternateRegister>REG1W</alternateRegister>
	<description>reg1</description>
	<addressOffset>0x00000040</addressOffset>
	<size>16</size>

	<fields>
		<!-- ============= [15:0] ============= -->
		<field>
		<name>BIT1H</name> <bitRange>[15:0]</bitRange> <access>read-write</access>
		<description>test</description>
		</field>
	</fields> 
	</register>
	<!--  ================= REG1H End =================  -->

	<!--  ================= REG1 Information =================  -->
	<register>
	<name>REG1</name>
	<alternateRegister>REG1W</alternateRegister>
	<description>reg1</description>
	<addressOffset>0x00000040</addressOffset>
	<size>8</size>

	<fields>
		<!-- ============= [7:0] ============= -->
		<field>
		<name>BIT1</name> <bitRange>[7:0]</bitRange> <access>read-write</access>
		<description>test</description>
		</field>
	</fields> 
	</register>
	<!--  ================= REG1 End =================  -->

	</registers>
</peripheral>
  </peripherals>
</device>

