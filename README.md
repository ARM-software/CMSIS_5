# CMSIS Version 6 - Preview (Work in Progres)

[![Version](https://img.shields.io/github/v/release/arm-software/CMSIS_6)](https://github.com/ARM-software/CMSIS_6/releases/latest) [![License](https://img.shields.io/github/license/arm-software/CMSIS_6)](https://arm-software.github.io/CMSIS_6/General/html/LICENSE.txt)

The branch *main* of this GitHub repository contains ![Version](https://img.shields.io/github/v/release/arm-software/CMSIS_6?display_name=release&label=%20&sort=semver).

The [documentation](http://arm-software.github.io/CMSIS_6/General/html/index.html) is available under http://arm-software.github.io/CMSIS_6/General/html/index.html

Use [Issues](https://github.com/ARM-software/CMSIS_6#issues-and-labels) to provide feedback and report problems for CMSIS Version 6.

**Note:** The branch *main* of this GitHub repository reflects our current state of development and is constantly updated. It gives our users and partners contiguous access to the CMSIS development. It allows you to review the work and provide feedback or create pull requests for contributions. A [pre-built documentation](https://arm-software.github.io/CMSIS_5/develop/General/html/index.html) is updated from time to time, but may be also generated using the instructions under [Generate CMSIS Pack for Release](https://github.com/ARM-software/CMSIS_5#generate-cmsis-pack-for-release).

For a list of all CMSIS components refer to [**Introduction - CMSIS Components**](./CMSIS/DoxyGen/General/src/introduction.md#structure)


## Other related GitHub repositories

| Repository                  | Description                                               |
|:--------------------------- |:--------------------------------------------------------- |
| [cmsis-pack-eclipse](https://github.com/ARM-software/cmsis-pack-eclipse)    |  CMSIS-Pack Management for Eclipse reference implementation Pack support  |
| [CMSIS-FreeRTOS](https://github.com/arm-software/CMSIS-FreeRTOS)            | CMSIS-RTOS adoption of FreeRTOS                                                      |
| [CMSIS-Driver](https://github.com/arm-software/CMSIS-Driver)                | Generic MCU driver implementations and templates for Ethernet MAC/PHY and Flash.  |
| [CMSIS-Driver_Validation](https://github.com/ARM-software/CMSIS-Driver_Validation) | CMSIS-Driver Validation can be used to verify CMSIS-Driver in a user system |
| [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP)                      | DSP library collection with hundreds of functions for various data types: fixed-point (fractional q7, q15, q31) and single precision floating-point (32-bit). Implementations optimized for the SIMD instruction set are available for Armv7E-M and later devices. |
| [CMSIS-Zone](https://github.com/ARM-software/CMSIS-Zone)                    | CMSIS-Zone Utility along with example projects and FreeMarker templates         |
| [NXP_LPC](https://github.com/ARM-software/NXP_LPC)                          | CMSIS Driver Implementations for the NXP LPC Microcontroller Series       |
| [mdk-packs](https://github.com/mdk-packs)                                   | IoT cloud connectors as trail implementations for MDK (help us to make it generic)|
| [trustedfirmware.org](https://www.trustedfirmware.org/)                     | Arm Trusted Firmware provides a reference implementation of secure world software for Armv8-A and Armv8-M.|

## Directory Structure

Directory            | Content
:------------------- |:---------------------------------------------------------
CMSIS/Core           | CMSIS-Core(M) related files (for release)
CMSIS/Core_A         | CMSIS-Core(A) related files (for release)
CMSIS/CoreValidation | Validation for Core(M) and Core(A) (NOT part of release)  
CMSIS/Driver         | CMSIS-Driver API headers and template files
CMSIS/RTOS2          | RTOS v2 related files (for Cortex-M & Armv8-M)

## Generate CMSIS Pack for Release

This GitHub development repository lacks pre-built libraries of various software components (RTOS, RTOS2).
In order to generate a full pack one needs to have the build environment available to build these libraries.
This causes some sort of inconvenience. Hence the pre-built libraries may be moved out into separate pack(s)
in the future.

To build a complete CMSIS pack for installation the following additional tools are required:

- **doxygen.exe**    Version: 1.9.2 (Documentation Generator)
- **mscgen.exe**     Version: 0.20  (Message Sequence Chart Converter)
- **7z.exe (7-Zip)** Version: 16.02 (File Archiver)

Using these tools, you can generate on a Windows PC:

- **CMSIS Documentation** using the shell script **gen_doc.sh** (located in ./CMSIS/DoxyGen).
- **CMSIS Software Pack** using the shell script **gen_pack.sh**.

## License

Arm CMSIS is licensed under Apache 2.0.

## Contributions and Pull Requests

Contributions are accepted under Apache 2.0. Only submit contributions where you have authored all of the code.

### Issues and Labels

Please feel free to raise an [issue on GitHub](https://github.com/ARM-software/CMSIS_5/issues)
to report misbehavior (i.e. bugs) or start discussions about enhancements. This
is your best way to interact directly with the maintenance team and the community.
We encourage you to append implementation suggestions as this helps to decrease the
workload of the very limited maintenance team.

We will be monitoring and responding to issues as best we can.
Please attempt to avoid filing duplicates of open or closed items when possible.
In the spirit of openness we will be tagging issues with the following:

- **bug** – We consider this issue to be a bug that will be investigated.

- **wontfix** - We appreciate this issue but decided not to change the current behavior.

- **enhancement** – Denotes something that will be implemented soon.

- **future** - Denotes something not yet schedule for implementation.

- **out-of-scope** - We consider this issue loosely related to CMSIS. It might by implemented outside of CMSIS. Let us know about your work.

- **question** – We have further questions to this issue. Please review and provide feedback.

- **documentation** - This issue is a documentation flaw that will be improved in future.

- **review** - This issue is under review. Please be patient.

- **DONE** - We consider this issue as resolved - please review and close it. In case of no further activity this issues will be closed after a week.

- **duplicate** - This issue is already addressed elsewhere, see comment with provided references.

- **Important Information** - We provide essential information regarding planned or resolved major enhancements.

