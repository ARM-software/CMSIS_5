# CMSIS Version 5

The branch *master* of this GitHub repository contains the CMSIS Version 5.2.0.  The [documentation](http://arm-software.github.io/CMSIS_5/General/html/index.html) is available under http://arm-software.github.io/CMSIS_5/General/html/index.html

Use [Issues](https://github.com/ARM-software/CMSIS_5#issues-and-labels) to provide feedback and report problems for CMSIS Version 5. 

**Note:** The branch *develop* of this GitHub repository reflects our current state of development and is constantly updated. It gives our users and partners contiguous access to the CMSIS development. It allows you to review the work and provide feedback or create pull requests for contributions.

A [pre-built documentation](http://www.keil.com/pack/doc/CMSIS_Dev/index.html) is updated from time to time, but may be also generated using the instructions under [Generate CMSIS Pack for Release](https://github.com/ARM-software/CMSIS_5#generate-cmsis-pack-for-release).

## What's Hot
 - CMSIS-RTOS2: RTX 5 is now available for IAR, GCC, ARM Compiler 5, ARM Compiler 6
 - CMSIS-RTOS2: FreeRTOS adoption (release) is available https://github.com/ARM-software/CMSIS-FreeRTOS
 - CMSIS-Core: compiler agnostic features extended to simplify transition on LLVM based front-end
 - CMSIS-Core-A: preview of the CMSIS-Core for Cortex-A
 - Coming soon: CMSIS-RTOS2 for Cortex-A

## Implemented Enhancements
 - Support for ARMv8-M Architecture (Mainline and Baseline) as well as devices Cortex-M23 and Cortex-M33

 - CMSIS-RTOS Version 2 API and RTX reference implementation with several enhancements:
     - Dynamic object creation, Flag events, C API, additional thread and timer functions

 - CMSIS-RTOS API Secure and Non-Secure support, multi-processor support

## Further Planned Enhancements
 - CMSIS-Core-A: Cortex-A processor support
 - CMSIS-RTOS: RTX5 for Cortex-A
 - CMSIS-DAP: extended trace support
 - CMSIS-Zone: management of complex system
     - Improvements for Cortex-A / M hybrid devices (focus on Cortex-M interaction)
 - CMSIS-Pack 
     - Additions for generic example and project templates
     - Adoption of IAR Flash Loader technology

For further details see also the [Slides of the Embedded World CMSIS Partner Meeting](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS_EW2017.pdf).

## Directory Structure

| Directory       | Content                                        |                
| --------------- | ---------------------------------------------- |
| CMSIS/Core      | CMSIS-Core related files (for release)         |
| CMSIS/DAP       | CMSIS-DAP related files and examples           |
| CMSIS/Driver    | CMSIS-Driver API headers and template files    |
| CMSIS/DSP       | CMSIS-DSP related files                        |
| CMSIS/RTOS      | RTOS v1 related files (for Cortex-M)           |
| CMSIS/RTOS2     | RTOS v2 related files (for Cortex-M & ARMv8-M) |
| CMSIS/Pack      | CMSIS-Pack examples and tutorials              |
| CMSIS/DoxyGen   | Source of the documentation                    |
| CMSIS/Utilities | Utility programs                               |

## Generate CMSIS Pack for Release

This GitHub development repository contains already pre-built libraries of various software components (DSP, RTOS, RTOS2).
These libraries are validated for release.

To build a complete CMSIS pack for installation the following additional tools are required:
 - **doxygen.exe**    Version: 1.8.6 (Documentation Generator)
 - **mscgen.exe**     Version: 0.20  (Message Sequence Chart Converter)
 - **7z.exe (7-Zip)** Version: 16.02 (File Archiver)
  
Using these tools, you can generate on a Windows PC:
 - **CMSIS Software Pack** using the batch file **gen_pack.bat** (located in ./CMSIS/Utilities). This batch file also generates the documentation.
  
 - **CMSIS Documentation** using the batch file **genDoc.bat** (located in ./CMSIS/Doxygen). 

The file ./CMSIS/DoxyGen/How2Doc.txt describes the rules for creating API documentation.

## License

ARM CMSIS is licensed under Apache-2.0.

## Contributions and Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have authored all of the code.

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

- **Important Information** - We provide essential informations regarding planned or resolved major enhancements.

