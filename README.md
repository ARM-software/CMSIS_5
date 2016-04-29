# CMSIS Version 5

CMSIS Version 5.0.0 is scheduled for release in June 2016. Once completed it will be release in this GitHub project.


Use *Issues* to provide feedback and report problems for CMSIS Version 5. Note that this repository gives our users and partners contiguous access to the CMSIS development. It allows you to review the work and provide feedback or create pull requests for contributions.


## Planned Enhancements
 - Add support for ARMv8-M Architecture (Mainline and Baseline)

 - Improvements for Cortex-A / M hybrid devices (focus on Cortex-M interaction)

 - CMSIS-RTOS API and RTX reference implementation with several enhancements:
     - Dynamic object creation, Flag events, C and C++ API, additional thread and timer functions
     - Secure and Non-Secure support, multi-processor support

 - CMSIS-Pack 
     - Additions for generic example, project templates, multiple download portals
     - Adoption of IAR Flash Loader technology

For further details see also the [Slides of the Embedded World CMSIS Partner Meeting](https://github.com/ARM-software/CMSIS_5/blob/master/CMSIS_EW2016.pdf).

## Directory Structure

*All CMSIS components will be available by end of March 2016*

| Directory       | Content                                        |                
| --------------- | ---------------------------------------------- |
| CMSIS/Core      | CMSIS-Core related files (for release)         |
| CMSIS/DAP       | CMSIS-DAP related files and examples           |
| CMSIS/Driver    | CMSIS-Driver API headers and template files    |
| CMSIS/DSP       | CMSIS-DSP related files                        |
| CMSIS/RTOS      | RTOS v1 related files (for Cortex-M)           |
| CMSIS/RTOS2     | RTOS v2 related files (for Cortex-M & ARMv8-M) |
| CMSIS/DoxyGen   | Source of the documentation                    |
| CMSIS/Utilities | Utility programs                               |

## Generate Documentation

The following tools are required to generate the documentation:
 - **doxygen.exe**  Version: 1.8.2 (Documentation Generator)
 - **mscgen.exe**   Version: 0.20  (Message Sequence Chart Converter)

Using these tools, documentation can be generated under Windows
with the batch file **genDoc.bat** in directory CMSIS/DoxyGen/.

The file CMSIS/DoxyGen/How2Doc.txt describes the rules for creating API
documentation.

## Generate CMSIS Software Pack

*coming soon*

## License

ARM CMSIS is licensed under Apache-2.0.

## Contributions and Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have authored all of the code.
