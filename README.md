# CMSIS_5
CMSIS Version 5 Development Repository for public review and feedback.


## Directory Structure

| Directory       | Content                                        |                
| --------------- | ---------------------------------------------- |
| CMSIS/Core      | CMSIS-Core related files (for release)         |
| CMSIS/Driver    | CMSIS-Driver API headers and template files    |
| CMSIS/RTOS      | RTOS related files (template + include)        |
| CMSIS/DoxyGen   | Source of the documentation                    |
| CMSIS/Utilities | Utility programs                               |

## Generate Documentation

The following tools are required to generate the documentation:
 - *doxygen.exe*  Version: 1.8.2 (Documentation Generator)
 - *mscgen.exe*   Version: 0.20  (Message Sequence Chart Converter)

Using these tools, documentation can be generated under Windows
with the batch file *genDoc.bat* in directory CMSIS/DoxyGen/.

The file CMSIS/DoxyGen/How2Doc.txt describes the rules for creating API
documentation.
