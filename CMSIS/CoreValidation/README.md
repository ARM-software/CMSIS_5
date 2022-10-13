# CMSIS-Core Validation

This repository contains a test suite that validates CMSIS-Core implementations. It uses [**Fixed Virtual Platforms**](https://developer.arm.com/Tools%20and%20Software/Fixed%20Virtual%20Platforms) to run tests to verify correct operation of the CMSIS core functionality on various Arm Cortex-M based processors.

## Repository structure

| Directory         | Contents                                                                          |
|-------------------|-----------------------------------------------------------------------------------|
| .github/workflows | Workflow YML files for running the test suite and for creating the documentation. |
| Include           | Include files for test cases etc.                                                 |
| Layer             | Layers for creating the projects.                                                 |
| Project           | An example project that shows unit testing.                                       |
| Source            | Test case source code.                                                            |

## Test matrix

Currently, the following tests are executed in the [CMSIS_xx](./.github/workflows/cmsis_xx.yml) workflow:

| Compiler |  Device      | Optimization Level    |
|----------|--------------|--------------------   |
| AC6      |  ARMCM0      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM0      | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM0P     | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM0P     | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM3      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM3      | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM4      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM4      | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM4_FP   | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM4_FP   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM7      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM7      | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM7_SP   | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM7_SP   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM7_DP   | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM7_DP   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM23     | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM23     | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM23S    | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM23S    | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM23NS   | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM23NS   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM33     | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM33     | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM33S    | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM33S    | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM33NS   | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM33NS   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM35P    | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM35P    | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM35PS   | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM35PS   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM35PNS  | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM35PNS  | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM55S    | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM55NS   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCM85S    | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCM85NS   | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCA5      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCA5      | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCA7      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCA7      | O1, O2, Ofast, Os, Oz |
| AC6      |  ARMCA9      | O1, O2, Ofast, Os, Oz |
| GCC      |  ARMCA9      | O1, O2, Ofast, Os, Oz |

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
