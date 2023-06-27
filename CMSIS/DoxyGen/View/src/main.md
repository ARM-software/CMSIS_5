# CMSIS-View {#mainpage}

**CMSIS-View** equips software developers with methodologies, software components, and utilities that help to analyze operation of embedded software programs on devices with Arm Cortex-M processors.

Key elements of CMSIS-View are:

- **Event Recorder** is an embedded software component that implements functions for event annotations in the code.
- **Exception Fault Analysis** provides functions to store, record, and analyze exception fault information.
- **eventlist** is a command line tool for processing Event Recorder log files.

## Access to CMSIS-View

CMSIS-View is maintained in a GitHub repository and is also released as a standalone package in CMSIS Software Pack format.

- [**CMSIS-View GitHub Repo**](https://github.com/Arm-Software/CMSIS-View) provides the full source code as well as releases in CMSIS-Pack format.
- [**CMSIS-View Documentation**](https://arm-software.github.io/CMSIS-View/latest/) explains how to use the library and describes the implemented functions in details.

## Key Features and Benefits

- CMSIS-View enables visibility into the dynamic execution of an application with minimal memory and timing overhead.
- Works on all Cortex-M devices with only simple debug adapters necessary.
- Compiler agnostic implementation allows simple integration in application projects.
- Events are captured with accurate time-stamps.
- Event Statistic functions allow you to collect and analyze statistical data about the code execution.
- Enables RTOS-aware debug for CMSIS-RTX and CMSIS-FreeRTOS.
- Provides logging capabilities for use in regression tests on Arm Virtual Hardware FVP models ([via semihosting](https://arm-software.github.io/CMSIS-View/latest/er_use.html#er_semihosting)).
- Natively supported in [Keil MDK uVision IDE](https://developer.arm.com/documentation/101407/0538/Debugging/Debug-Windows-and-Dialogs/Event-Recorder).
