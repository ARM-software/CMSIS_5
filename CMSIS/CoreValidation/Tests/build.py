#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import re

from datetime import datetime
from enum import Enum
from lxml.etree import XMLSyntaxError
from zipfile import ZipFile

from matrix_runner import main, matrix_axis, matrix_action, matrix_command, matrix_filter, ConsoleReport, CropReport, \
    TransformReport, JUnitReport


@matrix_axis("device", "d", "Device(s) to be considered.")
class DeviceAxis(Enum):
    CM0 = ('Cortex-M0', 'CM0')
    CM0PLUS = ('Cortex-M0plus', 'CM0plus', 'Cortex-M0+', 'CM0+')
    CM3 = ('Cortex-M3', 'CM3')
    CM4 = ('Cortex-M4', 'CM4')
    CM4FP = ('Cortex-M4FP', 'CM4FP')
    CM7 = ('Cortex-M7', 'CM7')
    CM7SP = ('Cortex-M7SP', 'CM7SP')
    CM7DP = ('Cortex-M7DP', 'CM7DP')
    CM23 = ('Cortex-M23', 'CM23')
    CM23S = ('Cortex-M23S', 'CM23S')
    CM23NS = ('Cortex-M23NS', 'CM23NS')
    CM33 = ('Cortex-M33', 'CM33')
    CM33S = ('Cortex-M33S', 'CM33S')
    CM33NS = ('Cortex-M33NS', 'CM33NS')
    CM35P = ('Cortex-M35P', 'CM35P')
    CM35PS = ('Cortex-M35PS', 'CM35PS')
    CM35PNS = ('Cortex-M35PNS', 'CM35PNS')
    CM55 = ('Cortex-M55', 'CM55')
    CM55S = ('Cortex-M55S', 'CM55S')
    CM55NS = ('Cortex-M55NS', 'CM55NS')
    CA5 = ('Cortex-A5', 'CA5')
    CA7 = ('Cortex-A7', 'CA7')
    CA9 = ('Cortex-A9', 'CA9')
    CA5NEON = ('Cortex-A5neon', 'CA5neon')
    CA7NEON = ('Cortex-A7neon', 'CA7neon')
    CA9NEON = ('Cortex-A9neon', 'CA9neon')
    
    def has_bootloader(self):
        return self in [
            DeviceAxis.CM23NS,
            DeviceAxis.CM33NS,
            DeviceAxis.CM35PNS,
            DeviceAxis.CM55NS
        ]


@matrix_axis("compiler", "c", "Compiler(s) to be considered.")
class CompilerAxis(Enum):
    AC5 = ('AC5', 'ArmCompiler5', 'armcc')
    AC6 = ('AC6', 'ArmCompiler6', 'armclang')
    AC6LTM = ('AC6LTM', 'ArmCompiler6-LTM', 'armclang-ltm')
    GCC = ('GCC')
    

@matrix_axis("optimize", "o", "Optimization level(s) to be considered.")
class OptimizationAxis(Enum):
    LOW = ('low', 'O1', 'O0', 'O1')
    MID = ('mid', 'O2', 'O1', 'O2' )
    HIGH = ('high', 'Ofast', 'Otime', 'Ofast')
    SIZE = ('size', 'Os', 'O2', 'Os')
    TINY = ('tiny', 'Oz', 'O3', 'O3')

    def for_compiler(self, compiler):
        COMPILER = {
            CompilerAxis.AC5: 2,
            CompilerAxis.AC6: 1,
            CompilerAxis.AC6LTM: 1,
            CompilerAxis.GCC: 3
        }        
        return self[COMPILER[compiler]]


FVP_MODELS = {
    DeviceAxis.CM0: ("FVP_MPS2_Cortex-M0", ""),
    DeviceAxis.CM0PLUS: ("FVP_MPS2_Cortex-M0plus", ""),
    DeviceAxis.CM3: ("FVP_MPS2_Cortex-M3", ""),
    DeviceAxis.CM4: ("FVP_MPS2_Cortex-M4", ""),
    DeviceAxis.CM4FP: ("FVP_MPS2_Cortex-M4", ""),
    DeviceAxis.CM7: ("FVP_MPS2_Cortex-M7", ""),
    DeviceAxis.CM7SP: ("FVP_MPS2_Cortex-M7", ""),
    DeviceAxis.CM7DP: ("FVP_MPS2_Cortex-M7", ""),
    DeviceAxis.CM23: ("FVP_MPS2_Cortex-M23", "cpu0"),
    DeviceAxis.CM23S: ("FVP_MPS2_Cortex-M23", "cpu0"),
    DeviceAxis.CM23NS: ("FVP_MPS2_Cortex-M23", "cpu0"),
    DeviceAxis.CM33: ("FVP_MPS2_Cortex-M33", "cpu0"),
    DeviceAxis.CM33S: ("FVP_MPS2_Cortex-M33", "cpu0"),
    DeviceAxis.CM33NS: ("FVP_MPS2_Cortex-M33", "cpu0"),
    DeviceAxis.CM35P: ("FVP_MPS2_Cortex-M35P", "cpu0"),
    DeviceAxis.CM35PS: ("FVP_MPS2_Cortex-M35P", "cpu0"),
    DeviceAxis.CM35PNS: ("FVP_MPS2_Cortex-M35P", "cpu0"),
    DeviceAxis.CM55: ("FVP_MPS2_Cortex-M55", "cpu0"),
    DeviceAxis.CM55S: ("FVP_MPS2_Cortex-M55", "cpu0"),
    DeviceAxis.CM55NS: ("FVP_MPS2_Cortex-M55", "cpu0"),
    DeviceAxis.CA5: ("FVP_VE_Cortex-A5x1", ""),
    DeviceAxis.CA7: ("FVP_VE_Cortex-A7x1", ""),
    DeviceAxis.CA9: ("FVP_VE_Cortex-A9x1", ""),
    DeviceAxis.CA5NEON: ("FVP_VE_Cortex-A5x1", ""),
    DeviceAxis.CA7NEON: ("FVP_VE_Cortex-A7x1", ""),
    DeviceAxis.CA9NEON: ("FVP_VE_Cortex-A9x1", ""),
}


@matrix_action
def build(config, results):
    """Build the selected configurations."""
    if config.device.has_bootloader():
        logging.info("Compiling Bootloader...")
        yield rtebuild(config, f"bootloader/{config.compiler[0].lower()}.rtebuild")
    
    logging.info("Compiling Tests...")
    yield rtebuild(config, f"{config.compiler[0].lower()}.rtebuild")
    
    if not all(r.success for r in results):
        return
        
    devname = config.device[1]
    file = f"CoreValidation_{config.compiler}_{devname}_{config.optimize[1]}.zip"
    logging.info(f"Archiving build output to {file}...")
    with ZipFile(file, "w") as archive:
        archive.write(f"build/arm{devname.lower()}/arm{devname.lower()}.elf")
        archive.write(f"build/arm{devname.lower()}/arm{devname.lower()}.map")
        if config.device.has_bootloader():
            archive.write(f"bootloader/build/arm{devname.lower()}/arm{devname.lower()}.elf")
            archive.write(f"bootloader/build/arm{devname.lower()}/arm{devname.lower()}.map")


@matrix_action
def run(config, results):
    """Run the selected configurations."""
    logging.info("Running CoreValidation in Fast Model...")
    yield fastmodel(config)
    
    try:
        results[0].test_report.write(f"corevalidation_{datetime.now().strftime('%Y%m%d%H%M%S')}.junit")
    except RuntimeError as e:
        if isinstance(e.__cause__ , XMLSyntaxError):
            logging.error("No valid test report found in model output!")
        else:
            logging.exception(e)

@matrix_command(needs_shell=True)
def rtebuild(config, spec):
    return ["rtebuild.py", "build", 
            "-c", spec,
            "-l",
            "-t", f"arm{config.device[1].lower()}", 
            "-o", f"optimize={config.optimize.for_compiler(config.compiler)}"]


@matrix_command(test_report=ConsoleReport() |
                            CropReport('<\?xml version="1.0"\?>', '</report>') |
                            TransformReport('validation.xsl') |
                            JUnitReport(title=lambda title, result: f"{result.command.config.compiler}."
                                                                    f"{result.command.config.device}."
                                                                    f"{result.command.config.optimize[1]}."
                                                                    f"{title}"))
def fastmodel(config):
    devname = config.device[1]
    cmdline = [FVP_MODELS[config.device][0], "-q", "--cyclelimit", 1000000000, "-f", f"config/ARM{devname}_config.txt"]
    cmdline += ["-a", f"{get_model_inst(config.device)}build/arm{devname.lower()}/arm{devname.lower()}.elf"]
    if config.device.has_bootloader():
        cmdline += ["-a", f"{get_model_inst(config.device)}bootloader/build/arm{devname.lower()}/arm{devname.lower()}.elf"]
    return cmdline


def get_model_inst(model):
    if FVP_MODELS[model][1]:
        return f"{FVP_MODELS[model][1]}="
    return ""


@matrix_filter
def filter_ac5_armv8m(config):
    return (config.compiler == CompilerAxis.AC5 and 
            config.device.match("CM[235][35]*"))


@matrix_filter
def filter_ac6ltm_armv81m(config):
    return (config.compiler == CompilerAxis.AC6LTM and 
            config.device.match("CM[35]5*"))


if __name__ == "__main__":
    main()
