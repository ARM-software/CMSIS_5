#! python

import os
import shutil
import sys

from datetime import datetime
from buildutils.builder import Device, Compiler, Axis, Step, BuildStep, RunModelStep, Builder, Filter

TARGET_FVP = 'FVP'
  
FVP_MODELS = { 
    Device.CM0      : { 'cmd': "FVP_MPS2_Cortex-M0",      'args': { 'limit': "1000000000", 'config': "config/ARMCM0_config.txt" } },
    Device.CM0PLUS  : { 'cmd': "FVP_MPS2_Cortex-M0plus",  'args': { 'limit': "1000000000", 'config': "config/ARMCM0plus_config.txt" } },
    Device.CM3      : { 'cmd': "FVP_MPS2_Cortex-M3",      'args': { 'limit': "1000000000", 'config': "config/ARMCM3_config.txt" } },
    Device.CM4      : { 'cmd': "FVP_MPS2_Cortex-M4",      'args': { 'limit': "1000000000", 'config': "config/ARMCM4_config.txt" } },
    Device.CM4FP    : { 'cmd': "FVP_MPS2_Cortex-M4",      'args': { 'limit': "1000000000", 'config': "config/ARMCM4FP_config.txt" } },
    Device.CM7      : { 'cmd': "FVP_MPS2_Cortex-M7",      'args': { 'limit': "1000000000", 'config': "config/ARMCM7_config.txt" } },
    Device.CM7SP    : { 'cmd': "FVP_MPS2_Cortex-M7",      'args': { 'limit': "1000000000", 'config': "config/ARMCM7SP_config.txt" } },
    Device.CM7DP    : { 'cmd': "FVP_MPS2_Cortex-M7",      'args': { 'limit': "1000000000", 'config': "config/ARMCM7DP_config.txt" } },
    Device.CM23     : { 'cmd': "FVP_MPS2_Cortex-M23",     'args': { 'limit': "1000000000", 'config': "config/ARMCM23_config.txt",           'target': "cpu0" } },
    Device.CM33     : { 'cmd': "FVP_MPS2_Cortex-M33",     'args': { 'limit': "1000000000", 'config': "config/ARMCM33_config.txt",           'target': "cpu0" } },
    Device.CM23NS   : { 'cmd': "FVP_MPS2_Cortex-M23",     'args': { 'limit': "1000000000", 'config': "config/ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    Device.CM33NS   : { 'cmd': "FVP_MPS2_Cortex-M33",     'args': { 'limit': "1000000000", 'config': "config/ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } },
    Device.CM23S    : { 'cmd': "FVP_MPS2_Cortex-M23",     'args': { 'limit': "1000000000", 'config': "config/ARMCM23_TZ_config.txt",        'target': "cpu0" } },
    Device.CM33S    : { 'cmd': "FVP_MPS2_Cortex-M33",     'args': { 'limit': "1000000000", 'config': "config/ARMCM33_DSP_FP_TZ_config.txt", 'target': "cpu0" } },
    Device.CA5      : { 'cmd': "fvp_ve_cortex-a5x1.exe",  'args': { 'limit': "1000000000", 'config': "config/ARMCA5_config.txt" } },
    Device.CA7      : { 'cmd': "fvp_ve_cortex-a7x1.exe",  'args': { 'limit': "1000000000", 'config': "config/ARMCA7_config.txt" } },
    Device.CA9      : { 'cmd': "fvp_ve_cortex-a9x1.exe",  'args': { 'limit': "1000000000", 'config': "config/ARMCA9_config.txt" } },
    Device.CA5NEON  : { 'cmd': "fvp_ve_cortex-a5x1.exe",  'args': { 'limit': "1000000000", 'config': "config/ARMCA5neon_config.txt" } },
    Device.CA7NEON  : { 'cmd': "fvp_ve_cortex-a7x1.exe",  'args': { 'limit': "1000000000", 'config': "config/ARMCA7neon_config.txt" } },
    Device.CA9NEON  : { 'cmd': "fvp_ve_cortex-a9x1.exe",  'args': { 'limit': "1000000000", 'config': "config/ARMCA9neon_config.txt" } }
  }

def format(str, dev, cc, target = "FVP", **kwargs):
  return str.format(dev = dev.value[0], cc = cc.value, target = target, **kwargs)
  
def testProject(dev, cc, target):
  rtebuild = format("{dev}/{cc}/default.rtebuild", dev = dev, cc = cc, target=target)
  if os.path.exists(rtebuild):
    return [
        rtebuild,
        format("{dev}/{cc}/build/{target}/{target}.elf", dev = dev, cc = cc, target=target)
      ]
  elif (cc == Compiler.AC5) or (cc == Compiler.AC6):
    return [
        format("{dev}/{cc}/CMSIS_CV.uvprojx", dev = dev, cc = cc),
        format("{dev}/{cc}/Objects/CMSIS_CV.axf", dev = dev, cc = cc)
      ]
  elif (cc == Compiler.AC6LTM):
    return [
        format("{dev}/{cc}/CMSIS_CV.uvprojx", dev = dev, cc = Compiler.AC6),
        format("{dev}/{cc}/Objects/CMSIS_CV.axf", dev = dev, cc = Compiler.AC6)
      ]
  elif (cc == Compiler.GCC):
    return [
        format("{dev}/{cc}/CMSIS_CV.uvprojx", dev = dev, cc = cc),
        format("{dev}/{cc}/Objects/CMSIS_CV.elf", dev = dev, cc = cc)
      ]
  elif (cc == Compiler.IAR):
    return [
        format("{dev}/{cc}/CMSIS_CV.ewp", dev = dev, cc = cc),
        format("{dev}/{cc}/{target}/Exe/CMSIS_CV.out", dev = dev, cc = cc, target = target)
      ]
  raise "Unknown compiler!"

def bootloaderProject(dev, cc, target):
  rtebuild = format("{dev}/{cc}/Bootloader/default.rtebuild", dev = dev, cc = cc, target=target)
  if os.path.exists(rtebuild):
    return [
        rtebuild,
        format("{dev}/{cc}/Bootloader/build/{target}/{target}.elf", dev = dev, cc = cc, target=target)
      ]
  elif (cc == Compiler.AC5) or (cc == Compiler.AC6):
    return [
        format("{dev}/{cc}/Bootloader/Bootloader.uvprojx", dev = dev, cc = cc),
        format("{dev}/{cc}/Bootloader/Objects/Bootloader.axf", dev = dev, cc = cc)
      ] 
  elif (cc == Compiler.AC6LTM):
    return [
        format("{dev}/{cc}/Bootloader/Bootloader.uvprojx", dev = dev, cc = Compiler.AC6),
        format("{dev}/{cc}/Bootloader/Objects/Bootloader.axf", dev = dev, cc = Compiler.AC6)
      ] 
  elif (cc == Compiler.GCC):
    return [
        format("{dev}/{cc}/Bootloader/Bootloader.uvprojx", dev = dev, cc = cc),
        format("{dev}/{cc}/Bootloader/Objects/Bootloader.elf", dev = dev, cc = cc)
      ] 
  elif (cc == Compiler.IAR):
    return [
        format("{dev}/{cc}/Bootloader/Bootloader.ewp", dev = dev, cc = cc),
        format("{dev}/{cc}/Bootloader/{target}/Exe/Bootloader.out", dev = dev, cc = cc, target = target)
      ]
  raise "Unknown compiler!"

def projects(step, config):
  dev = config['device']
  cc = config['compiler']
  target = config['target']
  
  projects = []
  blPrj = bootloaderProject(dev, cc, target)
  if os.path.exists(blPrj[0]):
    projects += [ blPrj[0] ]
  
  projects += [ testProject(dev, cc, target)[0] ]
  
  return projects
  
def images(step, config):
  dev = config['device']
  cc = config['compiler']
  target = config['target']
  
  images = [ testProject(dev, cc, target)[1] ]
  blPrj = bootloaderProject(dev, cc, target)
  if os.path.exists(blPrj[1]):
    images += [ blPrj[1] ]
  
  return images

def storeResult(step, config, cmd):
  result = format("result_{cc}_{dev}_{target}_{now}.xml", config['device'], config['compiler'], config['target'], now = datetime.now().strftime("%Y%m%d%H%M%S"))
  resultfile = step.storeResult(cmd, result, format("{cc}.{dev}.{target}", config['device'], config['compiler'], config['target']))
  if not resultfile:
    cmd.appendOutput("Storing results failed!");
    cmd.forceResult(1)
  
def create():
  deviceAxis = Axis("device", abbrev="d", values=Device, desc="Device(s) to be considered.")
  compilerAxis = Axis("compiler", abbrev="c", values=Compiler, desc="Compiler(s) to be considered.")
  targetAxis = Axis("target", abbrev="t", values=[ TARGET_FVP ], desc="Target(s) to be considered.")
  
  buildStep = BuildStep("build", abbrev="b", desc="Build the selected configurations.")
  buildStep.projects = projects 
  buildStep.target = lambda step, config: config['target']
  
  runStep = RunModelStep("run", abbrev="r", desc="Run the selected configurations.")
  runStep.images = images
  runStep.model = lambda step, config: FVP_MODELS[config['device']]
  runStep.post = storeResult

  debugStep = RunModelStep("debug", abbrev="d", desc="Debug the selected configurations.")
  debugStep.images = images
  debugStep.args = lambda step, config: { 'cadi' : True }
  debugStep.model = lambda step, config: FVP_MODELS[config['device']]
  
  filterAC5 = Filter().addAxis(compilerAxis, Compiler.AC5).addAxis(deviceAxis, "CM[23]3*")
  filterAC6LTM = Filter().addAxis(compilerAxis, Compiler.AC6LTM).addAxis(deviceAxis, "CM[23]3*")

  builder = Builder()
  builder.addAxis([ compilerAxis, deviceAxis, targetAxis ])
  builder.addStep([ buildStep, runStep, debugStep ])
  builder.addFilter([ filterAC5, filterAC6LTM ])

  return builder

def complete(builder, success):
  builder.saveJunitResult("build_{now}.xml".format(now = datetime.now().strftime("%Y%m%d%H%M%S")))
