#!/usr/bin/env python3
#
# Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import re
import sys
import json
import copy
import glob
import time
import queue
import shutil
import serial
import argparse
import threading
import subprocess

from os import path
from termcolor import colored

OUTPUT = "Output/"
BASE_PATH = "../../"
CMSIS_PATH = "../../../../../"
UNITY_PATH = "Unity/"
UNITY_BASE = BASE_PATH + UNITY_PATH
UNITY_SRC = UNITY_BASE + "src/"
CMSIS_FLAGS = " -DARM_MATH_DSP -DARM_MATH_LOOPUNROLL"


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMSIS-NN unit tests.",
                                     epilog="Runs on all connected HW supported by Mbed.")
    parser.add_argument('--testdir', type=str, default='TESTRUN', help="prefix of output dir name")
    parser.add_argument('-s', '--specific-test', type=str, default=None, help="Run a specific test, e.g."
                        " -s TestCases/test_arm_avgpool_s8 (also this form will work: -s test_arm_avgpool_s8)."
                        " So basically the different options can be listed with:"
                        " ls -d TestCases/test_* -1")
    parser.add_argument('-c', '--compiler', type=str, default='GCC_ARM', choices=['GCC_ARM', 'ARMC6'])
    parser.add_argument('--download-and-generate-test-runners', dest='download_and_generate', action='store_true',
                        help="Just download Unity and generate test runners if needed")

    args = parser.parse_args()
    return args


def error_handler(code, text=None):
    print("Error: {}".format(text))
    sys.exit(code)


def detect_targets(targets):
    process = subprocess.Popen(['mbedls'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    print(process.stdout.readline().strip())
    while True:
        line = process.stdout.readline()
        print(line.strip())
        if not line:
            break
        if re.search(r"^\| ", line):
            words = (line.split('| '))
            target = {"model": words[1].strip(),
                      "name": words[2].strip()[:-1].replace('[', '_'),
                      "port": words[4].strip(),
                      "tid": words[5].strip()}  # Target id can be used to filter out targets
            targets.append(target)
    return_code = process.poll()
    if return_code != 0:
        error_handler(return_code, 'RETURN CODE {}'.format(process.stderr.read()))


def run_command(command, error_msg=None, die=True):
    # TODO handle error:
    # cp: error writing '/media/mannil01/NODE_F411RE/TESTRUN_NUCLEO_F411RE_GCC_ARM.bin': No space left on device
    # https://os.mbed.com/questions/59636/STM-Nucleo-No-space-left-on-device-when-/

    # print(command)
    command_list = command.split(' ')
    process = subprocess.run(command_list)
    if die and process.returncode != 0:
        error_handler(process.returncode, error_msg)
    return process.returncode


def detect_architecture(target_name, target_json):
    arch = None

    try:
        with open(target_json, "r") as read_file:
            data = json.load(read_file)

            if 'core' in data[target_name]:
                core = data[target_name]['core']
            elif 'inherits' in data[target_name]:
                target_inherits = data[target_name]['inherits'][0]
                core = data[target_inherits]['core']
            else:
                raise Exception("Cannot detect architecture")

            if core:
                arch = core[:9]
                if core[:8] == 'Cortex-M':
                    return arch
            error_handler(168, 'Unsupported target: {} with architecture: {}'.format(
                target_name, arch))
    except Exception as e:
        error_handler(167, e)

    return arch


def test_target(target, args, main_test):
    result = 3
    compiler = args.compiler
    target_name = target['name']
    target_model = target['model']
    cmsis_flags = None
    unittestframework = 'UNITY_UNITTEST'

    dir_name = OUTPUT + args.testdir + '_' + unittestframework + '_' + target_name + '_' + compiler

    os.makedirs(dir_name, exist_ok=True)
    start_dir = os.getcwd()
    os.chdir(dir_name)

    try:
        target_json = 'mbed-os/targets/targets.json'
        mbed_path = BASE_PATH + 'Mbed/'

        if not path.exists("mbed-os.lib"):
            print("Initializing mbed in {}".format(os.getcwd()))
            shutil.copyfile(mbed_path + 'mbed-os.lib', 'mbed-os.lib')
            shutil.copyfile(mbed_path + 'mbed_app.json', 'mbed_app.json')
            run_command('mbed config root .')
            run_command('mbed deploy')

        arch = detect_architecture(target_model, target_json)
        if arch == 'Cortex-M4' or arch == 'Cortex-M7':
            cmsis_flags = CMSIS_FLAGS

        print("----------------------------------------------------------------")
        print("Running {} on {} target: {} with compiler: {} and cmsis flags: {} in directory: {} test: {}\n".format(
            unittestframework, arch, target_name, compiler, cmsis_flags, os.getcwd(), main_test))

        die = False
        flash_error_msg = 'failed to flash'
        mbed_command = "compile"
        test = ''
        additional_options = ' --source ' + BASE_PATH + main_test + \
                             ' --source ' + UNITY_SRC + \
                             ' --profile ' + mbed_path + 'release.json' + \
                             ' -f'

        result = run_command("mbed {} -v -m ".format(mbed_command) + target_model + ' -t ' + compiler +
                             test +
                             ' --source .'
                             ' --source ' + BASE_PATH + 'TestCases/Utils/'
                             ' --source ' + CMSIS_PATH + 'NN/Include/'
                             ' --source ' + CMSIS_PATH + 'DSP/Include/'
                             ' --source ' + CMSIS_PATH + 'Core/Include/'
                             ' --source ' + CMSIS_PATH + 'NN/Source/ConvolutionFunctions/'
                             ' --source ' + CMSIS_PATH + 'NN/Source/PoolingFunctions/'
                             ' --source ' + CMSIS_PATH + 'NN/Source/NNSupportFunctions/'
                             ' --source ' + CMSIS_PATH + 'NN/Source/FullyConnectedFunctions/'
                             ' --source ' + CMSIS_PATH + 'NN/Source/SoftmaxFunctions/'
                             ' --source ' + CMSIS_PATH + 'NN/Source/SVDFunctions/'
                             + cmsis_flags +
                             additional_options,
                             flash_error_msg, die=die)

    except Exception as e:
        error_handler(166, e)

    os.chdir(start_dir)
    return result


def read_serial_port(ser, inputQueue, stop):
    while True:
        if stop():
            break
        line = ser.readline()
        inputQueue.put(line.decode('latin-1').strip())


def test_target_with_unity(target, args, main_test):
    port = target['port']
    stop_thread = False
    baudrate = 9600
    timeout = 30
    inputQueue = queue.Queue()
    tests = copy.deepcopy(target["tests"])

    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
    except Exception as e:
        error_handler(169, "serial exception: {}".format(e))

    # Clear read buffer
    time.sleep(0.1)  # Workaround in response to: open() returns before port is ready
    ser.reset_input_buffer()

    serial_thread = threading.Thread(target=read_serial_port, args=(ser, inputQueue, lambda: stop_thread), daemon=True)
    serial_thread.start()

    test_target(target, args, main_test)

    start_time = time.time()
    while time.time() < start_time + timeout:
        if inputQueue.qsize() > 0:
            str_line = inputQueue.get()
            print(str_line)
            test = None
            try:
                test = str_line.split(':')[2]
                test_result = ':'.join(str_line.split(':')[2:4])
            except IndexError:
                pass
            if test in tests:
                tests.remove(test)
                target[test]["tested"] = True
                if test_result == test + ':PASS':
                    target[test]["pass"] = True
            if len(tests) == 0:
                break

    stop_thread = True
    serial_thread.join()
    ser.close()


def print_summary(targets):
    """
    Return 0 if all test passed
    Return 1 if all test completed but one or more failed
    Return 2 if one or more tests did not complete or was not detected
    """
    passed = 0
    failed = 0
    tested = 0
    expected = 0
    return_code = 3
    verdict_pass = colored('[ PASSED ]', 'green')
    verdict_fail = colored('[ FAILED ]', 'red')
    verdict_error = colored('[ ERROR ]', 'red')

    print("-----------------------------------------------------------------------------------------------------------")

    # Find all passed and failed
    for target in targets:
        for test in target["tests"]:
            expected += 1
            if target[test]["tested"]:
                tested += 1
            if target[test]["pass"]:
                passed += 1
            else:
                failed += 1

    if tested != expected:
        print("{} Not all tests found".format(verdict_error))
        print("{} Expected: {} Actual: {}".format(verdict_error, expected, tested))
        return_code = 2
    elif tested == passed:
        return_code = 0
    else:
        return_code = 1

    # print all test cases
    sorted_tc = []
    for target in targets:
        for test in target["tests"]:
            if not target[test]["tested"]:
                tc_verdict = verdict_error
            elif target[test]["pass"]:
                tc_verdict = verdict_pass
            else:
                tc_verdict = verdict_fail
            sorted_tc.append("{} {}: {}".format(tc_verdict, target["name"], test))
    sorted_tc.sort()
    for tc in sorted_tc:
        print(tc)

    total = 0
    if (passed > 0):
        total = passed / expected
    if (total == 1.0):
        verdict = verdict_pass
    else:
        verdict = verdict_fail
    print("{} Summary: {} tests in total passed on {} targets ({})".
          format(verdict, passed, len(targets), ', '.join([t['name'] for t in targets])))
    print("{} {:.0f}% tests passed, {} tests failed out of {}".format(verdict, total*100, failed, expected))

    return return_code


def test_targets(args):
    """
    Return 0 if successful
    Return 3 if no targets are detected
    Return 4 if no tests are found
    """
    result = 0
    targets = []
    main_tests = []

    if not args.download_and_generate:
        detect_targets(targets)
        if len(targets) == 0:
            print("No targets detected!")
            return 3

    download_unity()

    if not parse_tests(targets, main_tests, args.specific_test):
        print("No tests found?!")
        return 4

    if args.download_and_generate:
        return result

    for target in targets:
        for tst in main_tests:
            test_target_with_unity(target, args, tst)

    result = print_summary(targets)

    return result


def download_unity(force=False):
    unity_dir = UNITY_PATH
    unity_src = unity_dir+"src/"
    process = subprocess.run(['mktemp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    download_dir = process.stdout.strip()
    run_command("rm -f {}".format(download_dir))
    download_dir += '/'

    # Check if already downloaded
    if not force and path.isdir(unity_dir) and path.isfile(unity_src+"unity.c") and path.isfile(unity_src+"unity.h"):
        return

    if path.isdir(download_dir):
        shutil.rmtree(download_dir)
    if path.isdir(unity_dir):
        shutil.rmtree(unity_dir)
    os.mkdir(unity_dir)
    os.makedirs(download_dir, exist_ok=False)
    current_dir = os.getcwd()
    os.chdir(download_dir)
    process = subprocess.Popen('curl -LJ https://api.github.com/repos/ThrowTheSwitch/Unity/tarball/v2.5.0 --output unity_tarball.tar.gz'.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    for line in process.stderr:
        print(line.strip())
    print()
    for line in process.stdout:
        pass
    if not line:
        error_handler(171)
    downloaded_file = download_dir + "unity_tarball.tar.gz"
    os.chdir(current_dir)
    try:
        filename_base = downloaded_file.split('-')[0]
    except IndexError as e:
        error_handler(174, e)
    if not filename_base:
        error_handler(175)
    run_command("tar xzf "+downloaded_file+" -C "+unity_dir+" --strip-components=1")
    os.chdir(current_dir)

    # Cleanup
    shutil.rmtree(download_dir)


def parse_generated_test_runner(test_runner):
    parsed_functions = ['setUp', 'tearDown', 'resetTest', 'verifyTest']

    def is_func_to_parse(func):
        for f in parsed_functions:
            if f in func:
                return True
        return False

    with open(test_runner, "r") as f:
        lines = f.readlines()
    with open(test_runner, "w") as f:
        for line in lines:
            sline = line.strip('\n')
            if not re.search(r"\(void\);", sline):
                f.write(line)
            else:
                if not is_func_to_parse(sline):
                    f.write(line)


def parse_tests(targets, main_tests, specific_test=None):
    """
    Generate test runners, extract and return path to unit test(s).
    Also parse generated test runners to avoid warning: redundant redeclaration.
    Return True if successful.
    """
    test_found = False
    directory = 'TestCases'

    if specific_test and '/' in specific_test:
        specific_test = specific_test.strip(directory).replace('/', '')

    for dir in next(os.walk(directory))[1]:
        if re.search(r'test_arm', dir):
            if specific_test and dir != specific_test:
                continue
            test_found = True
            testpath = directory + '/' + dir + '/Unity/'
            ut_test_file = None
            for content in os.listdir(testpath):
                if re.search(r'unity_test_arm', content):
                    ut_test_file = content
            if ut_test_file is None:
                print("Warning: invalid path: ", testpath)
                continue
            main_tests.append(testpath)
            ut_test_file_runner = path.splitext(ut_test_file)[0] + '_runner' + path.splitext(ut_test_file)[1]
            test_code = testpath + ut_test_file
            test_runner_path = testpath + 'TestRunner/'
            if not os.path.exists(test_runner_path):
                os.mkdir(test_runner_path)
            test_runner = test_runner_path + ut_test_file_runner
            for old_files in glob.glob(test_runner_path + '/*'):
                if not old_files.endswith('readme.txt'):
                    os.remove(old_files)

            # Generate test runners
            run_command('ruby '+UNITY_PATH+'auto/generate_test_runner.rb ' + test_code + ' ' + test_runner)
            test_found = parse_test(test_runner, targets)
            if not test_found:
                return False

            parse_generated_test_runner(test_runner)

    if not test_found:
        return False
    return True


def parse_test(test_runner, targets):
    tests_found = False

    # Get list of tests
    try:
        read_file = open(test_runner, "r")
    except IOError as e:
        error_handler(170, e)
    else:
        with read_file:
            for line in read_file:
                if not line:
                    break
                if re.search(r"  run_test\(", line) and len(line.strip().split(',')) == 3:
                    function = line.strip().split(',')[0].split('(')[1]
                    tests_found = True
                    for target in targets:
                        if 'tests' not in target.keys():
                            target['tests'] = []
                        target["tests"].append(function)
                        target[function] = {}
                        target[function]["pass"] = False
                        target[function]["tested"] = False
    return tests_found


if __name__ == '__main__':
    args = parse_args()
    sys.exit(test_targets(args))
