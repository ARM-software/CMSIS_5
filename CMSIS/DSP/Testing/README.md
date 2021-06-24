# TEST FRAMEWORK

This framework is for our own internal use. We decided to release it but, at least in short term, we won't give any help or support about it.

## REQUIREMENTS

### Test descriptions

#### R1 : The tests shall be described in a file
We need a source of truth which is describing all the tests and can be used
to generate code, format output etc ...

#### R2 : The test description should support a hierarchy
We have lots of tests. We need to be able to organize them in a
hierarchical way

#### R3 : A test shall be uniquely identified
We need a way to identify in an unique way each test to ensure traceability and enable to create
history of test results and benchmark.

#### R4 : The unique identifier shall not change when tests are added or removed.
It is important to keep traceability.

#### R5 : The test description shall list the test patterns and input patterns required by the tests

#### R6 : It shall be possible to parametrize the tests

For benchmarking, we may need to vary some dimensions of the tests (like input length).
The tests may depend on several parameters (width, height etc ...)
We need to be able to specify how those parameters are varied.

#### R7 : It shall be possible to specify a subset of parameters (which could be empty) to compute regression.
For instance, if our test is dependent on a vector size, we may want to compute a linear regression
to know how the performances are dependent on the vector size.

But, our test may also depend on another parameter B which is not interesting us in the regression. In that case, the regression formula should not take into account B. And we would have several regression formula for each value of the parameter B.

The parameters of the tests would be Vector Size and B but the Summary parameter only Vector Size.

#### R8 : The concept of a test suite shall be supported.
A test suite is a set of tests packaged with some data.

### Test execution

For following requirements, we define a device under tests (DUT) as the place where the function to test is executed. But the test itself (to check that the execution has been successful could be running on the DUT or on a host like a PC).


#### R9 : The memory should be cleaned between 2 tests
A test should start (as far as possible) in a clean state. There should not be interferences between the tests.

#### R10 : The test may be run on the DUT or on the host

#### R11 : Output of tested functions could be dumped or not

#### R12 : The tested function should not know where are the patterns and how to get them 

#### R13 : Control of the tests could run on the DUT but could also be run on a host

#### R14 : Summary of test execution shall support several formats
(CSV, HTML, Text etc ...)

#### R15 : One should not assume the test environment on the DUT has access to IOs.


## DESIGN PRINCIPLES

The design is a consequence of all the requirements.

### Test description

A test description file is defined with a specific syntax to support R1 to R8.

#### Hierachical structure

    group Root {
        class = Root
        group DSP Test {
            class = DSPTest
            folder = DSP
            suite Basic Tests {
               class = BasicTests
               folder = BasicMaths

The tests are organized in a hierarchy. For each node of the hierarchy, a C++ class is specified.
The script processTest.py is generating C++ codee for the group.
For the test suite, the script is generating a partial implementation since a test suite is containing tests and you need to add the test themselves.

The patterns, output of tests, parameters are also following a hierarchical structure. But they do not need
to be organized in exactly the same way. So, the folder property of a node is optional.

A folder can be reused for different nodes. For instance, you may have a suite for testing and one for benchmarking and both may use the same pattern folder.

A test suite is more complex than a group since it contains the description of the tests and related information.

#### Test suite

The simplest suite is just containing functions:

    suite Basic Tests {
           class = BasicTests
           folder = BasicMaths
     
           Functions {
             Test arm_add_f32:test_add_f32
           }
    }

A function is described with some text and followed by the name of the function in the C++ class.
The text is used when reporting the results of the tests.

The same function can be used for different tests in the suite. The tests will be different due to different input data or parameters.

A test is requiring input patterns, reference patterns and outputs (to be compared to the reference).
Since the test must not know where is the data and how to get it, this information is provided in the test description file.

So, the test suite would be:

    suite Basic Tests {
           class = BasicTests
           folder = BasicMaths
     
           Pattern INPUT1_F32_ID : Input1_f32.txt 
           Pattern INPUT2_F32_ID : Input2_f32.txt 
           Pattern REF_ADD_F32_ID : Reference1_f32.txt
           Output  OUT_SAMPLES_F32_ID : Output
     
           Functions {
             Test arm_add_f32:test_add_f32
           }
    }

A pattern or output description is an ID (to be used in the code) followed by a filename.

The file is in the folder defined with the folder properties of the group / suites.

The root folder for pattern files and output files is different.

#### Benchmarks

A benchmark will often have to be run with different lengths for the input.
So we need a way to communicate arguments to a function.

We make the assumption that those arguments are integers.
In the benchmark results, we may want to generate a CSV (or any other format) with different columns for those arguments.

And we may want to compute a regression formula using only a subset of those arguments.

So, we have the possibility in the suite section to add a parameter section to describe all of this.

    suite Complex Tests {
            class = ComplexTests
            folder = ComplexMaths
     
            ParamList {
                A,B,C
                Summary A,B
                Names "Param A", "Param B"
                Formula "A*B"
            }
     
            Pattern INPUT1_F32_ID : Input1_f32.txt 


In above example we declare that the functions of the suite are using 3 parameters named A,B and C.
We declare that a regression formula will use only A and B. So for each C value, we will get a different
regression formula.

We list the names to use when formatting the output of benchmarks.
We define a regression formula using R syntax. (We do not write "cycles ~ A*B" but only "A*B")

Once parameters have been described, we need a way to feed parameter values to a test.

There are 2 ways. First way is a parameter file. Problem of a parameter file when it has to be included in the test (C array) is that it may be big. So, we also have a parameter generator. It is less flexible but enough for lot of cases.

Those parameters values, when specified with a file, are described with:

            Output  OUT_SAMPLES_F32_ID : Output
            Params PARAM1_ID : Params1.txt

They follow the outputs section and use similar syntax.

When the parameter is specified with a generator then the syntax is :

    Params PARAM3_ID = {
                A = [1,3,5]
                B = [1,3,5]
                C = [1,3,5]
            }

This generator will compute the cartesian product of the 3 lists.

To use parameters with a function the syntax is:

    Functions {
               Test A:testA -> PARAM3_ID
            } -> PARAM1_ID

PARAM1_ID is the default applied to all functions.
In this example we decide to use PARAM3_ID for the testA function.

#### File formats
Pattern files have the following format:

    W
    128
    // 1.150898
    0x3f93509c
    ...
     

First line if the word size (W,H or B)
Second line is the number of samples
Then for each samples we have an human representation of the value:
// 1.150898

and an hexadecimal representation
0x3f93509c

Output files are only containing the hexadecimal values.

Parameters files have the following format:

    81
    1
    1
    1
    1
    1
    3
    ...

First line is the number of samples. Then the samples.

First line must be a multiple of the number of parameters. In our above example we have 3 parameters A,B,C.
So, the number of possible run must be a multiple of 3 since we need to specify values for all parameters.

#### disabled

Any node (Group, Suite or Function) can be disabled by using disabled { ...}.

A disabled group/suite/test is not executed (and its code not generated for group/suite).
Using disabled for tests is allowing to disable a test without changing the test ID of following tests.


### Memory manager
Memory manager is coming from requirement R9
Its API is defined by virtual class Memory. An implementation ArrayMemory is provided which is using a buffer.
The details of the APIs are in Test.h

A memory manager can provide new buffer, free all the already allocated buffers and give a generation number which is incremented each time all buffer are released.

#### Runner 
According to R13 , the test may be controlled on the DUT or from an external host.
It is implemented with a Runner class. The only implementation provided is IORunner,

A Runner is just an implementation of the visitor pattern. A runner is applied to the tree of tests.
In case of the IO runner, an IO mechanism and a memory manager must be provided.

The runner is running a test and for benchmark measuring the cycles.
Cycles measurement can be based on internal counter or external trace.
Generally, there is a calibration at beginning of the Runner to estimate the overhead of
cycle measurements. This overhead is then removed when doing the measurement.

#### IO
According to R12 and R15, tests do not know how to access patterns. It is a responsiblity implemented with the IO, Pattern and PatternMgr.

IO is about loading patterns and dumping output. It is not about IO in general.
We provide 2 IO implementations : Semihosting and FPGA.

FPGA is when you need to run the tests in a constrained environment where you only have stdout. The inputs of tests are in C array. The script processTest.py will generate those C arrays.

Patterns is the interface to patterns and output from the test point of view.
They will return NULL when a pattern is still referencing a generation of memory older than the current one.

PatternMgr is the link between IO and Memory and knows how to load a pattern and save it into memory.

#### Dump feature
According to R10 and R11, one must be able to disable tests done on the DUT and dump the output so that the test itself can be done on the host.
When instantiating a runner, you can specify the running mode with an enum. For instance Testing::kTestAndDump.
There are 3 modes, Test only, Dump only, Test and dump. 

In dump only mode, tests using pattern will fail but the tests will be considered as passed (because we are only interested in the output).

But it means that no test using patterns shoudl be used in the middle of the test or some part of it may not be executed. Those tests must be kept at the end.

#### processResult
For R14, we have a python script which will process the result of tests and format it into several possible formats like text, CSV, Mathematica dataset. 


## HOW TO RUN IT

### Needed packages

    pip install pyparsing 
    pip install Colorama
    
If you want to compute summary statistics with regression:

    pip install statsmodels
    pip install numpy
    pip install panda

If you want to run the script which is launching all the tests on all possible configurations then
you'll need yaml:

    pip install pyyaml

### Generate the test patterns in Patterns folder

We have archived lot of test patterns on github. So this step is needed only if you write new test patterns.

    cd Testing
    python PatternGeneration\BasicMaths.py


### Generate the cpp,h and txt files from the desc.txt file

First time the project is cloned from github, you'll need to create some missing folders as done
in the script createDefaultFolder.sh

Those folders are used to contain the files generated by the scripts.

Once those folders have been created. You can use following commands to create the generated C files.

    cd ..

    python preprocess.py -f desc.txt 

This will create a file Output.pickle which is containing a Python object representing
the parsed data structure. It is done because parsing a big test description file is quite slow.

So, it is needed to be done only once or if you modify the test description file.

Then, the tests can be processed to configure the test environment with

    python processTests.py -f Output.pickle

or just

    python processTests.py

You can also use the -e option (for embedded). It will include all the patterns (for the selected tests) into a C array. It is the preferred method if you want to run on a board. In below examples, we will
always use -e option.

    python processTests.py -e

You can pass a C++ class to specifiy that you want to generate tests only for a specific group or suite.

    python processTests.py -e BasicTests

You can add a test ID to specify that you wan to run only a specific test in the suite:

    python processTests.py -e BasicTests 4

Before filtering desc.txt by using a C++ class, you should (at least once) parse the full file without filtering.

The reason is that the cmake build is not aware of the filtering and will include some source files which
are not needed when filtered out. So those files should at least be present to allow the compilation to proceed. They need to be generated at least once.


### Generate the build system

    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH="path/to/tools" -DCMAKE_TOOLCHAIN_FILE=../../armcc.cmake -DARM_CPU="cortex-a5" -DPLATFORM="FVP" -DBENCHMARK=OFF -G "Unix Makefiles" ..

If BENCHMARK=ON is used, other options should be enabled to have better performances.

### Build and run the tests

Folder Output/BasicMaths should exist. For example, on Windows with ArmDS:

    cd build
    make VERBOSE=1
    "C:\Program Files\ARM\Development Studio 2019.0\sw\models\bin\FVP_VE_Cortex-A5x1.exe" -a Testing  > result.txt

### Parse the results

    cd ..
    python processResult.py -e -r build\result.txt

-e option is needed if the mode -e was used with processTests because the output has a different
format with or without -e option.


Some cycles are displayed with the test status (passed or failed). **Don't trust** those cycles for a benchmark.

At this point they are only an indication. The timing code will have to be tested and validated.

### Generate summary statistics

The parsing of the results may have generated some statistics in FullBenchmark folder.

The script summaryBench can parse those results and compute regression formula.

    python summaryBench.py -r build\result.txt

The file result.txt must be placed inside the build folder for this script to work.
Indeed, this script is using the path to result.txt to also find the file currentConfig.csv which has
been created by the cmake command.

The Output.pickle file is used by default. It can be changed with -f option.

The output of this script may look like:

    "ID","CATEGORY","Param C","Regression","MAX"
    1,"DSP:ComplexMaths",1,"225.3749999999999 + A * 0.7083333333333606 + B * 0.7083333333333641 + A*B * 1.3749999999999876",260

Each test is uniquely identified with the CATEGORY and test ID (ID in the suite).
The MAX column is the max of cycles computed for all values of A and B which were used for this benchmark.

### Other tools

To convert some benchmark to an older format.
The PARAMS must be compatible between all suites which are children of AGroup

    python convertToOld.py -e AGroup

Output.pickle is used by default. It can be changed with -f option.

To add a to sqlite3 databse:

    python addToDB.py -e AGroup

Output.pickle is used by default. It can be changed with -f option.

The database must be created with createDb.sql before this script can be used.

### Semihosting or FPGA mode
The script processTests and processResult must be used with additional option -e for the FPGA (embedded mode)

testmain.cpp, in semihosting mode, must contain:

```cpp
Client::Semihosting io("../TestDesc.txt","../Patterns","../Output");
```

In FPGA (embedded mode), this lne must be replaced with:

```cpp
Client::FPGA io(testDesc,patterns);
```

testDesc and patterns are char* generated by the script processTests and containing the description
of the tests to run and the test pattern samples to be used.

### Dumping outputs 

To dump the output of the tests, the line

```cpp
Client::IORunner runner(&io,&mgr,Testing::kTestOnly);
```

Must be replaced by

```cpp
Client::IORunner runner(&io,&mgr,Testing::DumpOnly);
```

or

```cpp
Client::IORunner runner(&io,&mgr,Testing::kTestAndDump);
```

and of course, the test must contain a line to dump the outputs.

In DumpOnly mode, reference patterns are not loaded and the test assertions are "failing" but reporting passed.

So, if a test is in the middle of useful code, some part of the code will not execute.

As consequence, if you intend to use the DumpOnly mode, you must ensure that all test assertions are at the
end of your test.

## testmain.cpp

To start the tests you need to:

* Allocate a memory manager
* Choose IO (Semihosting or FPGA)
* Instantiate a pattern manager (linking IO and memory)
* Choose a test Runner (IORunner)
* Instantiate the root object which is containing all tests
* Apply the runner to the root object

This is done in testmain.cpp.

## HOW TO ADD NEW TESTS

For a test suite MyClass, the scripts are generating an include file MyClass_decl.h 

You should create another include Include/MyClass.h and another cpp file Source/MyClass.cpp in TEsting folder.

MyClass.h should contain:

```cpp
 #include "Test.h"
 #include "Pattern.h"
 class MyClass:public Client::Suite
     {
         public:
             MyClass(Testing::testID_t id);
             void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
             void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
         private:
             #include "MyClass_decl.h"
             
             // Definitions of the patterns you have in the test description file
             // for this test suite
             Client::Pattern<float32_t> input1;
             Client::Pattern<float32_t> input2;
             Client::LocalPattern<float32_t> output;
             // Reference patterns are not loaded when we are in dump mode
             Client::RefPattern<float32_t> ref;
     };
```

Then, you should provide an implementation of setUp, tearDown and of course your tests.

So, MyClass.cpp could be:

```cpp
 #include "MyClass.h"
 #include "Error.h"
 
 
     // Implementation of your test
     void MyClass::test_add_f32()
     {
         // Ptr to input patterns, references and output. 
         // Input and references have been loaded in setUp
         const float32_t *inp1=input1.ptr();
         const float32_t *inp2=input2.ptr();
         float32_t *refp=ref.ptr();
         float32_t *outp=output.ptr();
 
         // Execution of the tests
         arm_add_f32(inp1,inp2,outp,input1.nbSamples());
         
 
         // Testing.
         // Warning : in case of benchmarking this will be taken into account in the
         // benchmark. So a benchmark should not contain tests.
         ASSERT_NEAR_EQ(ref,output,(float)1e-6);
 
     } 
```

Warning : in case of a benchmark the xxx.ptr() function calls should be done in the setup function because they have an overhead.

If you use regression formula, this overhead will modify the intercept but the coefficient of highest
degree should not be changed.

Then setUp should load the patterns:

```cpp
 void MyClass::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
     {
       
        Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
 
        // We can load different pattern or length according to the test ID
        switch(id)
        {
         case MyClass::TEST_ADD_F32_1:
           nb = 3;
           ref.reload(MyClass::REF_ADD_F32_ID,mgr,nb);
           break;
         }

       input1.reload(BasicTests::INPUT1_F32_ID,mgr,nb);
       input2.reload(BasicTests::INPUT2_F32_ID,mgr,nb);

       output.create(input1.nbSamples(),BasicTests::OUT_SAMPLES_F32_ID,mgr);
    }
```

In tearDown we have to clean the test. No need to free the buffer since the memory manager will do it
in an automatic way. But if other allocations were done outside of the memory manager, then the clean up should be done here.

It is also here that you specify what you want to dump if you're in dump mode.

```cpp
    void MyClass::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
```
## Benchmarks and database

### Creating and filling the databases

To add a to sqlite3 databse:

    python addToDB.py AGroup

Output.pickle is used by default. It can be changed with -f option.

AGroup should be the class name of a Group in the desc.txt

The suite in this Group should be compatible and have the same parameters.

For instance, we have a BasicBenchmarks group is desc.txt
This group is containing the suites BasicMathsBenchmarksF32, BasicMathsBenchmarksQ31, BasicMathsBenchmarks15 and BasicMathsBenchmarksQ7.

Each suite is defining the same parameters : NB.

If you use:

    python addToDB.py BasicBenchmarks

Output.pickle is used by default. It can be changed with -f option.

A table BasicBenchmarks will be create and the benchmarks result for F32, Q31, Q15 and Q7 will be added to this table.

But, if you do:

    python addToDB.py BasicMathsBenchmarksF32

The a table BasicMathsBenchmarksF32 will be created which is probably not what you want since the table is containing a type column (f32,q31, q15, q7)

The script addToRegDB.py is working on the same principle but using the regression csv to fill a regression database.

To create an empty database you can use  (for default database)

    sqlite3.exe bench.db < createDb.sql 

And for regression database:

    sqlite3.exe reg.db < createDb.sql 

Since the python scripts are using bench.db and reg.db as default names for the databases.

### Processing the database

Database schema (defined in createDb.sql) is creating several columns for the fields which are common to lot of rows like core, compiler, compiler version, datatype etc ...

Like that it is easier to change the name of this additional information and it makes the database smaller.

But then it means that to display the tables in a readable format by the user, some joins are needed.

examples.sql and diff.sql are showing some examples.

examples.sql : how to do simple queries and join with the configuration columns to get a readable format.

diff.sql : How to compute a performance ratio (max cycle and regression) based on a reference core (which could be extended to a reference configuration if needed).

## HOW TO EXTEND IT

## FLOAT16 support

On Arm AC5 compiler \_\_fp16 type (float16_t in CMSIS-DSP) can't be used as argument or return value of a function.

Pointer to \_fp16 arrays are allowed.

In CMSIS-DSP, we want to keep the possibility of having float16_t as an argument.

As consequences, 

* the functions using float16_t in the API won't be supported by AC5 compiler.
* The correspondingfloat16_t tests are put in a different test file desc_f16.txt
* Code for those float16_t test is not built when ac5.cmake toolchain is used
* BasicMath cmake has been modified to show hot to avoid including float16 code
when building with ac5.cmake toolchain

In current example, we assume all float16_t code and tests are not supported by AC5 just to
show how the cmake must be modified.

When more float16_t code is added to the CMSIS-DSP, this will be refined with a better
separation.

