@ECHO OFF 

echo "Basic Maths"
python processTests.py -f bench.txt -e BasicBenchmarks
call:runBench

echo "Complex Maths"
python processTests.py -f bench.txt -e ComplexBenchmarks
call:runBench

echo "FIR"
python processTests.py -f bench.txt -e FIR
call:runBench

echo "Convolution / Correlation"
python processTests.py -f bench.txt -e MISC
call:runBench

echo "Decimation / Interpolation"
python processTests.py -f bench.txt -e DECIM
call:runBench

echo "BiQuad"
python processTests.py -f bench.txt -e BIQUAD
call:runBench

echo "Controller"
python processTests.py -f bench.txt -e Controller
call:runBench

echo "Fast Math"
python processTests.py -f bench.txt -e FastMath
call:runBench

echo "Barycenter"
python processTests.py -f bench.txt -e SupportBarF32
call:runBench

echo "Support"
python processTests.py -f bench.txt -e Support
call:runBench

echo "Unary Matrix"
python processTests.py -f bench.txt -e Unary 
call:runBench

echo "Binary Matrix"
python processTests.py -f bench.txt -e Binary
call:runBench

echo "Transform"
python processTests.py -f bench.txt -e Transform
call:runBench

EXIT /B

:runBench
REM pushd build_m7
REM pushd build_m0
pushd build_a5
make
REM "C:\Program Files\ARM\Development Studio 2019.0\sw\models\bin\FVP_MPS2_Cortex-M7.exe" -a Testing > result.txt
REM "C:\Program Files\ARM\Development Studio 2019.0\sw\models\bin\FVP_MPS2_Cortex-M0.exe" -a Testing > result.txt
"C:\Program Files\ARM\Development Studio 2019.0\sw\models\bin\FVP_VE_Cortex-A5x1.exe" -a Testing  > result.txt
popd
echo "Parse result"
REM python processResult.py -f bench.txt -e -r build_m7\result.txt
REM python processResult.py -f bench.txt -e -r build_m0\result.txt
python processResult.py -f bench.txt -e -r build_a5\result.txt
goto:eof