echo "Basic Maths"
python summaryBench.py -f bench.txt  -r build_m7\result.txt BasicBenchmarks
echo "Complex Maths"
python summaryBench.py -f bench.txt  -r build_m7\result.txt ComplexBenchmarks
echo "FIR"
python summaryBench.py -f bench.txt  -r build_m7\result.txt FIR
echo "Convolution / Correlation"
python summaryBench.py -f bench.txt  -r build_m7\result.txt MISC
echo "Decimation / Interpolation"
python summaryBench.py -f bench.txt  -r build_m7\result.txt DECIM
echo "BiQuad"
python summaryBench.py -f bench.txt  -r build_m7\result.txt BIQUAD
echo "Controller"
python summaryBench.py -f bench.txt  -r build_m7\result.txt Controller
echo "Fast Math"
python summaryBench.py -f bench.txt  -r build_m7\result.txt FastMath
echo "Barycenter"
python summaryBench.py -f bench.txt  -r build_m7\result.txt SupportBarF32
echo "Support"
python summaryBench.py -f bench.txt  -r build_m7\result.txt Support
echo "Unary Matrix"
python summaryBench.py -f bench.txt  -r build_m7\result.txt Unary 
echo "Binary Matrix"
python summaryBench.py -f bench.txt  -r build_m7\result.txt Binary
echo "Transform"
python summaryBench.py -f bench.txt  -r build_m7\result.txt Transform