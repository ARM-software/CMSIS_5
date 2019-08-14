echo "Basic Maths"
python summaryBench.py -f bench.txt  BasicBenchmarks
echo "Complex Maths"
python summaryBench.py -f bench.txt  ComplexBenchmarks
echo "FIR"
python summaryBench.py -f bench.txt  FIR
echo "Convolution / Correlation"
python summaryBench.py -f bench.txt  MISC
echo "Decimation / Interpolation"
python summaryBench.py -f bench.txt  DECIM
echo "BiQuad"
python summaryBench.py -f bench.txt  BIQUAD
echo "Controller"
python summaryBench.py -f bench.txt  Controller
echo "Fast Math"
python summaryBench.py -f bench.txt  FastMath
echo "Barycenter"
python summaryBench.py -f bench.txt  SupportBarF32
echo "Support"
python summaryBench.py -f bench.txt  Support
echo "Unary Matrix"
python summaryBench.py -f bench.txt  Unary 
echo "Binary Matrix"
python summaryBench.py -f bench.txt  Binary