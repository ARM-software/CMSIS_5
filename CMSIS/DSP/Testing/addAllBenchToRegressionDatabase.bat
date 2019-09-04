echo "Basic Maths"
python addToRegDB.py -f bench.txt  BasicBenchmarks
echo "Complex Maths"
python addToRegDB.py -f bench.txt  ComplexBenchmarks
echo "FIR"
python addToRegDB.py -f bench.txt  FIR
echo "Convolution / Correlation"
python addToRegDB.py -f bench.txt  MISC
echo "Decimation / Interpolation"
python addToRegDB.py -f bench.txt  DECIM
echo "BiQuad"
python addToRegDB.py -f bench.txt  BIQUAD
echo "Controller"
python addToRegDB.py -f bench.txt  Controller
echo "Fast Math"
python addToRegDB.py -f bench.txt  FastMath
echo "Barycenter"
python addToRegDB.py -f bench.txt  SupportBarF32
echo "Support"
python addToRegDB.py -f bench.txt  Support
echo "Unary Matrix"
python addToRegDB.py -f bench.txt  Unary 
echo "Binary Matrix"
python addToRegDB.py -f bench.txt  Binary
echo "Transform"
python addToRegDB.py -f bench.txt  Transform