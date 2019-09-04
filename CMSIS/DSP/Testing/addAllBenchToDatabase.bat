echo "Basic Maths"
python addToDB.py -f bench.txt  BasicBenchmarks
echo "Complex Maths"
python addToDB.py -f bench.txt  ComplexBenchmarks
echo "FIR"
python addToDB.py -f bench.txt  FIR
echo "Convolution / Correlation"
python addToDB.py -f bench.txt  MISC
echo "Decimation / Interpolation"
python addToDB.py -f bench.txt  DECIM
echo "BiQuad"
python addToDB.py -f bench.txt  BIQUAD
echo "Controller"
python addToDB.py -f bench.txt  Controller
echo "Fast Math"
python addToDB.py -f bench.txt  FastMath
echo "Barycenter"
python addToDB.py -f bench.txt  SupportBarF32
echo "Support"
python addToDB.py -f bench.txt  Support
echo "Unary Matrix"
python addToDB.py -f bench.txt  Unary 
echo "Binary Matrix"
python addToDB.py -f bench.txt  Binary
echo "Transform"
python addToDB.py -f bench.txt  Transform