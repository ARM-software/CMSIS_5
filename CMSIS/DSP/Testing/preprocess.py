import argparse
import TestScripts.NewParser as parse
import pickle

parser = argparse.ArgumentParser(description='Parse test description')

parser.add_argument('-f', nargs='?',type = str, default=None, help="Test description file path")

parser.add_argument('-o', nargs='?',type = str, default="Output.pickle", help="output file for parsed description")

args = parser.parse_args()

if args.f is not None:
    p = parse.Parser()
    # Parse the test description file
    root = p.parse(args.f)
    with open(args.o,"wb") as output:
         pickle.dump(root, output)
