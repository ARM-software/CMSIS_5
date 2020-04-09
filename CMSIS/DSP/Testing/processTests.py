import argparse
import TestScripts.NewParser as parse
import TestScripts.CodeGen
import TestScripts.Deprecate as d


parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-f', nargs='?',type = str, default="Output.pickle", help="File path")

parser.add_argument('-p', nargs='?',type = str, default="Patterns", help="Pattern dir path")
parser.add_argument('-d', nargs='?',type = str, default="Parameters", help="Parameter dir path")

# -e true when no semihosting
# Input is include files
# Output is only one stdout
# So the .h for include files need to be generated.
parser.add_argument('-e', action='store_true', help="Embedded test")

parser.add_argument('others', nargs=argparse.REMAINDER)

args = parser.parse_args()


if args.f is not None:
    # Create a treeelemt object
    #p = parse.Parser()
    # Create a codegen object
    c = TestScripts.CodeGen.CodeGen(args.p,args.d, args.e)
    # Parse the test description.
    #root = p.parse(args.f)
    root=parse.loadRoot(args.f)
    d.deprecate(root,args.others)
    #print(root)
    # Generate code with the tree of tests
    c.genCodeForTree(root)
else:
    parser.print_help()