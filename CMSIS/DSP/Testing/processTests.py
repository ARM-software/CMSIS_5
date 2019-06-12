import argparse
import TestScripts.NewParser as parse
import TestScripts.CodeGen
from collections import deque

# When deprecation is forced on some nodes
# we ensure that a parent of a valid node is also valid
def correctDeprecation(node):
    current = node.data["deprecated"] 
    for c in node.children:
        if not correctDeprecation(c):
            current = False 
    node.data["deprecated"] = current
    return(current)

def deprecateRec(root,others,deprecated):
    if others:
        newOthers=others.copy()
        newOthers.popleft()
        if root.kind == TestScripts.Parser.TreeElem.TEST:
            if others[0].isdigit() and int(root.id) == int(others[0]):
               root.data["deprecated"]=False
               for c in root.children:
                  deprecateRec(c,newOthers,False)
            else:
               root.data["deprecated"]=True
               for c in root.children:
                  deprecateRec(c,others,deprecated)
        else:
           if root.data["class"] == others[0]:
             root.data["deprecated"]=False
             for c in root.children:
                 deprecateRec(c,newOthers,False)
           else:
             root.data["deprecated"]=deprecated
             for c in root.children:
                 deprecateRec(c,others,deprecated)

def deprecate(root,others):
    if others:
       deprecateRec(root,deque(others),True)
       correctDeprecation(root)

parser = argparse.ArgumentParser(description='Parse test description')
parser.add_argument('-f', nargs='?',type = str, default="test.txt", help="File path")

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
    p = parse.Parser()
    # Create a codegen object
    c = TestScripts.CodeGen.CodeGen(args.p,args.d, args.e)
    # Parse the test description.
    root = p.parse(args.f)
    deprecate(root,args.others)
    print(root)
    # Generate code with the tree of tests
    c.genCodeForTree(root)
else:
    parser.print_help()