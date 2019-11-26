from collections import deque
import TestScripts.Parser

# When deprecation is forced on some nodes
# we ensure that a parent of a valid node is also valid
def correctDeprecation(node):
    current = node.data["deprecated"] 
    for c in node.children:
        if not correctDeprecation(c):
            current = False 
    node.data["deprecated"] = current
    return(current)

def inheritDeprecation(node,deprecated):
    current = node.data["deprecated"] or deprecated
    node.data["deprecated"] = current
    if node.kind != TestScripts.Parser.TreeElem.TEST:
      for c in node.children:
        inheritDeprecation(c,current)
        

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
    inheritDeprecation(root,False)
    if others:
       deprecateRec(root,deque(others),True)
       correctDeprecation(root)