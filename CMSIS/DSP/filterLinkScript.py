import sys
import re 

filePath = sys.argv[1]

text = None 

with open(filePath, "r") as f:
   text = f.readlines() 

if text:
    with open(filePath, "w") as f:
        for l in text:
            if not re.match(r'^#.*$',l) and not re.match(r'^[ ]+[0-9].*$',l):
                f.write(l)
                #print(l.rstrip())

