import re

parseRe = re.compile('(.*)\s+([0-9]+):([0-9a-f]+):(.*)')

dbgCnt=0

clk0=0
clk1=0

def getCycles(t):
    global dbgCnt
    global clk0
    global clk1
    while(True):
      try:
        line = next(t)
        if line:
          m = parseRe.match(line)
          if m:
             if (('OP_HINT_DBG_32' in line) or ('DBG' in line)):
                 curClk = int(m.group(2))
                 if dbgCnt==0:
                  clk0 =curClk
                 if dbgCnt == 1:
                  clk1 = curClk
                 dbgCnt += 1
                 if dbgCnt == 2:
                    dbgCnt = 0
                    return(clk1 - clk0)
      except StopIteration:
        dbgCnt = 0
        return(0)



