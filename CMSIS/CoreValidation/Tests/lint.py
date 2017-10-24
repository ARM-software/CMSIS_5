import sys
import os.path
from argparse import ArgumentParser

sys.path.append('buildutils') 

from rtecmd import RteCmd 

def main(argv):
  parser = ArgumentParser()
  parser.add_argument('-d', '--device', required=True, help = 'Device to be considered.')
  parser.add_argument('-c', '--compiler', required=True, help = 'Compiler to be considered.')
  parser.add_argument('-t', '--target', nargs='?', default="default", help = 'Target to be considered.')
  args = parser.parse_args()
  
  rtebuild = os.path.join(args.device, args.compiler, "default.rtebuild")
  
  if not os.path.isfile(rtebuild):
    raise IOError("rtebuild project not available:'"+rtebuild+"'")
    
  cmd = RteCmd(rtebuild, args.target, "lint")
  cmd.run()
  
if __name__ == "__main__":
  try:
    main(sys.argv[1:])
  except Exception as e:
    print(e)
