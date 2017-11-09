
import sys
import os
import hashlib
import shutil
from glob import iglob
from argparse import ArgumentParser

def main():
  parser = ArgumentParser()
  parser.add_argument("-n", "--dry-run", action="store_true")
  parser.add_argument("old")
  parser.add_argument("new", nargs="+")
  args = parser.parse_args()
  
  for old in iglob(args.old, recursive=True):
    for new in args.new:
      dest = os.path.join(os.path.dirname(old), os.path.basename(new))
      print("Updating {0}".format(dest))
      if not args.dry_run:
        if os.path.exists(dest):
          os.remove(dest)
        shutil.copy2(new, dest)

if __name__ == "__main__":
  main()