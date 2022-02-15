# Sources
https://docs.getwoke.tech/
https://github.com/get-woke/woke.git

# Instructions
1. Install woke to somewhere in your path, for example:
   ```
   sudo curl -sSfL https://git.io/getwoke |   bash -s -- -b /home/<USER>/bin/
   ```
2. Make sure woke is in your path. This cmd should give you the installation
   path:
   ```
   which woke
   ```
3. Make sure you in the root of the CMSIS repo.
4. Run
   ```
   woke -c woke.cmsis.yml ./CMSIS/NN/
   ```
5. The result will show up in your terminal.