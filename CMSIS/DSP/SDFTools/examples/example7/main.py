import sched as s 
import signal, os


if __name__ == '__main__':

   nb=0 
   error=0

   try:
      nb,error = s.scheduler()
   except Exception as inst:
      print(inst)






