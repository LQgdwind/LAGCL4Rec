import os.path
from time import strftime,localtime,time

class Log(object):
    def __init__(self,logName,logDir,isDebug=False):
        self.name = logName
        self.dir = logDir
        self.filePath = logDir+strftime('%Y-%m-%d %H-%M-%S',localtime(time()))+'.txt'
        self.debug = isDebug

    def log(self,content,fl=True,typ='info'):
        if self.debug==False and typ=='debug':
            pass
        else:
            cont = ""
            if fl:
                cont = strftime("%Y-%m-%d %H-%M-%S", localtime(time())) + ": " + str(content)
            else:
                cont = content
            print(cont)
            with open(self.filePath,'a') as f:
                f.write(cont+'\n')

    def add(self,content):
        self.log(content)
