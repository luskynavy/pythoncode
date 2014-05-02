# get time for local and ftp
# http://alexharvey.eu/code/python/get-a-files-last-modified-datetime-using-python/

# ftplib tuto
# http://fr.openclassrooms.com/informatique/cours/utiliser-le-module-ftp-de-python/toutes-les-fonctions-une-a-une

# load a binary file
# http://www.blog.pythonlibrary.org/2012/07/19/python-101-downloading-a-file-with-ftplib/

import os
import time
from ftplib import FTP
from datetime import datetime
 
#ftp = FTP("ftp.free.fr", "", "")
#ftp.login()

ftp = FTP("ftpperso.free.fr", "", "")
path = "/work/python/ftpmirror/gen"

ftp.cwd(path)

etat = ftp.getwelcome() # grace a la fonction getwelcome(), on recupere le "message de bienvenue"
print "Etat : ", etat 


ftp.retrlines("LIST")

# ls = []
# ftp.retrlines('MLSD', ls.append) #not supported on ftp.free.fr and ftpperso.free.fr

remotedir = ftp.nlst()
print "dir: ", remotedir

localdir = os.listdir('.')
print " local", localdir

#get ftp file time
modifiedTime = ftp.sendcmd('MDTM ' + remotedir[2])
dt = datetime.strptime(modifiedTime[4:], "%Y%m%d%H%M%S")
print remotedir[2], " time: ", modifiedTime, dt.strftime("%Y%m%d %H:%M:%S")

 
# download the file
lf = open(remotedir[2], "wb")
ftp.retrbinary("RETR " + remotedir[2], lf.write, 8*1024)
lf.close()
st = time.mktime(dt.timetuple())
#os.utime(remotedir[2], None) # allowed but useless since it is the actual time
#os.utime(remotedir[2], (1330712280, 1330712292)) # (st,st))
try:
    os.utime(remotedir[2], (st,st)) #OSError: [Errno 1] Operation not permitted: on android
except:
    print "os.utime not permitted" 

#upload the file, file will have the utc time of the upload
fichier = "Fichier"
file = open(fichier, 'rb') # ici, j'ouvre le fichier
ftp.storbinary('STOR ' + fichier, file) # ici (ou ftp est encore la variable de la connexion), j'indique le fichier a envoyer
file.close() # on ferme le fichier


#get local file time
firstFileTime = os.path.getmtime("totouch.txt")
print "totouch.txt time: ", firstFileTime, datetime.fromtimestamp(firstFileTime).strftime("%Y%m%d %H:%M:%S") #("%Y%m%d %H:%M:%S")

#if ftp file is newer than local
if datetime.strptime(modifiedTime[4:], "%Y%m%d%H%M%S") > datetime.fromtimestamp(firstFileTime):
    print ">"
else:
    print "<="

#update the local time file
print "touch"
os.utime('totouch.txt', None)

#get the new local time and utc time
touchTime = os.path.getmtime("totouch.txt")
print "totouch.txt time: ", touchTime, datetime.fromtimestamp(touchTime).strftime("%Y%m%d %H:%M:%S") #("%Y%m%d %H:%M:%S")
print "totouch.txt utc time: ", touchTime, datetime.utcfromtimestamp(touchTime).strftime("%Y%m%d %H:%M:%S") #("%Y%m%d %H:%M:%S")


if datetime.fromtimestamp(touchTime) > datetime.fromtimestamp(firstFileTime):
    print ">"
else:
    print "<="

ftp.rename(path + "/Fichier", path + "/Fichier2")



listing = []
ftp.retrlines("LIST", listing.append)
words = listing[0].split(None, 8)
filename = words[-1].lstrip()
#print words

ftp.quit()