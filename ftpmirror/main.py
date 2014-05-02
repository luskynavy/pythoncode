import os
from ftplib import FTP
from datetime import datetime

localPath = "gen"
remotePath = "/syncback/gen"

#localPath = "D:\Data\scummvm-1.6.0-win32\Saves"
#remotePath = "/syncback/scummvm saves"

localPath = "D:\Data\psp\ppsspp_win 9.8\memstick\PSP\SAVEDATA\ULES01431GAMEDATA"
remotePath = "/syncback/PPSSPP SAVEDATA/ULES01431GAMEDATA"


ftp = FTP("ftpperso.free.fr", "", "")

ftp.cwd(remotePath)

remoteDir = ftp.nlst()
print "remote: ", remoteDir

localDir = os.listdir(localPath)
print " local", localDir

print

#check if files must be uploaded or downloaded
for localFile in localDir:
    try:
        remoteIndex = remoteDir.index(localFile)
    except:
        remoteIndex = -1

    if remoteIndex != -1:
        #print localFile, "is present on remote"
        
        #get ftp file time
        remoteTime = ftp.sendcmd('MDTM ' + remoteDir[remoteIndex])
        #print remoteTime
        remoteTime = datetime.strptime(remoteTime[4:], "%Y%m%d%H%M%S")
        localTime = os.path.getmtime(os.path.join(localPath, localFile))
        localTime = datetime.utcfromtimestamp(localTime) #syncback seems to work in utc
        #localTime = datetime.fromtimestamp(localTime)
        
        if localTime > remoteTime:
            print "local is newer than remote :", localTime, remoteTime # must upload local then touch local file to synchronize time
        elif localTime == remoteTime:
            print "local is same than remote :", localTime, remoteTime #nothing to do
        else:
            print "local is older than remote :", localTime, remoteTime #must download remote then try to set local time with remote time 
    else:
        print localFile, "is not present on remote" #must upload local


#check if new remote files must be downloaded
for remoteFile in remoteDir:
    #skip . and ..
    if remoteFile != '.' and remoteFile != '..':
        try:
            remoteIndex = localDir.index(remoteFile)
        except:
            #error file not found me be downloaded 
            print remoteFile, " is only present on remote"

ftp.quit()