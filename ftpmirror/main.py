from kivy.uix.listview import ListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

import os
from ftplib import FTP
from datetime import datetime

localPath = "gen"
remotePath = "/syncback/gen"

localPath = "D:\Data\scummvm-1.6.0-win32\Saves"
remotePath = "/syncback/scummvm saves"

#localPath = "D:\Data\psp\ppsspp_win 9.8\memstick\PSP\SAVEDATA\ULES01431GAMEDATA"
#remotePath = "/syncback/PPSSPP SAVEDATA/ULES01431GAMEDATA"

class FTPView(GridLayout):
    def __init__(self, **kwargs):        
        list = []

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
                    list.append(localFile + "  is newer than remote :" + str(localTime) + str(remoteTime))
                elif localTime == remoteTime:
                    print "local is same than remote :", localTime, remoteTime #nothing to do
                    list.append(localFile + "  is same than remote :" + str(localTime) + str(remoteTime))
                else:
                    print "local is older than remote :", localTime, remoteTime #must download remote then try to set local time with remote time 
                    list.append(localFile + "  is older than remote :" + str(localTime) + str(remoteTime))
            else:
                print localFile, " is not present on remote" #must upload local
                list.append(str(localFile) + "is not present on remote")


        #check if new remote files must be downloaded
        for remoteFile in remoteDir:
            #skip . and ..
            if remoteFile != '.' and remoteFile != '..':
                try:
                    remoteIndex = localDir.index(remoteFile)
                except:
                    #error file not found me be downloaded 
                    print remoteFile, " is only present on remote"
                    list.append(str(remoteFile) + " is only present on remote")

        ftp.quit()
        
        kwargs['cols'] = 1
        super(FTPView, self).__init__(**kwargs)
        
        list_view = ListView(item_strings = list)        
        
        print ["a", "b"]
        print list
        
        buttonDoIt = Button(text='Do it', size_hint_y = .1)
        buttonDoIt.bind(on_release=self.DoIt)
        
        buttonCancel = Button(text='Cancel', size_hint_y = .1)#, center_x = 150)
        buttonCancel.bind(on_release=self.Cancel)

        self.add_widget(list_view)
        self.add_widget(buttonDoIt)
        self.add_widget(buttonCancel)
        
    def DoIt(self, *l):
        print "DoIt"
        
    def Cancel(self, *l):
        print "Cancel"
    
        

if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(FTPView(width=800))
