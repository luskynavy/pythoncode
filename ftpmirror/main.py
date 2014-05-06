from kivy.uix.listview import ListView, CompositeListItem
from kivy.adapters.dictadapter import DictAdapter
from kivy.uix.listview import ListItemButton, ListItemLabel
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

import os
from ftplib import FTP
from datetime import datetime

localPath = "gen"
remotePath = "/syncback/gen"

#localPath = "D:\Data\scummvm-1.6.0-win32\Saves"
#remotePath = "/syncback/scummvm saves"

#localPath = "D:\Data\psp\ppsspp_win 9.8\memstick\PSP\SAVEDATA\ULES01431GAMEDATA"
#remotePath = "/syncback/PPSSPP SAVEDATA/ULES01431GAMEDATA"

class FTPView(GridLayout):
    def __init__(self, **kwargs):        
        listftp = []
        
        file = open("ftp", "r")
        user = file.readline()
        mdp = file.readline()
        file.close()

        ftp = FTP("ftpperso.free.fr", user, mdp)

        ftp.cwd(remotePath)

        remoteDir = ftp.nlst()
        print "remote: ", remoteDir

        localDir = os.listdir(localPath)
        print " local", localDir

        print

        #check if files must be uploaded or downloaded
        for localFile in localDir:
            localTime = os.path.getmtime(os.path.join(localPath, localFile))
            localTime = datetime.utcfromtimestamp(localTime) #syncback seems to work in utc
            #localTime = datetime.fromtimestamp(localTime)
                
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
                
                if localTime > remoteTime:
                    print "local is newer than remote :", localTime, remoteTime # must upload local then touch local file to synchronize time
                    listftp.append([str(localTime), ">", str(remoteTime), localFile])
                elif localTime == remoteTime:
                    print "local is same than remote :", localTime, remoteTime #nothing to do
                    listftp.append([str(localTime), "=", str(remoteTime), localFile])
                else:
                    print "local is older than remote :", localTime, remoteTime #must download remote then try to set local time with remote time 
                    listftp.append([str(localTime), "<", str(remoteTime), localFile])
            else:
                print localFile, " is not present on remote" #must upload local
                listftp.append([str(localTime)[:19], " ", "", localFile])


        #check if new remote files must be downloaded
        for remoteFile in remoteDir:
            #skip . and ..
            if remoteFile != '.' and remoteFile != '..':
                try:
                    remoteIndex = localDir.index(remoteFile)
                except:
                    #error file not found me be downloaded 
                    print remoteFile, " is only present on remote"
                    listftp.append(["", " ", str(remoteTime), remoteFile])

        ftp.quit()
        
        kwargs['cols'] = 1
        super(FTPView, self).__init__(**kwargs)
        
        args_converter = \
            lambda row_index, rec: \
                {'text': rec['text'],
                 #'size_hint_x': .3,
                 #'width' : 800,
                 'height': 25,
                 'cls_dicts': [{'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][3]}},
                       {'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][0], 'size_hint_x':.3}},
                       {'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][1], 'size_hint_x':.02}},
                       {'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][2], 'size_hint_x':.3}}]}

        item_strings = [listftp[index][2] for index in range(0, len(listftp))]
        integers_dict =         { listftp[i][2]: {'text': listftp[i], 'is_selected': False} for i in range(0, len(listftp))}


        dict_adapter = DictAdapter(#sorted_keys=item_strings,
                                   data= integers_dict,
                                   args_converter=args_converter,
                                   selection_mode='single',
                                   allow_empty_selection=False,
                                   cls=CompositeListItem)

        # Use the adapter in our ListView:
        list_view = ListView(adapter=dict_adapter)
       
        #list_view = ListView(item_strings = list)
       
        
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
    runTouchApp(FTPView(width=1280))
