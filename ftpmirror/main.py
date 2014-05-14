from kivy.uix.listview import ListView, CompositeListItem
from kivy.adapters.dictadapter import DictAdapter
from kivy.uix.listview import ListItemButton, ListItemLabel
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView

import os
from ftplib import FTP
from datetime import datetime

#localPath = "gen"
#remotePath = "/syncback/gen"

#localPath = "D:\Data\scummvm-1.6.0-win32\Saves"
#remotePath = "/syncback/scummvm saves"

#localPath = "D:\Data\psp\ppsspp_win 9.8\memstick\PSP\SAVEDATA\ULES01431GAMEDATA"
#remotePath = "/syncback/PPSSPP SAVEDATA/ULES01431GAMEDATA"

class FTPView(GridLayout):
    def __init__(self, **kwargs):
        self.listftp = []
        
        self.first_time = 1        
        
        kwargs['cols'] = 1
        super(FTPView, self).__init__(**kwargs)
        
        '''self.localPath = "gen"
        self.remotePath = "/syncback/gen"
        self.ScanFTP()
        self.BuildList()'''
        
        self.Load()
    
    def BuildList(self):
        args_converter = \
            lambda row_index, rec: \
                {'text': rec['text'],
                 'height': 25,                 
                 'size_hint_y': None,
                 'cls_dicts': [{'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][3], 'height': 25,}},
                       {'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][0], 'size_hint_x':.3}},
                       {'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][1], 'size_hint_x':.02}},
                       {'cls': ListItemLabel,
                        'kwargs': {'text': rec['text'][2], 'size_hint_x':.3}}]}

        item_strings = [self.listftp[index][3] for index in xrange(len(self.listftp))]
        integers_dict = { self.listftp[i][3]: {'text': self.listftp[i], 'is_selected': False} for i in xrange(len(self.listftp))}


        dict_adapter = DictAdapter(#sorted_keys=item_strings,
                                   data= integers_dict,
                                   args_converter=args_converter,
                                   selection_mode='single',
                                   allow_empty_selection=False,
                                   cls=CompositeListItem)

        # Use the adapter in our ListView:
        self.list_view = ListView(adapter=dict_adapter)
        print "len", len(self.listftp), self.listftp
       
        #self.list_view = ListView(item_strings = self.listftp)
       
        self.buttonLoad = Button(text='Load', size_hint_y = .1)
        self.buttonLoad.bind(on_release = self.Load)
        
        self.buttonDoIt = Button(text = 'Do it', size_hint_y = .1)
        self.buttonDoIt.bind(on_release = self.DoIt)
        
        self.buttonCancel = Button(text = 'Cancel', size_hint_y = .1)#, center_x = 150)
        self.buttonCancel.bind(on_release = self.Cancel)

        self.add_widget(self.buttonLoad)
        self.add_widget(self.list_view)
        self.add_widget(self.buttonDoIt)
        self.add_widget(self.buttonCancel)
        
    def ScanFTP(self):
        self.listftp = []
        self.listftp.append(["LOCAL", " ", "FTP", " "])
        
        file = open("ftp", "r")
        user = file.readline()
        mdp = file.readline()
        file.close()

        ftp = FTP("ftpperso.free.fr", user, mdp)

        ftp.cwd(self.remotePath)

        remoteDir = ftp.nlst()
        print "remote: ", remoteDir

        localDir = os.listdir(self.localPath)
        print " local", localDir

        print

        #check if files must be uploaded or downloaded
        for localFile in localDir:
            localTime = os.path.getmtime(os.path.join(self.localPath, localFile))
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
                    print "local ", localFile, " is newer than remote :", localTime, remoteTime # must upload local then touch local file to synchronize time
                    self.listftp.append([str(localTime), ">", str(remoteTime), localFile])
                elif localTime == remoteTime:
                    print "local ", localFile, " is same than remote :", localTime, remoteTime #nothing to do
                    self.listftp.append([str(localTime), "=", str(remoteTime), localFile])
                else:
                    print "local ", localFile, " is older than remote :", localTime, remoteTime #must download remote then try to set local time with remote time 
                    self.listftp.append([str(localTime), "<", str(remoteTime), localFile])
            else:
                print localFile, " is not present on remote" #must upload local
                self.listftp.append([str(localTime)[:19], " ", "", localFile])


        #check if new remote files must be downloaded
        for remoteFile in remoteDir:
            #skip . and ..
            if remoteFile != '.' and remoteFile != '..':
                try:
                    remoteIndex = localDir.index(remoteFile)
                except:
                    #error file not found me be downloaded 
                    print remoteFile, " is only present on remote"
                    self.listftp.append(["", " ", str(remoteTime), remoteFile])

        ftp.quit()
        
    def Load(self, *l):
        print "Load"
        
        self.fl = FileChooserListView(path = ".", rootpath = ".", filters = ["*.ftp"],
                 dirselect=False)
        self.fl.bind(selection = self.on_selected)
        self.add_widget(self.fl)
    
    def on_selected(self, filechooser, selection):
        print 'on_selected', selection
        self.remove_widget(self.fl)
        
        if self.first_time != 1:
            self.remove_widget(self.buttonLoad)
            self.remove_widget(self.list_view)
            self.remove_widget(self.buttonDoIt)
            self.remove_widget(self.buttonCancel)
            
        self.first_time = 0
            
        fileRead = open(selection[0], "r")
        self.localPath = fileRead.readline().rstrip('\n')
        self.remotePath = fileRead.readline().rstrip('\n')
        fileRead.close()
        self.ScanFTP()
        
        self.BuildList()
        
    def DoIt(self, *l):
        print "DoIt"
        
    def Cancel(self, *l):
        print "Cancel"
        self.remove_widget(self.buttonDoIt)
        self.add_widget(self.buttonDoIt)
    
        

if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(FTPView(width = 1280))
