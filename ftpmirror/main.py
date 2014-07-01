from kivy.uix.listview import ListView, CompositeListItem
from kivy.adapters.dictadapter import DictAdapter
from kivy.uix.listview import ListItemButton, ListItemLabel
from kivy.uix.gridlayout import GridLayout
#from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView

from kivy.uix.popup import Popup #for error message box
from kivy.uix.label import Label

import os
from ftplib import FTP
import time
from datetime import datetime
#from datetime import timedelta
#from math import Math

#localPath = "gen"
#remotePath = "/syncback/gen"

#localPath = "D:\Data\scummvm-1.6.0-win32\Saves"
#remotePath = "/syncback/scummvm saves"

#localPath = "D:\Data\psp\ppsspp_win 9.8\memstick\PSP\SAVEDATA\ULES01431GAMEDATA"
#remotePath = "/syncback/PPSSPP SAVEDATA/ULES01431GAMEDATA"

class FTPView(GridLayout):
#class FTPView(BoxLayout):
    def __init__(self, **kwargs):
        self.listftp = []
        
        self.first_time = 1        
        
        #1 column for the global grid layout
        kwargs['cols'] = 1
        super(FTPView, self).__init__(**kwargs)
    
        #show the load button
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

        #item_strings = [self.listftp[index][3] for index in xrange(len(self.listftp))]
        integers_dict = { self.listftp[i][3]: {'text': self.listftp[i], 'is_selected': False} for i in xrange(len(self.listftp))}


        dict_adapter = DictAdapter(#sorted_keys=item_strings,
                                   data= integers_dict,
                                   args_converter=args_converter,
                                   selection_mode='single',
                                   allow_empty_selection=False,
                                   cls=CompositeListItem)

        # Use the adapter in our ListView:
        self.list_view = ListView(adapter=dict_adapter)
        #print "len", len(self.listftp), self.listftp
       
        #self.list_view = ListView(item_strings = self.listftp)
       
        self.buttonLoad = Button(text='Load', size_hint_y = .1)
        self.buttonLoad.bind(on_release = self.Load)
        
        self.buttonDoIt = Button(text = 'Do it', size_hint_y = .1)
        self.buttonDoIt.bind(on_release = self.DoIt)
        
        #self.buttonCancel = Button(text = 'Cancel', size_hint_y = .1)#, center_x = 150)
        #self.buttonCancel.bind(on_release = self.Cancel)

        #add the widgets
        self.add_widget(self.buttonLoad)
        self.add_widget(self.list_view)
        self.add_widget(self.buttonDoIt)
        #self.add_widget(self.buttonCancel)
        
    def ScanFTP(self):
        try:
            self.listftp = []
            #set the title
            self.listftp.append(["LOCAL", " ", "FTP", " "])
            
            #read the user and password in a file
            fileaccount = open("ftp", "r")
            user = fileaccount.readline()
            mdp = fileaccount.readline()
            fileaccount.close()

            ftp = FTP("ftpperso.free.fr", user, mdp)

            ftp.cwd(self.remotePath)

            #get remote files list
            remoteDir = ftp.nlst()
            #print "remote: ", remoteDir

            #get local files list
            localDir = os.listdir(self.localPath)
            #print " local", localDir

            print

            #check local files in remote
            for localFile in localDir:
                localTime = os.path.getmtime(os.path.join(self.localPath, localFile))
                #localTime = datetime.utcfromtimestamp(localTime) #syncback seems to work in utc #fix utc method 1
                localTime = datetime.fromtimestamp(localTime) #fix utc method 2
                    
                try:
                    remoteIndex = remoteDir.index(localFile)
                except:
                    remoteIndex = -1

                if remoteIndex != -1:
                    #print localFile, "is present on remote"
                    
                    #get ftp file time
                    remoteTime = ftp.sendcmd('MDTM ' + remoteDir[remoteIndex])
                    #print remoteTime
                    remoteTime = datetime.strptime(remoteTime[4:], "%Y%m%d%H%M%S") #fix utc method 1 : nothing    
                    remoteTime = remoteTime + (datetime.now() - datetime.utcnow()) #fix utc method 2            
                    
                    #check time diff
                    if abs((localTime - remoteTime).total_seconds()) < 10: #delta 10 seconds
                    #if localTime == remoteTime:
                        #check size diff to test if copy was interrupted : "!!" if error in copy, "=" if size are same too 
                        #get distant file size
                        remoteSize = ftp.sendcmd('SIZE ' + remoteDir[remoteIndex])
                        #print "remote size :", int(remoteSize[4:])
                        
                        localSize = os.path.getsize(os.path.join(self.localPath, localFile))
                        #print "local size :", localSize
                                            
                        #print "local ", localFile, " is same than remote :", localTime, remoteTime #nothing to do
                        if int(remoteSize[4:]) != localSize:
                            self.listftp.append([str(localTime)[:19], "!!", str(remoteTime)[:19], localFile])                    
                        else:
                            self.listftp.append([str(localTime)[:19], "=", str(remoteTime)[:19], localFile])
                    elif localTime > remoteTime:
                        #print "local ", localFile, " is newer than remote :", localTime, remoteTime, localTime - remoteTime # must upload local then touch local file to synchronize time
                        self.listftp.append([str(localTime)[:19], ">", str(remoteTime)[:19], localFile])
                    else:
                        #print "local ", localFile, " is older than remote :", localTime, remoteTime #must download remote then try to set local time with remote time 
                        self.listftp.append([str(localTime)[:19], "<", str(remoteTime)[:19], localFile])
                else:
                    #print localFile, " is not present on remote" #must upload local
                    self.listftp.append([str(localTime)[:19], "", "", localFile])


            #check if new remote files must be downloaded (not present at all in local)
            for remoteFile in remoteDir:
                #skip . and ..
                if remoteFile != '.' and remoteFile != '..':
                    try:
                        remoteIndex = localDir.index(remoteFile)
                    except:
                        #error file not found me be downloaded 
                        print remoteFile, " is only present on remote"
                        self.listftp.append(["", "", str(remoteTime)[:19], remoteFile])

            #close the connection
            ftp.quit()
        except Exception as e:
            #show modal message box if error
            content = Button(text=str(e))
            popup = Popup(title='Scanning error', content=content, size_hint=(1, 1))
            content.bind(on_press=popup.dismiss)
            popup.open()            
        
    def Load(self, *l):
        print "Load"
        
        self.fl = FileChooserIconView(path = ".", rootpath = ".", filters = ["*.ftp"],
                 dirselect=False)#, size_hint_y = None)
        #self.fl.height = 500
        self.fl.bind(selection = self.on_selected)
        self.add_widget(self.fl)
    
    def on_selected(self, filechooser, selection):
        print 'on_selected', selection
        self.remove_widget(self.fl)
        
        if self.first_time != 1:
            self.remove_widget(self.buttonLoad)
            self.remove_widget(self.list_view)
            self.remove_widget(self.buttonDoIt)
            #self.remove_widget(self.buttonCancel)
            
        self.first_time = 0
            
        fileRead = open(selection[0], "r")
        self.localPath = fileRead.readline().rstrip('\n').rstrip('\r')
        self.remotePath = fileRead.readline().rstrip('\n').rstrip('\r')
        fileRead.close()
        self.ScanFTP()
        
        self.BuildList()
        
    def DoIt(self, *l):
        print "DoIt"
        
        #read the user and password in a file
        fileaccount = open("ftp", "r")
        user = fileaccount.readline()
        mdp = fileaccount.readline()
        fileaccount.close()

        #connect
        ftp = FTP("ftpperso.free.fr", user, mdp)

        ftp.cwd(self.remotePath)
        
        for current_file in self.listftp:
            if current_file[1] == ">":
                print "must upload ", current_file[3]
                self.upload(ftp, current_file[3])
            elif current_file[1] == "<":
                print "must download ", current_file[3], "and set time ", current_file[2]  
                self.download(ftp, current_file[3], current_file[2])            
            elif current_file[1] == "" and current_file[0]:
                print "must upload ", current_file[3]
                self.upload(ftp, current_file[3])
            elif current_file[1] == "" and current_file[2]:
                print "must download ", current_file[3], "and set time ", current_file[2]
                self.download(ftp, current_file[3], current_file[2])
                
        #close the connection
        ftp.quit()
        
    '''def Cancel(self, *l):
        print "Cancel"
        self.remove_widget(self.buttonDoIt)
        self.add_widget(self.buttonDoIt)'''
        
    def upload(self, ftp, filename):
        fileup = open(os.path.join(self.localPath, filename), 'rb') # open the file
        ftp.storbinary('STOR ' + filename, fileup) # send the file
        fileup.close() # close the file
        
        #must set the localtime to the current time, so times are synchronized if allowed
        try:
            os.utime(os.path.join(self.localPath, filename), None)
        except:
            print "os.utime not permitted"
    
    def download(self, ftp, filename, remotetimestr):
        # download the file        
        lf = open(os.path.join(self.localPath, filename), "wb")
        ftp.retrbinary("RETR " + filename, lf.write, 8*1024)
        lf.close()
        
        #set the localtime to the remote time
        # MUST CONVERT REMOTE TIME FROM UTC (add two hours for free)
        dt = datetime.strptime(remotetimestr, "%Y-%m-%d %H:%M:%S")
        print dt.strftime("%Y-%m-%d %H:%M:%S")
        st = time.mktime(dt.timetuple())
        #st = time.mktime(dt.utctimetuple()) #fix utc method 1, 1h problem
        #print dt.timetuple(), dt.utctimetuple(), st
        #os.utime(os.path.join(self.localPath, filename), None) # allowed but useless since it is the actual time
        try:
            os.utime(os.path.join(self.localPath, filename), (st, st)) #OSError: [Errno 1] Operation not permitted: on android
        except:
            print "os.utime not permitted"     
        

if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(FTPView())#width = 1280))
