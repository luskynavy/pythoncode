#!/usr/bin/python3

# BGPyrtrait is a small program that crop and resize pictures to be used
# in Baldur's Gate : Enhanced Edition.

from os import path, walk
import webbrowser
import threading
import queue
import sys
import time

import ntpath
import traceback

from tkinter import *
from tkinter.ttk import *
import tkinter.filedialog
import tkinter.messagebox

# BG     : M 110*170 S 38*60
# BG  EE : M 210*330 S 38*60
# IWD EE : L 20x660 M 169*266 S 38*60
# POE EE : L 420x660 M 169*266 S 38*60 (210x330 76*96 ?)
            
IMAGE_L_WIDTH = 220 #210
IMAGE_L_HEIGHT = 340 #330
IMAGE_M_WIDTH = 210 #169
IMAGE_M_HEIGHT = 330 #266
IMAGE_S_WIDTH = 85 #54
IMAGE_S_HEIGHT = 133 #84

IMAGE_L_WIDTH_OLD = 110
IMAGE_L_HEIGHT_OLD = 170
IMAGE_S_WIDTH_OLD = 38
IMAGE_S_HEIGHT_OLD = 60

class BatchWin(Toplevel):
    """Window for batch processing. The user choose an input folder and
    an output folder, then any pictures located in input folder and all
    of its subfolders are directly resized to BG:EE pictures sizes.
    Names are automaticaly formatted, and duplicated names are taken
    care of."""
    def __init__(self, master=None, portraits=None, queue=None):
        super().__init__(master)
        self.portraits = portraits #Reference to portraits frames, to resize and save pictures
        self.queue = queue #To update the progress label

        "Geometry to ensure that the batch window is somewhere inside the main window"
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        height = master.winfo_height()
        width = master.winfo_width()
        geometry = "+%d+%d" % (x+width/4,y+height/4)
        self.geometry(geometry)
        
        self.inputdir = None
        self.outputdir = None
        self.filepaths = []
        self.total_file_num = 0
        self.image_file_num = 0
        self.completed_file_num = 0
        
        self.title("Batch processing")
        self.resizable(0,0)
        self.create_widgets()
        self.transient(master)
        self.grab_set()

    def create_widgets(self):
        """Building the window"""
        "Frame for input folder"
        frame_in = Frame(self)
        self.label_in = Label(frame_in, width=22, text="Input Folder...", relief=SUNKEN, anchor=CENTER)
        self.label_in.pack(side=LEFT, padx=(8,0))
        Button(frame_in, text="Input Folder", width=12, command=self.getinputfolder).pack(side=LEFT, padx=8, pady=8)
        frame_in.pack(fill=X)

        "Frame for output folder"
        frame_out = Frame(self)
        self.label_out = Label(frame_out, width=22, text="Output Folder...", relief=SUNKEN, anchor=CENTER)
        self.label_out.pack(side=LEFT, padx=(8,0))
        Button(frame_out, text="Output Folder", width=12, command=self.getoutputfolder).pack(side=LEFT, padx=8, pady=8)
        frame_out.pack()
        
        "Progress label, to show the progress of the batch processing"
        self.progresslabelvar = StringVar()
        self.progresslabelvar.set("Idle")
        self.progresslabel = Label(self, textvariable=self.progresslabelvar)
        self.progresslabel.pack(pady=(0,8))
        
        "Buttons"
        frame_buttons = Frame(self)
        self.button_ok = Button(frame_buttons, text="OK", command=self.launchbatch)
        button_cancel = Button(frame_buttons, text="Cancel", command=self.quitbatch)
        self.button_ok.pack(side=RIGHT, padx=8)
        button_cancel.pack(side=RIGHT, padx=8)
        frame_buttons.pack(pady=8)
        
    def getinputfolder(self):
        self.inputdir = tkinter.filedialog.askdirectory(title="Input Folder", parent=self)
        if self.inputdir:
            lastfolder = path.split(self.inputdir)[-1]
            self.label_in.configure(text=lastfolder)
        
    def getoutputfolder(self):
        self.outputdir = tkinter.filedialog.askdirectory(title="Output Folder", parent=self)
        if self.outputdir:
            lastfolder = path.split(self.outputdir)[-1]
            self.label_out.configure(text=lastfolder)
        
    def launchbatch(self):
        if self.inputdir and self.outputdir:
            self.total_file_num = 0
            self.image_file_num = 0
            self.completed_file_num = 0
            self.filepaths = []
            
            self.getfile_event = threading.Event() #To tell that the search of files is finished
            self.getfile_event.clear()
            t_getfiles = threading.Thread(target=self.getfilelist)
            t_conv = threading.Thread(target=self.startconversion)
            t_getfiles.start()
            t_conv.start()
        else:
            tkinter.messagebox.showerror(title="Error", message="Select input and output folders !")
        
    def quitbatch(self):
        self.destroy()
        
    def startconversion(self):
        self.getfile_event.wait()
        namelist = [] # Pool of names, to ensure no duplicate names appear
        
        for file in self.filepaths:
            im = Image.open(file)
            basename = format_base_filename(file)
            i = 1
            while basename in namelist: # Searching for duplicate
                basename = basename[0:6] + str(i)
                i += 1
            namelist.append(basename)
            """Loading the picture into the frames, auto resize apply"""
            for i in range(0,3):
                self.portraits[i].setImage(im)
            self.portraits[0].im.save(path.join(self.outputdir, basename + "L.bmp"))
            self.portraits[1].im.save(path.join(self.outputdir, basename + "M.bmp"))
            self.portraits[2].im.save(path.join(self.outputdir, basename + "S.bmp"))
                
            self.completed_file_num += 1
            text = "Processing : " + str(self.completed_file_num) + "/" + str(self.image_file_num)
            self.queue.put(text)
            
        """Clearing after the batch process"""
        for portrait in self.portraits:
            portrait.clearImage()
        self.queue.put("Idle")
        
    def getfilelist(self):
        """Search for valid picture files in folder and subfolder"""
        for root, subFolders, files in walk(self.inputdir):
            for file in files:
                self.total_file_num += 1
                try:
                    Image.open(path.join(root,file)).verify()
                    """Valid picture file is saved into filepaths list"""
                    self.filepaths.append(path.join(root,file))
                    self.image_file_num += 1
                except:
                    pass
                text = "Scanning : " + str(self.image_file_num) + "/" + str(self.total_file_num)
                self.queue.put(text)
        """Searching for files ended, triggering the resize/save method"""
        self.getfile_event.set()
        

class Portrait(Frame):
    """Custom class herited from Tkinter Frame, which contains image datas and
    different values for picture croping. This class can be easily
    reused for other software (with minor modifications)"""
    def __init__(self, width, height, app, **kargs):
        super().__init__(**kargs)
        self.app = app
        
        self.max_scale = 0.0
        self.current_scale = 0.0
        self.x = 0.0
        self.y = 0.0
        self.relx = 0.0
        self.rely = 0.0
        self.angle = 0.0
        self.relAngle = 0.0
        
        self.height = height
        self.width = width
        
        self.view = Label(self)
        self.view.pack()
        
        self.im = None
        self.source_image = None
        self.imtk = None
        
    def clearImage(self):
        self.im = None
        self.source_image = None
        self.imtk = None
        self.view["image"] = ""
        
    "Bind/unbind for mouse dragging and scrolling"
    def setBinds(self):
        self.view.bind("<B1-Motion>", self.change_position)
        self.view.bind("<Button-1>", self.save_position)
        # with Windows OS
        self.view.bind("<Enter>",lambda event:self.view.focus_set())
        self.view.bind("<MouseWheel>", self.change_scale)
        # with Linux OS
        self.view.bind("<Button-4>", self.change_scale)
        self.view.bind("<Button-5>", self.change_scale)
        #rotate
        self.view.bind("<B3-Motion>", self.change_angle)
        self.view.bind("<Button-3>", self.save_angle)
        
    def unsetBinds(self):
        self.view.unbind("<B1-Motion>")
        self.view.unbind("<Button-1>")
        # with Windows OS
        self.view.unbind("<MouseWheel>")
        # with Linux OS
        self.view.unbind("<Button-4>")
        self.view.unbind("<Button-5>")
        #rotate
        self.view.unbind("<B3-Motion>")
        self.view.unbind("<Button-3>")
        
    "Relative angle for dragging"
    def save_angle(self, event):
        self.relAngle = event.y
        
    "Angle Drag occurs"
    def change_angle(self, event):
        self.angle = self.angle + (self.relAngle - event.y)
        self.source_image = self.app.source_image.rotate(self.angle/12, Image.BICUBIC, True)
        self.scaleCropImage()
        self.displayImage()
        if self.app.connect:
            self.app.portrait_M.setImage(self.im)
            self.app.portrait_S.setImage(self.im)
        
    "Relative position for dragging"
    def save_position(self, event):
        self.relx, self.rely = event.x, event.y
        
    "Drag occurs"
    def change_position(self, event):
        self.x = self.x + (self.relx - event.x)*self.current_scale
        self.y = self.y + (self.rely - event.y)*self.current_scale
        self.clipCenter()
        self.relx, self.rely = event.x, event.y
        self.scaleCropImage()
        self.displayImage()
        if self.app.connect:
            self.app.portrait_M.setImage(self.im)
            self.app.portrait_S.setImage(self.im)
        
    "The picture can't go outside the frame"
    def clipCenter(self):
        minx = self.width*self.current_scale/2
        maxx = self.source_image.size[0] - self.width*self.current_scale/2
        #self.x = minx if self.x < minx else maxx if self.x > maxx else self.x
        
        miny = self.height*self.current_scale/2
        maxy = self.source_image.size[1] - self.height*self.current_scale/2
        #self.y = miny if self.y < miny else maxy if self.y > maxy else self.y        
        
    "Scroll occurs"
    def change_scale(self, event):
        if event.num == 5 or event.delta == -120:
            self.current_scale *= 1.05 #1.1
        if event.num == 4 or event.delta == 120:
            self.current_scale *= 0.95 #0.9
        self.current_scale = min([self.current_scale, self.max_scale])
        self.clipCenter()
        self.scaleCropImage()
        self.displayImage()
        if self.app.connect:
            self.app.portrait_M.setImage(self.im)
            self.app.portrait_S.setImage(self.im)
        
    def setImage(self, image):
        self.source_image = image.convert("RGB") # For png files with transparency
        #self.max_scale = min([image.size[0]/self.width, image.size[1]/self.height])
        self.max_scale = max([image.size[0]/self.width, image.size[1]/self.height])
        
        #Initial setup
        self.current_scale = self.max_scale
        self.x, self.y = image.size[0]/2, image.size[1]/2
        
        self.angle = 0.0
        
        self.scaleCropImage()
        self.displayImage()
        
    def scaleCropImage(self):
        x0 = self.x - self.width*self.current_scale/2
        x1 = self.x + self.width*self.current_scale/2
        y0 = self.y - self.height*self.current_scale/2
        y1 = self.y + self.height*self.current_scale/2
        #self.im = self.source_image.transform((self.width, self.height), Image.EXTENT, (x0, y0, x1, y1), Image.BICUBIC)        
        temp = self.source_image.crop((int(x0), int(y0), int(x1), int(y1)))        
        self.im = temp.resize((self.width, self.height), Image.ANTIALIAS)        
        if self.app.connect:
            if self.width == IMAGE_M_WIDTH:
                self.im = self.app.portrait_L.im.resize((IMAGE_M_WIDTH, IMAGE_M_HEIGHT), Image.ANTIALIAS)
            if self.width == IMAGE_S_WIDTH:
                self.im = self.app.portrait_L.im.resize((IMAGE_S_WIDTH, IMAGE_S_HEIGHT), Image.ANTIALIAS)
        
    "Display image in the tkinter frame"
    def displayImage(self):
        self.imtk = ImageTk.PhotoImage(self.im)
        self.view.config(image=self.imtk)
        

class Application(Frame):
    """Main application frame"""
    def __init__(self, master=None):
        super().__init__(master)
        
        self.source_image = None
        
        self.source_image_filepath = ""
        self.connect = True
        
        self.master = master
        self.create_menus(master)
        self.create_toolbar(master)
        self.create_labelframes(master)
        self.pack()
        
        "To update the GUI during the batch processing"
        self.guieventqueue = queue.Queue()
        self.after(2, self.updategui)
        
        self.not_saved = False
        self.filename_saved = ""

    def updategui(self):
        try:
            while True:
                text = self.guieventqueue.get_nowait()
                self.batchwindow.progresslabelvar.set(text)
                self.update_idletasks()
                self.guieventqueue.task_done()
        except:
            pass
        self.after(2, self.updategui)
        
    def create_menus(self, master=None):
        menubar = Menu(master)
        
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.openfile)
        filemenu.add_command(label="Open Folder", command=self.openfolder)
        filemenu.add_command(label="Save", command=self.save)
        filemenu.add_command(label="Save As", command=self.saveas)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quitapp)
        menubar.add_cascade(label="File", menu=filemenu)
        
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        helpmenu.add_command(label="Donate", command=openDonate)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        master.config(menu=menubar)
        
    def create_toolbar(self, master=None):
        toolbar = Frame(master, relief=FLAT)
        openbutton = Button(toolbar, text="Open", command=self.openfile)
        openfolder = Button(toolbar, text="Open Folder", command=self.openfolder)
        savebutton = Button(toolbar, text="Save", command=self.save)
        saveasbutton = Button(toolbar, text="Save As", command=self.saveas)
        connectbutton = Button(toolbar, text="Disconnect", command= lambda: self.connectPressed(connectbutton))
        openbutton.pack(side=LEFT)
        openfolder.pack(side=LEFT)
        savebutton.pack(side=LEFT)
        saveasbutton.pack(side=LEFT)
        connectbutton.pack(side=LEFT)
        toolbar.pack(fill=X)
        
    def create_labelframes(self, master=None):
        
        self.frame_L = LabelFrame(master, text= str(IMAGE_L_WIDTH) + "x" + str(IMAGE_L_HEIGHT), height=360, width=230, labelanchor=N)
        self.frame_M = LabelFrame(master, text= str(IMAGE_M_WIDTH) + "x" + str(IMAGE_M_HEIGHT) + " (locked)", height=360, width=230, labelanchor=N)
        self.frame_S = LabelFrame(master, text= str(IMAGE_S_WIDTH) + "x" + str(IMAGE_S_HEIGHT) + " (locked)", height=360, width=230, labelanchor=N)
        
        self.portrait_L = Portrait(master=self.frame_L, height=IMAGE_L_HEIGHT, width=IMAGE_L_WIDTH, app=self)
        self.portrait_L.place(anchor=CENTER, relx=0.5, rely=0.5)
        self.portrait_L.setBinds()
        
        self.portrait_M = Portrait(master=self.frame_M, height=IMAGE_M_HEIGHT, width=IMAGE_M_WIDTH, app=self)
        self.portrait_M.place(anchor=CENTER, relx=0.5, rely=0.5)
        
        self.portrait_S = Portrait(master=self.frame_S, height=IMAGE_S_HEIGHT, width=IMAGE_S_WIDTH, app=self)
        self.portrait_S.place(anchor=CENTER, relx=0.5, rely=0.5)
        
        self.frame_L.pack(side=LEFT, fill=Y, padx=10, pady=10)
        self.frame_M.pack(side=LEFT, fill=Y, padx=10, pady=10)
        self.frame_S.pack(side=LEFT, fill=Y, padx=10, pady=10)
        
    def connectPressed(self, button):
        if self.connect:
            self.connect = False
            button.configure(text="Connect")
            self.frame_M.configure(text=str(IMAGE_M_WIDTH) + "x" + str(IMAGE_M_HEIGHT))
            self.frame_S.configure(text=str(IMAGE_S_WIDTH) + "x" + str(IMAGE_S_HEIGHT))
            self.portrait_M.setBinds()
            self.portrait_S.setBinds()
            self.portrait_M.setImage(self.source_image)
            self.portrait_S.setImage(self.source_image)
            
            self.portrait_M.x = self.portrait_L.x
            self.portrait_M.y = self.portrait_L.y
            self.portrait_M.current_scale = self.portrait_L.current_scale*IMAGE_L_HEIGHT/IMAGE_M_HEIGHT #1.24
            #set the angle and apply the rotation
            self.portrait_M.angle = self.portrait_L.angle
            self.portrait_M.source_image = self.source_image.rotate(self.portrait_M.angle/12, Image.BICUBIC, True)
            self.portrait_M.clipCenter()
            self.portrait_M.scaleCropImage()
            self.portrait_M.displayImage()
            self.portrait_S.x = self.portrait_L.x
            self.portrait_S.y = self.portrait_L.y
            self.portrait_S.current_scale = self.portrait_L.current_scale*IMAGE_L_HEIGHT/IMAGE_S_HEIGHT #3.93
            #set the angle and apply the rotation
            self.portrait_S.angle = self.portrait_L.angle            
            self.portrait_S.source_image = self.source_image.rotate(self.portrait_S.angle/12, Image.BICUBIC, True)
            self.portrait_S.clipCenter()
            self.portrait_S.scaleCropImage()
            self.portrait_S.displayImage()
        else:
            self.connect = True
            button.configure(text="Disconnect")
            self.frame_M.configure(text=str(IMAGE_M_WIDTH) + "x" + str(IMAGE_M_HEIGHT) + " (locked)")
            self.frame_S.configure(text=str(IMAGE_S_WIDTH) + "x" + str(IMAGE_S_HEIGHT) + " (locked)")
            self.portrait_M.unsetBinds()
            self.portrait_S.unsetBinds()
            self.portrait_M.setImage(self.portrait_L.im)
            self.portrait_S.setImage(self.portrait_L.im)
        
    def openfile(self):
        self.source_image_filepath = tkinter.filedialog.askopenfilename(filetypes = (("Images", "*.jpg;*.jpeg;*.gif;*.png")
                                                         ,("All files", "*.*") ))
        if self.source_image_filepath:
            try:
                self.source_image = Image.open(self.source_image_filepath)
                self.master.title(ntpath.split(self.source_image_filepath)[1])
                self.portrait_L.setImage(self.source_image)
                self.portrait_M.setImage(self.source_image)
                self.portrait_S.setImage(self.source_image)
                self.not_saved = True
                self.filename_saved = ""
            except Exception:                
                traceback.print_exc()
                tkinter.messagebox.showerror(title="Unable to open File", message="Can't open file !\n Is it a image file ?")
        
    def quitapp(self):
        if self.source_image:
            if tkinter.messagebox.askyesno("Quit", "A Portrait is loaded !\n Are you sure you want to quit ?"):
                self.master.quit()
        else:
            self.master.quit()
            
    def about(self):
        tkinter.messagebox.showinfo("About BGPyrtraits", "BGPyrtraits : Made by Vagdish\nwith Python3 and Tkinter")
        
    def save(self):
        #generate and increment new filename only on first save
        if self.not_saved:
            filesnames = format_save_filepath(self.source_image_filepath, self.not_saved)
            self.filename_saved = filesnames
        else:
            filesnames = self.filename_saved
            
        self.master.title(ntpath.split(self.source_image_filepath)[1] + " " + ntpath.split(self.filename_saved[0])[1])
            
        #self.portrait_L.im.save(filesnames[0], "BMP")
        self.portrait_M.im.save(filesnames[1], "BMP")
        #imgM = self.portrait_L.im.resize((IMAGE_M_WIDTH, IMAGE_M_HEIGHT), Image.ANTIALIAS)
        #imgM.save(filesnames[1], "BMP")
        self.portrait_S.im.save(filesnames[2], "BMP")
        #imgS = self.portrait_L.im.resize((IMAGE_S_WIDTH, IMAGE_S_HEIGHT), Image.ANTIALIAS)
        #imgS.save(filesnames[2], "BMP")
        
        #keep saved filename 
        self.not_saved = False
        
    def saveas(self):
        requestedfilesnames = tkinter.filedialog.asksaveasfilename()
        if requestedfilesnames:
            filesnames = format_save_filepath(requestedfilesnames)
            #self.portrait_L.im.save(filesnames[0], "BMP")
            self.portrait_M.im.save(filesnames[1], "BMP")
            self.portrait_S.im.save(filesnames[2], "BMP")
            
    def openfolder(self):
        portraits = [self.portrait_L, self.portrait_M, self.portrait_S]
        self.batchwindow = BatchWin(self.master, portraits, self.guieventqueue)
            
        """Next 2 methods format the names of the output files
        as BG:EE needs a strict name formatting"""
def format_save_filepath(filepath, increment=False):
        folder, filename = path.split(filepath)
        filename = filename.replace(" ", "")[0:7]
        filename = filename.replace(".", "")

        s_name = path.join(folder, filename + "S.bmp")
        
        #increment filename number if needed
        if increment:            
            if path.isfile(s_name):
                filename = filename.replace(" ", "")[0:6]
                i = 0
                while path.isfile(s_name) and i < 999:
                    s_name = path.join(folder, filename + str(i) + "S.bmp")
                    i+= 1
                filename = filename.replace(" ", "")[0:6] + str(i - 1)
        
        l_name = path.join(folder, filename + "L.bmp")
        m_name = path.join(folder, filename + "M.bmp")
        #s_name = path.join(folder, filename + "S.bmp")
        return [l_name, m_name, s_name]
        
def format_base_filename(filepath):
        folder, filename = path.split(filepath)
        filename = filename.replace(" ", "")[0:7]
        return filename
        
def openDonate():
    url = "https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=7ZLRM6ABY963U"
    webbrowser.open_new(url)
    
    """Customs cmd line options asked by community members to resize
    at specific size. Nice reuse of the portrait objects (instanciated by Application)"""
def cmdbatch(width, height, suffixe, folderinpath, folderoutpath):
    portrait = Portrait(height=int(height), width=int(width), app=None)
    for root, subFolders, files in walk(folderinpath):
        for file in files:
            print("Resizing \"{}\" to {}x{} with suffixe \"{}\" to {}".format(file,width,height,suffixe,folderoutpath), end=" - ")			
            try:
                image = Image.open(path.join(root,file))
                portrait.setImage(image)
                filename = file[0:7] + suffixe + ".BMP"
                portrait.im.save(path.join(folderoutpath,filename), "BMP")
                print("Done !")
            except:
                print("Not a picture ?")
                
def cmdfile(width, height, suffixe, inputfile, folderoutpath):
    print("Resizing \"{}\" to {}x{} with suffixe \"{}\" to {}".format(path.basename(inputfile),width,height,suffixe,folderoutpath), end=" - ")
    portrait = Portrait(height=int(height), width=int(width), app=None)
    try:
        image = Image.open(inputfile)
        portrait.setImage(image)
        filename = path.basename(inputfile)[0:7] + suffixe + ".BMP"
        portrait.im.save(path.join(folderoutpath,filename), "BMP")
        print("Done !")
    except:
        print("Error occured !")
        
if __name__ == "__main__" :
    from PIL import Image, ImageTk
    if len(sys.argv) == 1:
        root = Tk()
        root.resizable(0,0)
        root.title("BGPyrtraits")
        Application(master=root).mainloop()
    elif len(sys.argv) == 6:
        # Args: width height suffixe inputfolder outputfolder
        if path.isdir(sys.argv[4]):
            cmdbatch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        elif path.isfile(sys.argv[4]):
            cmdfile(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("Wrong number of arguments (0 for gui, 5 for cmd batch processing)")
