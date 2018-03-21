#!/usr/bin/python3

# BGPyrtrait is a small program that crop and resize pictures to be used
# in Baldur's Gate : Enhanced Edition.

#from os import path, walk
#import webbrowser
#import threading
#import queue
#import sys
#import time

#import ntpath
#import traceback

from tkinter import *
from tkinter.ttk import *
#import tkinter.filedialog
#import tkinter.messagebox

#from PIL import Image, ImageTk

from BGPyrtraitsEE import *

if __name__ == "__main__" :
    config = Config()
    config.IMAGE_L_WIDTH = 220 #210
    config.IMAGE_L_HEIGHT = 340 #330
    config.IMAGE_M_WIDTH = 210 #169
    config.IMAGE_M_HEIGHT = 330 #266
    config.IMAGE_S_WIDTH = 76 #54
    config.IMAGE_S_HEIGHT = 96 #84
 
    config.EXPORT_FORMAT = "png"
    
    config.M_SUFFIX = "_lg."
    config.S_SUFFIX = "_sm."
    
    config.OVERLAY_IMG = "/overlayPoE.png"
    
    if len(sys.argv) == 1:
        root = Tk()
        root.resizable(0,0)
        root.title("BGPyrtraits")
        Application(config, master=root).mainloop()
    elif len(sys.argv) == 6:
        # Args: width height suffixe inputfolder outputfolder
        if path.isdir(sys.argv[4]):
            cmdbatch(config, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        elif path.isfile(sys.argv[4]):
            cmdfile(config, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("Wrong number of arguments (0 for gui, 5 for cmd batch processing)")
