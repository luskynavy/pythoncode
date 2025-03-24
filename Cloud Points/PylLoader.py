import os
import math
import time
from struct import *
#import numpy as np
import plyfile

from kivy.logger import Logger

class MeshData(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.vertex_format = [
            ('v_pos', 3, 'float'),
            ('v_color', 4, 'float'),
        ]
        self.vertices = []
        self.indices = []

class PylLoader(object):
    """  """
    
    def __init__(self, infile, scale):
        """Loads a PLY ASCII file. """
        
        t = time.clock()
        self.objects = [] #the MeshData list
        
        '''indices = [0, 1, 2, 3]
        vertices = [
           .5,  .5, 1.0, 1.0, 0.0, 0.0, 1.0,
           .5, -.5, 1.0, 0.0, 1.0, 0.0, 1.0,
          -.5,  .5, 1.0, 0.0, 0.0, 1.0, 1.0,
          -.5, -.5, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]'''
        
        Logger.info( 'Loading ' + infile)
        
        with open(infile, 'r') as i_file:
            plydata = plyfile.PlyData.read(i_file)
            
        Logger.debug("Loaded in " + str(time.clock() - t))
        
        t = time.clock()
        
        nbTotalVertex = plydata['vertex'].data.size
        
        #max vertex allowed for a mesh (Open gl es limitation)
        maxVertexByMesh = 65535
        
        nbMesh = nbTotalVertex / maxVertexByMesh
        
        #split vertices
        for i in range(nbMesh):
            offset = i * maxVertexByMesh
            
            #get remaining vertices 
            if offset + maxVertexByMesh > nbTotalVertex:
                nbVertex = nbTotalVertex - offset
            else:
                nbVertex = maxVertexByMesh
        
            #if nbVertex >= maxVertexByMesh:
            #    nbVertex = maxVertexByMesh
            
            mesh = MeshData()
            
            #create current mesh
            i = 0
            mesh.indices = range(nbVertex)
            mesh.vertices = [0] * nbVertex * (3 + 4)
            for v in plydata['vertex'].data[offset:offset + nbVertex]:
                mesh.vertices[i * 7 + 0] = v['x'] * scale 
                mesh.vertices[i * 7 + 1] = v['y'] * scale 
                mesh.vertices[i * 7 + 2] = v['z'] * scale
                #mesh.vertices[i * 7 + 3] = 1.0
                #mesh.vertices[i * 7 + 4] = 0
                #mesh.vertices[i * 7 + 5] = 0
                mesh.vertices[i * 7 + 3] = float(v['red']) / 255
                mesh.vertices[i * 7 + 4] = float(v['green']) / 255
                mesh.vertices[i * 7 + 5] = float(v['blue']) / 255            
                #mesh.vertices[i * 7 + 3] = float(v['diffuse_red']) / 255
                #mesh.vertices[i * 7 + 4] = float(v['diffuse_green']) / 255
                #mesh.vertices[i * 7 + 5] = float(v['diffuse_blue']) / 255
                mesh.vertices[i * 7 + 6] = 1.0
                #print i            
                i += 1
                
            self.objects.append(mesh)
        
        Logger.debug('self.objects ' + str(len(self.objects)))
        
        Logger.debug("Generated meshes in " + str(time.clock() - t))
        