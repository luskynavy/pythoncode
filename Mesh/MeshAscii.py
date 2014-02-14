# non ascii format here https://github.com/cochrane/GLLara/blob/master/Documentation/XNALara%20Model%20File%20Format.md

# from import_xnalaura_mesh_ascii_extended.py

import math
import time
import sys
import os.path

import struct


def ReadLine(file):
    line = file.readline()
    if line.endswith('\n'):
        line = line[:-1]
    if line.endswith('\r'):
        line = line[:-1]
    return line

def ReadLineIgnoreComments(file):
    line = ReadLine(file)
    pos = line.find('#')
    if pos >= 0:
        line = line[0:pos]
    return line

def ReadTuple(file):
    line = ReadLineIgnoreComments(file)
    return line.split()

def meshAsciiImport(infile):
    global vertexlist, faces, faceslist, UVCoords, faceuv, facemat, matlist, dirname
    
    file = open(infile,'r')
    
    boneCount = int(ReadLineIgnoreComments(file))
    
    print "Bone count: " + str(boneCount)
    
    #bones
    for i in range(0, boneCount):
        boneName = ReadLine(file)
        print boneName
        parentID = int(ReadLineIgnoreComments(file))
        tuple = ReadTuple(file) # relPosition # .replace(',','.')
        
    meshCount = int(ReadLineIgnoreComments(file))
    print("Mesh count: " + str(meshCount))
    
    for i in range(0, meshCount):
        meshFullName = meshName = ReadLine(file)
        print("meshFullName: " + meshFullName )
        
        uvLayerCount = int(ReadLineIgnoreComments(file))
        print("UV Layer Count: " + str(uvLayerCount))
        
        textureCount = int(ReadLineIgnoreComments(file))
        print("Texture Count: " + str(textureCount))
        
        for textureID in range(0, textureCount):
            textureFilename = ReadLine(file)
            uvLayerID = int(ReadLineIgnoreComments(file))
        
        vertexCount = int(ReadLineIgnoreComments(file))
        
        vertexlist = []
        UVCoords = []
        faceslist = []
        
        for vertexID in range(0, vertexCount):
            tuple = ReadTuple(file)
            x = float(tuple[0])
            y = float(tuple[1])
            z = float(tuple[2])
            #coords.append([x, -z, y])
            vertexlist.extend([(x * scale ,z * scale ,y * scale)])
            
            tuple = ReadTuple(file)
            nx = float(tuple[0])
            ny = float(tuple[1])
            nz = float(tuple[2])
            #normals.append([nx, -nz, ny])
            
            tuple = ReadTuple(file)
            # r = int(tuple[0])
            # g = int(tuple[1])
            # b = int(tuple[2])
            # a = int(tuple[3])
            # colors.append([r, g, b, a])
            
            uvList = []
            for i in range(0, uvLayerCount):
                tuple = ReadTuple(file)
                u = float(tuple[0])
                v = float(tuple[1])
                uvList.append([u, 1 - v])
                UVCoords.append([u, v, 0])
            #uvs.append(uvList)
            
            if (boneCount != None):
                if (boneCount > 0):
                    indices = ReadTuple(file) # boneIndices
                    weights = ReadTuple(file) # boneWeights
                
                    for id in range(0, len(indices) ):
                        groupID = int(indices[id])
                        #print("groupID: " + str(groupID))
                        weight = float(weights[id])
                        #if (groupID > 0) and weight > 0:
                            #nameGroup = bonesNameList[int(groupID)]
                            #vrtxList.append([nameGroup, weight, int(vertexID+1)])
                            #nbVrtx = nbVrtx + 1
    

        faceCount = int(ReadLineIgnoreComments(file))
        # print faceCount
        
        indices = []

        for i in range(0, faceCount):
            tuple = ReadTuple(file)
            index1 = int(tuple[0])
            index2 = int(tuple[1])
            index3 = int(tuple[2])
            if (faceCount == 1):
                indices.append([1, 2, 3])
            else:
                indices.append([index1 + 1, index3 + 1, index2 + 1])
                faceslist.append([index1 + 1, index3 + 1, index2 + 1])

        # mesh.faces.extend(indices, ignoreDups=True)

        faces = []
        # face_uvs = []
        # 
        # # Monkey 13
        faceDiffs = 0
        for faceID in range(0, faceCount):
         # print faceID
         try:
             #face = mesh.faces[faceID]
             #face.image = faceImage
             index1 = indices[faceID][0]
             index2 = indices[faceID][1]
             index3 = indices[faceID][2]
             # color1 = colors[index1]
             # color2 = colors[index2]
             # color3 = colors[index3]
             # face.col[0].r = color1[0]
             # face.col[0].g = color1[1]
             # face.col[0].b = color1[2]
             # face.col[0].a = color1[3]
             # face.col[1].r = color2[0]
             # face.col[1].g = color2[1]
             # face.col[1].b = color2[2]
             # face.col[1].a = color2[3]
             # face.col[2].r = color3[0]
             # face.col[2].g = color3[1]
             # face.col[2].b = color3[2]
             # face.col[2].a = color3[3]
             # face.smooth = True
         
             faces.extend([index1, index2, index3, 0])
             
         except:
             print (str(faceID))
    
scale = .07

##################################main
if __name__ == "__main__":
    global vertexlist, faces, faceslist, UVCoords, faceuv

    meshAsciiImport("D:\Data\Python\Mesh\Pyro\Pyro Red\Generic_Item.mesh.ascii")
    
    print 'nb vertex', len(vertexlist), ', nb faces', len(faceslist)