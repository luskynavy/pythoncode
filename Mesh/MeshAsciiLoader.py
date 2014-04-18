import os
from struct import *

from kivy.logger import Logger

class MeshData(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.vertex_format = [
            ('v_pos', 3, 'float'),
            #('v_normal', 3, 'float'),
            ('v_tc0', 2, 'float')]
        self.vertices = []
        self.indices = []
        self.diffuse = ""
        self.normal = ""
        
        # Default basic material of mesh object
        self.diffuse_color = (1.0, 1.0, 1.0)
        self.ambient_color = (1.0, 1.0, 1.0)
        self.specular_color = (1.0, 1.0, 1.0)
        self.specular_coefficent = 16.0
        self.transparency = 1.0
        
    def set_materials(self, mtl_dict):
        self.diffuse_color = mtl_dict.get('Kd') or self.diffuse_color
        self.diffuse_color = [float(v) for v in self.diffuse_color]
        self.ambient_color = mtl_dict.get('Ka') or self.ambient_color
        self.ambient_color = [float(v) for v in self.ambient_color]
        self.specular_color = mtl_dict.get('Ks') or self.specular_color
        self.specular_color = [float(v) for v in self.specular_color]
        self.specular_coefficent = float(mtl_dict.get('Ns', self.specular_coefficent))
        transparency = mtl_dict.get('d')
        if not transparency: 
            transparency = 1.0 - float(mtl_dict.get('Tr', 0))
        self.transparency = float(transparency)
        

    def calculate_normals(self):
        for i in range(len(self.indices) / (3)):
            fi = i * 3
            v1i = self.indices[fi]
            v2i = self.indices[fi + 1]
            v3i = self.indices[fi + 2]

            vs = self.vertices
            p1 = [vs[v1i + c] for c in range(3)]
            p2 = [vs[v2i + c] for c in range(3)]
            p3 = [vs[v3i + c] for c in range(3)]

            u,v  = [0,0,0], [0,0,0]
            for j in range(3):
                v[j] = p2[j] - p1[j]
                u[j] = p3[j] - p1[j]

            n = [0,0,0]
            n[0] = u[1] * v[2] - u[2] * v[1]
            n[1] = u[2] * v[0] - u[0] * v[2]
            n[2] = u[0] * v[1] - u[1] * v[0]

            for k in range(3):
                self.vertices[v1i + 3 + k] = n[k]
                self.vertices[v2i + 3 + k] = n[k]
                self.vertices[v3i + 3 + k] = n[k]

                
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


class MeshAsciiLoader(object):
    """  """
    
    def __init__(self, infile, scale):
        """Loads a Wavefront OBJ file. """
        self.objects = [] #the MeshData list        
        
        global vertexlist, faces, faceslist, UVCoords, faceuv, facemat, matlist, dirname
        
        dirname = os.path.dirname(infile)
        if dirname != '':
            dirname += '/'
    
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
        
        #meshes
        for i in range(0, meshCount):
            mesh = MeshData()
        
            meshFullName = meshName = ReadLine(file)
            print("meshFullName: " + meshFullName )
            
            uvLayerCount = int(ReadLineIgnoreComments(file))
            #print("UV Layer Count: " + str(uvLayerCount))
            
            textureCount = int(ReadLineIgnoreComments(file))
            print("Texture Count: " + str(textureCount))
            
            for textureID in range(0, textureCount):
                textureFilename = ReadLine(file)
                uvLayerID = int(ReadLineIgnoreComments(file))
                if (textureID == 0): #diffuse is always 0
                    mesh.diffuse = dirname + os.path.basename(textureFilename)
                elif (textureCount == 2): #diffuse then normal
                    mesh.normal = dirname + os.path.basename(textureFilename)
                elif (textureCount == 3 and textureID == 1): #diffuse, normal then specular
                    mesh.normal = dirname + os.path.basename(textureFilename)
                elif (textureCount == 4 and textureID == 2): #diffuse, light, normal then specular                    
                    mesh.normal = dirname + os.path.basename(textureFilename)
                
                print("textureFilename: " + textureFilename + " uvLayerID: " + str(uvLayerID))
            
            vertexCount = int(ReadLineIgnoreComments(file))
            
            vertexlist = []
            UVCoords = []
            faceslist = []
            
            #vertices
            for vertexID in range(0, vertexCount):
                tuple = ReadTuple(file)
                x = float(tuple[0])
                y = float(tuple[1])
                z = float(tuple[2])
                #coords.append([x, -z, y])
                vertexlist.extend([(x * scale, z * scale, y * scale)])
                
                #normals
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
                
                #uv per vertex
                uvList = []
                for i in range(0, uvLayerCount):
                    tuple = ReadTuple(file)
                    u = float(tuple[0])
                    v = float(tuple[1])
                    uvList.append([u, 1 - v])
                    UVCoords.append([u, v, 0])
                #uvs.append(uvList)
                
                mesh.vertices.extend([-x * scale, y * scale, -z * scale])
                #mesh.vertices.extend([0,0,0])
                mesh.vertices.extend([u, v])
                
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

            #faces
            for i in range(0, faceCount):
                tuple = ReadTuple(file)
                index1 = int(tuple[0])
                index2 = int(tuple[1])
                index3 = int(tuple[2])
                if (faceCount == 1):
                    indices.append([1, 2, 3])
                    mesh.indices.extend([0,1,2])
                else:
                    indices.append([index1 + 1, index3 + 1, index2 + 1])
                    faceslist.append([index1 + 1, index3 + 1, index2 + 1])
                    mesh.indices.extend([index1, index2, index3])

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
                 
            self.objects.append(mesh)
        
        print 'nb vertex', len(vertexlist), ', nb faces', len(faceslist)
        print 'nb UVCoords', len(UVCoords)
        
        print 'self.objects ' + str(len(self.objects))
