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


class PSKFileLoader(object):
    """  """
    
    def __init__(self, infile, scale):
        """Loads a Wavefront OBJ file. """
        self.objects = [] #the MeshData list
        mesh = MeshData()
        
        global vertexlist, faces, faceslist, UVCoords, faceuv, facemat, matlist, dirname
        #global DEBUGLOG        
        #print ("Importing file: ", infile)

        #     md5_bones=[]
        pskfile = open(infile,'rb')
    #     if (DEBUGLOG):
    #         logpath = infile.replace(".psk", ".txt")
    #         print("logpath:",logpath)
    #         logf = open(logpath,'w')

    #     def printlog(strdata):
    #         if (DEBUGLOG):
    #             logf.write(strdata)

        objName = infile.split('\\')[-1].split('.')[0]
        
        #print("objName:",objName)
        #read general header
        indata = unpack('20s3i',pskfile.read(32))
        #not using the general header at this time
        #==================================================================================================
        # vertex point
        #==================================================================================================
        #read the PNTS0000 header
        indata = unpack('20s3i',pskfile.read(32))
        recCount = indata[3]
        Logger.debug("Nbr of PNTS0000 records: " + str(recCount))
        counter = 0
        vertexlist = []
        while counter < recCount:
            counter = counter + 1
            indata = unpack('3f',pskfile.read(12))
            #print(indata[0],indata[1],indata[2])
            vertexlist.extend([(indata[0] * scale ,indata[2] * scale ,indata[1] * scale)])
            #vertexlist.extend([(-indata[0] * scale ,indata[2] * scale ,indata[1] * scale)])            
            #Tmsh.vertices.append(NMesh.Vert(indata[0],indata[1],indata[2]))
                                   
            '''mesh.vertices.extend([indata[0],indata[1],indata[2]])
            mesh.vertices.extend([0,0,0])
            mesh.vertices.extend([0,0])'''
            

        #==================================================================================================
        # UV
        #==================================================================================================
        #read the VTXW0000 header
        indata = unpack('20s3i',pskfile.read(32))
        recCount = indata[3]
        Logger.debug("Nbr of VTXW0000 records: " + str(recCount))
        counter = 0
        UVCoords = []
        #UVCoords record format = [index to PNTS, U coord, v coord]
        while counter < recCount:
            counter = counter + 1
            indata = unpack('hhffhh',pskfile.read(16))
            UVCoords.append([indata[0],indata[2],indata[3]])
            
                       
            mesh.vertices.extend(vertexlist[indata[0]])
            #mesh.vertices.extend([0,0,0])
            mesh.vertices.extend([indata[2],indata[3]])
            
    #         print([indata[0],indata[2],indata[3]])
    #         print([indata[1],indata[2],indata[3]])

        #==================================================================================================
        # Face
        #==================================================================================================
        #read the FACE0000 header
        indata = unpack('20s3i',pskfile.read(32))
        recCount = indata[3]
        #recCount = 5000
        Logger.debug("Nbr of FACE0000 records: "+ str(recCount))
        #PSK FACE0000 fields: WdgIdx1|WdgIdx2|WdgIdx3|MatIdx|AuxMatIdx|SmthGrp
        #associate MatIdx to an image, associate SmthGrp to a material
        SGlist = []
        counter = 0
        faces = []
        faceslist=[]
        faceuv = []
        facemat = []
        while counter < recCount:
            
            indata = unpack('hhhbbi',pskfile.read(12))
            #the psk values are: nWdgIdx1|WdgIdx2|WdgIdx3|MatIdx|AuxMatIdx|SmthGrp
            #indata[0] = index of UVCoords
            #UVCoords[indata[0]]=[index to PNTS, U coord, v coord]
            #UVCoords[indata[0]][0] = index to PNTS
            PNTSA = UVCoords[indata[0]][0]
            PNTSB = UVCoords[indata[1]][0]
            PNTSC = UVCoords[indata[2]][0]
            #print(PNTSA,PNTSB,PNTSC) #face id vertex
            #faces.extend([0,1,2,0])
            faces.extend([PNTSA,PNTSB,PNTSC])
            faceslist.append([PNTSA,PNTSB,PNTSC])
            uv = []
            u0 = UVCoords[indata[0]][1]
            v0 = UVCoords[indata[0]][2]
            uv.append([u0,v0])
            u1 = UVCoords[indata[1]][1]
            v1 = UVCoords[indata[1]][2]
            uv.append([u1,v1])
            u2 = UVCoords[indata[2]][1]
            v2 = UVCoords[indata[2]][2]
            uv.append([u2,v2])
            faceuv.append(uv)
            
            '''mesh.indices.append(counter*3)
            mesh.indices.append(counter*3+1)
            mesh.indices.append(counter*3+2)'''
            
            mesh.indices.append(indata[0])
            mesh.indices.append(indata[1])
            mesh.indices.append(indata[2])
            
                        
            '''mesh.vertices.extend(vertexlist[PNTSA])
            #mesh.vertices.extend([0,0,0])
            mesh.vertices.extend([u0,v0])
            mesh.vertices.extend(vertexlist[PNTSB])
            #mesh.vertices.extend([0,0,0])
            mesh.vertices.extend([u1,v1])
            mesh.vertices.extend(vertexlist[PNTSC])
            #mesh.vertices.extend([0,0,0])
            mesh.vertices.extend([u2,v2])'''
            
            counter = counter + 1
            
 

            facemat.append(indata[3])
            #print("UV: ",u0,v0)
            #print indata[3] #mat index
            #update the uv var of the last item in the Tmsh.faces list
            # which is the face just added above
            ##Tmsh.faces[-1].uv = [(u0,v0),(u1,v1),(u2,v2)]
            #print("smooth:",indata[5])
            #collect a list of the smoothing groups
            if SGlist.count(indata[5]) == 0:
                SGlist.append(indata[5])
                #print("smooth:",indata[5])
            #assign a material index to the face
            #Tmsh.faces[-1].materialIndex = SGlist.index(indata[5])
    #     print "Using Materials to represent PSK Smoothing Groups...\n"
        #==========
        # skip something...
        #==========

        #==================================================================================================
        # Material
        #==================================================================================================
        ##
        #read the MATT0000 header
        '''indata = unpack('20s3i',pskfile.read(32))
        recCount = indata[3]
        Logger.debug( "Nbr of MATT0000 records: " +    str(recCount))
        #print " - Not importing any material data now. PSKs are texture wrapped! \n"
        matlist = []
        counter = 0
        while counter < recCount:
            matlist.append('')
            indata = unpack('64s6i',pskfile.read(88))
            Logger.debug( ' ' + str(counter) + '' + indata[0].rstrip('\x00') + ' ')

            diffuse = ''
            normal = ''
            try:
                matfile = open(dirname + indata[0].rstrip('\x00') + '.mat')
                for line in matfile:
                    if line.split('=')[0] == 'Diffuse':
                        diffuse = line.split('=')[1].replace('\n', '') + '.tga'
                    if line.split('=')[0] == 'Normal':
                        normal = line.split('=')[1].replace('\n', '') + '.tga'
            except:
                pass
            matlist[counter] = [diffuse, normal]
            counter = counter + 1'''
        #Logger.debug(matlist)
        
        
        #mesh.indices = faces
        self.objects.append(mesh)

        Logger.debug("vertex:"+str(len(vertexlist))+" faces:"+str(len(faces)) + " faceuv:"+ str(len(faceuv)))
        Logger.debug("mesh nb vertices:"+str(len(mesh.vertices)))
        Logger.debug("mesh nb indices:"+str(len(mesh.indices)))
        Logger.debug("mesh max indices:"+str(max(mesh.indices)))
        
        ##
