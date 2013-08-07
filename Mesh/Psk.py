#!/usr/bin/env python


from PIL import Image

import numpy as np
from itertools import izip

import math
import time
import sys
import os.path

# import pyglet
# import struct
from struct import *
from pyglet.gl import *
from pyglet import image #<==for image calls
#from pyglet.image.codecs.png import PNGImageDecoder
#from pyglet.image.codecs.pil import PILImageDecoder
from pyglet.window import key
#from OpenGL.GLUT import * #<<<==Needed for GLUT calls

from ctypes import *

from shader import Shader

''' to test
https://svn.blender.org/svnroot/bf-blender/trunk/blender/source/blender/gpu/
https://svn.blender.org/svnroot/bf-blender/trunk/blender/source/blender/gpu/shaders/
https://svn.blender.org/svnroot/bf-blender/trunk/blender/source/blender/gpu/intern/gpu_simple_shader.c

http://en.wikibooks.org/wiki/GLSL_Programming
'''

shader = Shader(['''
//varying vec4 pt;
varying vec3 colorL, eyeVec, lightDir1_, normalDirection;
//varying vec3 spec;
void main() {
    vec3 normal   = normalize(gl_NormalMatrix * gl_Normal);
    vec3 tangent  = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5));
    vec3 binormal = cross(normal, tangent);
    mat3 TBNMatrix = mat3(tangent, binormal, normal);       
    
    normalDirection = normalize(gl_NormalMatrix * gl_Normal);
    vec3 lightDir1 = normalize(vec3(gl_LightSource[1].position));
    lightDir1_ = -normalize(vec3(gl_ModelViewMatrix * gl_LightSource[1].position));
    //lightDir1_ = lightDir1;
    //vec3 lightDirection = normalize(vec3(0.0, 1.0, 1.0));
    
    vec3 diffuseReflection = 
       vec3(gl_LightSource[1].diffuse) 
       * 1.0 //* vec3(gl_FrontMaterial.emission)
       * max(0.0, dot(normalDirection, lightDir1));
       //* dot(normalDirection, lightDirection);
    
    colorL = diffuseReflection;
        
    //eyeVec = -normalize(vec3(gl_ModelViewMatrix * gl_Vertex));
    vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

    //lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex) * TBNMatrix;
    //lightDir1_ = vec3(gl_LightSource[1].position.xyz - vVertex) * TBNMatrix;
    eyeVec    = -vVertex * TBNMatrix;
    
    //pt = gl_Vertex;
    //pt.z /= 20.0;
    //pt.z += 1.5;
    
    gl_TexCoord[0] = gl_MultiTexCoord0;
    //gl_TexCoord[0]  = gl_TextureMatrix[0] * gl_MultiTexCoord0;
    
    // Set the position of the current vertex
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    //gl_Position = gl_ModelViewProjectionMatrix * pt;
}
'''], ['''
varying vec3 colorL;
varying vec3 eyeVec, lightDir1_, normalDirection;
uniform sampler2D color_texture;
uniform sampler2D normal_texture;
uniform int toggletexture; // false/true

void main() {
    
    // Extract the normal from the normal map
    vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rgb * 2. - 1.);
    //vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rgb);
    //vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rgb - .5);    
    
    // Calculate the lighting diffuse value
    //float diffuse = max(dot(normal, light_pos), 0.0);
    float diffuse = dot(normal, colorL);
    vec3 L1 = normalize(lightDir1_);
    
    vec3 color = (toggletexture == 1
        //? diffuse/2.0 + texture2D(color_texture, gl_TexCoord[0].st).rgb/2.0
        ? diffuse * texture2D(color_texture, gl_TexCoord[0].st).rgb
        //: diffuse/2.0 + vec3(0.75, 0.75, 0.75)/2.0);
        : diffuse * vec3(0.75, 0.75, 0.75));
    //vec3 color = colorL * vec3(0.5, 0.5, 0.5);
    
    vec3 E = normalize(eyeVec);
    //vec3 R = reflect(-L1, normalDirection);
    vec3 R = reflect(-L1, normal);
    float specular = pow( max(dot(R, E), 0.0),
        gl_FrontMaterial.shininess );
    color += vec3(gl_LightSource[1].specular) *
        vec3(gl_FrontMaterial.specular) *
        specular;

    // Set the output color of our current pixel
    gl_FragColor = vec4(color, 1.0);
}
'''])

shader1 = Shader(['''
#version 120
varying vec3 lightDir0, lightDir1, eyeVec;
varying vec3 normal, tangent, binormal;

void main()
{
// Create the Texture Space Matrix
    normal   = normalize(gl_NormalMatrix * gl_Normal);
    tangent  = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5));
    binormal = cross(normal, tangent);
    mat3 TBNMatrix = mat3(tangent, binormal, normal);

    vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

    lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex) * TBNMatrix;
    lightDir1 = vec3(gl_LightSource[1].position.xyz - vVertex) * TBNMatrix;
    eyeVec    = -vVertex * TBNMatrix;

    gl_Position =gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_TexCoord[0]  = gl_TextureMatrix[0] * gl_MultiTexCoord0;
}
'''], ['''
#version 120
varying vec3 normal, lightDir0, lightDir1, eyeVec;
//uniform sampler2D my_color_texture[+str(texturecnt)+]; //0 = ColorMap, 1 = NormalMap
uniform sampler2D color_texture;
uniform sampler2D normal_texture;
uniform int toggletexture = 1; // false/true
int togglebump = 1;    // false/true

void main (void)
{
        vec4 texColor = vec4(texture2D(color_texture, gl_TexCoord[0].st).rgb, 1.0);
        vec3 norm     = normalize( texture2D(normal_texture, gl_TexCoord[0].st).rgb - 0.5);
        //vec3 norm     = normalize( texture2D(normal_texture, gl_TexCoord[0].st).rgb * 2. - 1.);
        //vec3 norm     = normalize( texture2D(normal_texture, gl_TexCoord[0].st).rgb);

        if ( toggletexture == 0 ) texColor = gl_FrontMaterial.ambient;
        vec4 final_color = (gl_FrontLightModelProduct.sceneColor * vec4(texColor.rgb,1.0)) +
        //vec4 final_color = (.3* vec4(texColor.rgb,1.0)) +
    //(.3*gl_LightSource[0].ambient * vec4(texColor.rgb,1.0)) +
    (gl_LightSource[1].ambient * vec4(texColor.rgb,1.0));

    //vec3 N = (togglebump != 0) ? normalize(norm) : vec3(0.0, 0.0, 1.0 );
    vec3 N = (togglebump != 0) ? normalize(norm) : vec3(0.0, 1.0, 0.0 );
    vec3 L0 = normalize(lightDir0);
    vec3 L1 = normalize(lightDir1);

    float lambertTerm0 = .0 * dot(N,L0);
    float lambertTerm1 = 1. * dot(N,L1);

    if(lambertTerm0 > 0.0)
    {
        final_color += gl_LightSource[0].diffuse *
                       //gl_FrontMaterial.diffuse *
                       lambertTerm0;

        vec3 E = normalize(eyeVec);
        vec3 R = reflect(-L0, N);
        float specular = pow( max(dot(R, E), 0.0),
                         gl_FrontMaterial.shininess );
        final_color += gl_LightSource[0].specular *
                       //gl_FrontMaterial.specular *
                       specular;
    }
    if(lambertTerm1 > 0.0)
    {
        final_color += gl_LightSource[1].diffuse *
                       gl_FrontMaterial.diffuse *
                       lambertTerm1;

        vec3 E = normalize(eyeVec);
        vec3 R = reflect(-L1, N);
        float specular = pow( max(dot(R, E), 0.0),
                         gl_FrontMaterial.shininess );
        final_color += gl_LightSource[1].specular *
                       gl_FrontMaterial.specular *
                       specular;
    }
    //if (final_color.r > 0.1)
    //    discard;
    gl_FragColor = final_color;
}
'''])

#http://en.wikibooks.org/wiki/GLSL_Programming/Blender/Lighting_of_Bumpy_Surfaces
shader = Shader(['''
#version 120
//attribute vec4 tangent;
 
varying mat3 localSurface2View; // mapping from 
   // local surface coordinates to view coordinates
varying vec4 texCoords; // texture coordinates
varying vec4 position; // position in view coordinates

void main()
{
    //vec3 tangent = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5)); //maybe
    //vec3 tangent = vec3(0,0,1);
    vec3 tangent = normalize(cross(vec3(0,1,0), gl_Normal.xyz));
    // the signs and whether tangent is in localSurface2View[1]
    // or localSurface2View[0] depends on the tangent
    // attribute, texture coordinates, and the encoding
    // of the normal map 
    //localSurface2View[0] = normalize(vec3(gl_ModelViewMatrix * vec4(vec3(tangent), 0.0)));
    //localSurface2View[0]= vec3(1,0,0);
    //localSurface2View[2] = normalize(gl_NormalMatrix * gl_Normal);
    //localSurface2View[1] = normalize(cross(localSurface2View[2], localSurface2View[0]));
    
    localSurface2View[2] = normalize(gl_NormalMatrix * gl_Normal);
    localSurface2View[0] = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5));
    localSurface2View[1] = cross(localSurface2View[2], localSurface2View[0]);
    //mat3 TBNMatrix = mat3(tangent, binormal, normal);
    
    texCoords = gl_MultiTexCoord0;
    position = gl_ModelViewMatrix * gl_Vertex;            
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
'''], ['''
#version 120
varying mat3 localSurface2View; // mapping from 
// local surface coordinates to view coordinates
varying vec4 texCoords; // texture coordinates
varying vec4 position; // position in view coordinates

uniform sampler2D color_texture;
uniform sampler2D normal_texture;
uniform int toggletexture; // false/true

void main()
{
    //vec4 texColor = vec4(texture2D(color_texture, gl_TexCoord[0].st).rgb, 1.0);
    vec4 texColor =  texture2D(color_texture, vec2(texCoords));
    // in principle we have to normalize the columns of 
    // "localSurface2View" again; however, the potential 
    // problems are small since we use this matrix only
    // to compute "normalDirection", which we normalize anyways
    
    if (toggletexture == 0)
        texColor = vec4(0.75, 0.75, 0.75, 1.0);//gl_FrontMaterial.ambient;
    
    vec4 encodedNormal = texture2D(normal_texture, vec2(texCoords)); 
    
    //vec3 localCoords = normalize( texture2D(normal_texture, gl_TexCoord[0].st).rgb * 2. - 1.);
    //vec3 localCoords   = normalize( texture2D(normal_texture, gl_TexCoord[0].st).rgb);
    //vec3 localCoords = normalize( texture2D(normal_texture, gl_TexCoord[0].st).rgb - 0.5);
    vec3 localCoords = normalize(vec3(2.0, 2.0, 1.0) * vec3(encodedNormal) - vec3(1.0, 1.0, 0.0)); 
       // constants depend on encoding 
    vec3 normalDirection = normalize(localSurface2View * localCoords);
    
    // Compute per-pixel Phong lighting with normalDirection
    
    vec3 viewDirection = -normalize(vec3(position)); 
    vec3 lightDirection;
    float attenuation;
    if (0.0 == gl_LightSource[1].position.w) 
       // directional light?
    {
       attenuation = 1.0; // no attenuation
       lightDirection = 
          normalize(vec3(gl_LightSource[1].position));
    } 
    else // point light or spotlight (or other kind of light) 
    {
       vec3 positionToLightSource = 
          vec3(gl_LightSource[1].position - position);
       float distance = length(positionToLightSource);
       attenuation = 1.0 / distance; // linear attenuation 
       lightDirection = normalize(positionToLightSource);
    
       if (gl_LightSource[1].spotCutoff <= 90.0) // spotlight?
       {
          float clampedCosine = max(0.0, dot(-lightDirection, 
             gl_LightSource[1].spotDirection));
          if (clampedCosine < gl_LightSource[1].spotCosCutoff) 
             // outside of spotlight cone?
          {
             attenuation = 0.0;
          }
          else
          {
             attenuation = attenuation * pow(clampedCosine, 
                gl_LightSource[1].spotExponent);   
          }
       }
    }
    
    vec3 ambientLighting = vec3(gl_LightModel.ambient) 
       * vec3(texColor);//* vec3(gl_FrontMaterial.emission);
    
    vec3 diffuseReflection = 1.//attenuation 
       * vec3(gl_LightSource[1].diffuse)
       * vec3(texColor)//* vec3(gl_FrontMaterial.emission)       
       * max(0.0, dot(normalDirection, lightDirection));
    
    vec3 specularReflection;
    if (dot(normalDirection, lightDirection) < 0.0) 
       // light source on the wrong side?
    {
       specularReflection = vec3(0.0, 0.0, 0.0); 
          // no specular reflection
    }
    else // light source on the right side
    {
       specularReflection = attenuation 
          * vec3(gl_LightSource[1].specular) 
          * vec3(gl_FrontMaterial.specular) 
          * pow(max(0.0, dot(reflect(-lightDirection, 
          normalDirection), viewDirection)), 
          gl_FrontMaterial.shininess);
    }
    
    gl_FragColor = vec4(
        ambientLighting *
        1.0+//vec3(texColor) +// here?
        diffuseReflection *
        1.0//vec3(texColor) // here?
        + specularReflection
        , 1.0);
     //gl_FragColor = vec4(vec3(texColor), 1.0);
}
'''])

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def chunks(seq, n):
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def get_colors1(img, format, pitch, default_value=np.uint8(255)):
    l = np.fromstring(img.data, dtype=np.uint8)
    iformat = img.format
    nb_colors = len(iformat)
    l.shape = (len(l) // nb_colors, nb_colors)
    nb_pixels = l.shape[0]
    nb_final_colors = len(format)
    res = np.empty(nb_pixels * nb_final_colors, dtype=np.uint8)
    for i, f in enumerate(format):
        try:
            ic = iformat.index(f)
            c = l[:, ic]
            res[i::nb_final_colors] = c
        except ValueError:
            res[i::nb_final_colors] = default_value
    #print "%.3f enumerate done" % time.clock()
    #return res.tostring()
    if pitch > 0:
        return res.tostring()
    else:
        return ''.join(''.join(c) for c in izip(chunks(res.tostring(), img.width*4)[::-1]))

def get_colors(img, format, pitch, default_value='\xff'):
    l = img.data
    iformat = img.format
    nb_colors = len(iformat)
    old_colors = [l[i::nb_colors] for i in xrange(nb_colors)]
    print  "%.3f % old_colors done" % time.clock()
    nb_pixels = len(old_colors[0])
    new_colors = []
    for f in format:
        try:
            ic = iformat.index(f)
            c = old_colors[ic]
        except ValueError:
            c = default_value*nb_pixels
        new_colors.append(c)
    #return ''.join(''.join(c) for c in izip(*new_colors))
    if pitch > 0:
        res = ''.join(''.join(c) for c in izip(*new_colors))
    else:
        temp = ''.join(''.join(c) for c in izip(*new_colors))

        #don't work
        #res = ''.join(''.join(reversed(chunk)) for chunk in chunks(temp.split(), img.width*4))

        #slow
        #res =''
        #linesz = img.width*4
        #for i in range(len(temp)-linesz,-linesz,-linesz):
        #    res=res+temp[i:i+linesz]

        print "%.3f swap done" % time.clock()
        res = ''.join(''.join(c) for c in izip(chunks(temp, img.width*4)[::-1]))
    return res


##################################MyImage
class MyImage:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self,x,y,imagedata,texturedata):
        self.x = x
        self.y = y
        self.imagedata = imagedata
        self.texturedata = texturedata

##################################World
class World(pyglet.window.Window):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self):
        config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True,)
        try:
            super(World, self).__init__(resizable=True, config=config)
        except:
            super(World, self).__init__(resizable=True)
        self.setup()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup(self):
        self._width = 800        
        self._height = 600
        self.set_size(self._width,self._height)
        self.InitGL(self._width, self._height)
        pyglet.clock.schedule_interval(self.update, 1/600.0) # update at 60Hz
        self.listId = -1
        self.listIds = None
        self.angle = 0        
        self.rotateSpeed = 1
        self.texturesOn = 1
        self.normalMapOn = 0
        self.g_nFPS = 0
        self.g_nFrames = 0                                        # FPS and FPS Counter
        self.g_dwLastFPS = 0                                    # Last FPS Check Time
        self.myimage1 = None
        self.texturesList = None
        self.set_vsync(False)
        self.camHeight = -7.0
        self.camDistance = -20.0

        print "%.3f setup end" % time.clock()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self,dt):
        #self.DrawGLScene()
        pass #don't call self.DrawGLScene since it's already called in on_draw

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_draw(self):
        self.DrawGLScene()        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_resize(self,w,h):
        self.ReSizeGLScene(w,h)        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def vec(self,*args):
        #creates a c_types vector
        return (GLfloat * len(args))(*args)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # A general OpenGL initialization function.  Sets all of the initial parameters.
    def InitGL(self,Width, Height):             # We call this right after our OpenGL window is created.
        glClearColor(0.3, 0.3, 0.5, 0.0)       # This Will Clear The Background Color To Black        
        glClearDepth(1.0)                         # Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)                      # The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST)                 # Enables Depth Testing
        glShadeModel(GL_SMOOTH)             # Enables Smooth Color Shading
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()                          # Reset The Projection Matrix
                                                    # Calculate The Aspect Ratio Of The Window
        #(pyglet initializes the screen so we ignore this call)
        #gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        #glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )    #wireframe
        #glEnable(GL_CULL_FACE);
        #glCullFace(GL_FRONT) #or GL_BACK or even GL_FRONT_AND_BACK

        
        #self.LightPosition = self.vec(0.0, 1.0, 2.0, 1.0 )        

        glLightfv(GL_LIGHT1, GL_AMBIENT, self.vec(0.2, 0.2, 0.2, 1.0))  # add lighting. (ambient)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, self.vec(0.7, 0.7, 0.7, 1.0))  # add lighting. (diffuse).
        glLightfv(GL_LIGHT1, GL_POSITION, self.vec(1.0, 1.0, 1.5, 0.0 )) # set light position.
        glLightfv(GL_LIGHT1, GL_SPECULAR, self.vec(1.0, 1.0, 1.0, 1.0))
        glEnable(GL_LIGHT1)                             # turn light 1 on.

        #glEnable(GL_LIGHT0)                              #Quick And Dirty Lighting (Assumes Light0 Is Set Up)
        
        glEnable(GL_LIGHTING)                            #Enable Lighting
        glEnable(GL_COLOR_MATERIAL)                      #Enable Material Coloring'
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.vec(0.5, 0.5, 0.5, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.vec(1, 1, 1, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)

        glEnable(GL_TEXTURE_2D)                     # Enable texture mapping.
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # // Enable Pointers
        glEnableClientState(GL_VERTEX_ARRAY);                      # // Enable Vertex Arrays
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);               # // Enable Texture Coord Arrays
        glEnableClientState(GL_NORMAL_ARRAY);                      # // Enable Normal Arrays

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ImageLoad(self,filename):
        pic = pyglet.image.load(filename) #, decoder=PNGImageDecoder())
        #print "%.3f load done" % time.clock()
        texture = pic.get_texture()
        #print "%.3f get_texture done" % time.clock()
        ix = pic.width
        iy = pic.height
        rawimage = pic.get_image_data()
        #print "%.3f get_image_data done" % time.clock()
        #format = 'RGBA' #'RGB' #'RGBA'
#         pitch = rawimage.width * len(format)
        #print pitch
        #imagedata2 = rawimage.get_data(format, -pitch)
        #imagedata = rawimage.get_data(rawimage._current_format, rawimage._current_pitch)
        #imagedata = texture.image_data._current_data
        imagedata2 = get_colors1(rawimage, 'RGBA', -rawimage._current_pitch)
        print "%.3f ImageLoad done %s" % (time.clock(), filename)
        return MyImage(ix,iy,imagedata2,texture)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ImageLoadPil(self,filename):
        im = Image.open(filename)
        try:
            # get image meta-data (dimensions) and data
            ix, iy, image = im.size[0], im.size[1], im.tostring("raw", "RGBA", 0, -1)
        except SystemError:
            # has no alpha channel, synthesize one, see the
            # texture module for more realistic handling
            ix, iy, image = im.size[0], im.size[1], im.tostring("raw", "RGBX", 0, -1)
        return MyImage(ix,iy,image,None)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ArrangeMaterials(self):
        global matlist, faceByMat, faceByMatUV
        faceByMat = []
        faceByMatUV = []
        
        for m in xrange(0, len(matlist)):
            faceByMat.append([])
            faceByMatUV.append([])
        
        for f in xrange(0, len(facemat)):
            faceByMat[facemat[f]].append(faceslist[f])
            faceByMatUV[facemat[f]].append(faceuv[f])
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def CreateDisplayLists(self):
        global faceByMat, faceByMatUV
        
        self.listIds = []
        
        for f in xrange(len(faceByMat)):
            #prepare arrays
            vertices = np.array(vertexlist)
            nfaces = np.array(faceByMat[f])
            #tri_index = nfaces.reshape( len(nfaces) * 3) 
            '''var = vertices[nfaces]
            #var = vertices[tri_index]
            var1 = var.reshape( -1)
    #         vertices_gl = (GLfloat * len(var1))(*var1)
            #vertices_gl = (GLfloat * len(vertexlist))(*vertexlist)
            #glVertexPointer(3, GL_FLOAT, 0, vertices_gl)
            #glVertexPointer(3, GL_FLOAT, 0, var1.tostring())
            
            vavertices = []        
            for k in xrange(0, len(var1)): 
                vavertices.append(var1[k])
            
            vavertices_gl = (GLfloat * len(vavertices))(*vavertices)
                    
            nfaceuv = np.array(faceuv)         
            var1 = nfaceuv.reshape( -1)
            
            vauv = []
            for k in xrange(0, len(var1)): 
                vauv.append(var1[k])
            
            vauv_gl = (GLfloat * len(vauv))(*vauv)'''
            
            print "%.3f before normals compute" % time.clock()
            #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
            norm = np.zeros( vertices.shape, dtype=vertices.dtype )
            #Create an indexed view into the vertex array using the array of three indices for triangles
            tris = vertices[nfaces]
            #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
            #n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )# n is now an array of normals per triangle. The length of each normal is dependent the vertices,
            n = np.cross( tris[::,2 ] - tris[::,0]  , tris[::,1 ] - tris[::,0] )# n is now an array of normals per triangle. The length of each normal is dependent the vertices,
            # we need to normalize these, so that our next step weights each normal equally.normalize_v3(n)        
            # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
            # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
            # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
            # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
            norm[ nfaces[:,0] ] += n
            norm[ nfaces[:,1] ] += n
            norm[ nfaces[:,2] ] += n
            normalize_v3(norm)
            
            self.listIds.append(glGenLists(1))                # Generate 2 Different Lists
            glNewList(self.listIds[f], GL_COMPILE)      # Start With The Box List
            #glColor3f(0.85, 0.75, 0.7)
            glBegin(GL_TRIANGLES)
            for k in xrange(len(faceByMat[f])):
                x1 = vertexlist[faceByMat[f][k][0]][0]
                y1 = vertexlist[faceByMat[f][k][0]][1]
                z1 = vertexlist[faceByMat[f][k][0]][2]
                x2 = vertexlist[faceByMat[f][k][1]][0]
                y2 = vertexlist[faceByMat[f][k][1]][1]
                z2 = vertexlist[faceByMat[f][k][1]][2]
                x3 = vertexlist[faceByMat[f][k][2]][0]
                y3 = vertexlist[faceByMat[f][k][2]][1]
                z3 = vertexlist[faceByMat[f][k][2]][2]
                
                glNormal3f(norm[faceByMat[f][k][0]][0], norm[faceByMat[f][k][0]][1], norm[faceByMat[f][k][0]][2]);
                glTexCoord2f(faceByMatUV[f][k][0][0], faceByMatUV[f][k][0][1])            
                glVertex3f(x1, y1, z1)
                
                glNormal3f(norm[faceByMat[f][k][1]][0], norm[faceByMat[f][k][1]][1], norm[faceByMat[f][k][1]][2]);
                glTexCoord2f(faceByMatUV[f][k][1][0], faceByMatUV[f][k][1][1])
                glVertex3f(x2, y2, z2)
    
                glNormal3f(norm[faceByMat[f][k][2]][0], norm[faceByMat[f][k][2]][1], norm[faceByMat[f][k][2]][2]);
                glTexCoord2f(faceByMatUV[f][k][2][0], faceByMatUV[f][k][2][1])            
                glVertex3f(x3, y3, z3)            
            glEnd()
            glEndList();

        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def LoadMaterials(self):        
        global matlist, dirname
        
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        
        self.texturesList = []
        
        for i in matlist:
            diffuse = None            
            if i[0] != '':
                diffuse = self.ImageLoad(dirname + i[0])
                glTexImage2D(GL_TEXTURE_2D, 0, 4, diffuse.x, diffuse.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, diffuse.imagedata)
                glBindTexture(diffuse.texturedata.target, diffuse.texturedata.id)
            
            normal = None
            if i[1] != '':
                normal = self.ImageLoad(dirname + i[1])
                glTexImage2D(GL_TEXTURE_2D, 0, 4, normal.x, normal.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, normal.imagedata)
                glBindTexture(normal.texturedata.target, normal.texturedata.id)
            self.texturesList.append([diffuse, normal])
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def LoadGLTextures(self):
        global idTexture
        self.myimage1 = self.ImageLoad(dirname + "Batman_V3_Body_D.png") #good v Pierre _current_pitch    int: -6144
#        self.myimage1 = self.ImageLoad(dirname + "Batman_V3_Body_D_.tga") #bad v Pierre _current_pitch    int: 6144
#        self.myimage1 = self.ImageLoad(dirname + "Batman_V3_head_D.png")
#        self.myimage1 = self.ImageLoad(dirname + "BodyNude_D.tga") #bad v Pierre _current_pitch    int: 6144

        #print "%.3f ImageLoaded" % time.clock()

        #Batman_Masked_Face_D_Sick1
        #Batman_Rabbit_Head_Eye_Glow_Alpha
        #Batman_V3_Body_D
        #Batman_V3_head_D
        #Batman_Rabbit_Head_D
        #V2_Batman_Cape_A2

#         idTexture = GLuint()
#         glGenTextures(1, byref(idTexture))
#         glBindTexture(GL_TEXTURE_2D, idTexture.value)

        # texture 1 (poor quality scaling)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)

        # 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
        # border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
        glTexImage2D(GL_TEXTURE_2D, 0, 4, self.myimage1.x, self.myimage1.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.myimage1.imagedata)
        glBindTexture(self.myimage1.texturedata.target, self.myimage1.texturedata.id)   # 2d texture (x and y size)
        #glBindTexture(GL_TEXTURE_2D, idTexture.value)   # 2d texture (x and y size)
        
        self.myimage2 = self.ImageLoad(dirname + "Batman_V3_Body_N.tga") #good v Pierre _current_pitch    int: -6144
        
        glTexImage2D(GL_TEXTURE_2D, 0, 4, self.myimage2.x, self.myimage2.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.myimage2.imagedata)
        glBindTexture(self.myimage2.texturedata.target, self.myimage2.texturedata.id)   # 2d texture (x and y size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
    def ReSizeGLScene(self,Width, Height):
        if Height == 0:                           # Prevent A Divide By Zero If The Window Is Too Small
            Height = 1
        glViewport(0, 0, Width, Height)     # Reset The Current Viewport And Perspective Transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def CreateDisplayList(self):
        global vavertices, vavertices_gl, var1, vauv, vauv_gl, vanorm, vanorm_gl
        
        #prepare arrays
        vertices = np.array(vertexlist)
        nfaces = np.array(faceslist)
        #tri_index = nfaces.reshape( len(nfaces) * 3) 
        var = vertices[nfaces]
        #var = vertices[tri_index]
        var1 = var.reshape( -1)
#         vertices_gl = (GLfloat * len(var1))(*var1)
        #vertices_gl = (GLfloat * len(vertexlist))(*vertexlist)
        #glVertexPointer(3, GL_FLOAT, 0, vertices_gl)
        #glVertexPointer(3, GL_FLOAT, 0, var1.tostring())
        
        vavertices = []        
        for k in xrange(0, len(var1)): 
            vavertices.append(var1[k])
        
        vavertices_gl = (GLfloat * len(vavertices))(*vavertices)
                
        nfaceuv = np.array(faceuv)         
        var1 = nfaceuv.reshape( -1)
        
        vauv = []
        for k in xrange(0, len(var1)): 
            vauv.append(var1[k])
        
        vauv_gl = (GLfloat * len(vauv))(*vauv)
        
        print "%.3f before normals compute" % time.clock()
        #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros( vertices.shape, dtype=vertices.dtype )
        #Create an indexed view into the vertex array using the array of three indices for triangles
        tris = vertices[nfaces]
        #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
        #n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )# n is now an array of normals per triangle. The length of each normal is dependent the vertices,
        n = np.cross( tris[::,2 ] - tris[::,0]  , tris[::,1 ] - tris[::,0] )# n is now an array of normals per triangle. The length of each normal is dependent the vertices,
        # we need to normalize these, so that our next step weights each normal equally.normalize_v3(n)        
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norm[ nfaces[:,0] ] += n
        norm[ nfaces[:,1] ] += n
        norm[ nfaces[:,2] ] += n
        normalize_v3(norm)
        
        norm1 = norm[nfaces]
        
        var1 = norm1.reshape( -1)
        
        vanorm = []
        for k in xrange(0, len(var1)): 
            vanorm.append(var1[k])
        
        vanorm_gl = (GLfloat * len(vanorm))(*vanorm)
        
        print "%.3f after normals compute" % time.clock()
        
        #prepare display list
        self.listId = glGenLists(1)                # Generate 2 Different Lists
        glNewList(self.listId,GL_COMPILE)      # Start With The Box List
        #glColor3f(0.85, 0.75, 0.7)
        glBegin(GL_TRIANGLES)
        for k in xrange(0, len(faceslist)):
            #normal by face
            x1 = vertexlist[faceslist[k][0]][0]
            y1 = vertexlist[faceslist[k][0]][1]
            z1 = vertexlist[faceslist[k][0]][2]
            x2 = vertexlist[faceslist[k][1]][0]
            y2 = vertexlist[faceslist[k][1]][1]
            z2 = vertexlist[faceslist[k][1]][2]
            x3 = vertexlist[faceslist[k][2]][0]
            y3 = vertexlist[faceslist[k][2]][1]
            z3 = vertexlist[faceslist[k][2]][2]
            '''ux = x2 - x1
            uy = y2 - y1
            uz = z2 - z1
            vx = x3 - x1
            vy = y3 - y1
            vz = z3 - z1
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            l = math.sqrt(nx * nx + ny * ny + nz * nz)
            if l == 0:
                l = 1
            nx /= l
            ny /= l
            nz /= l'''

            #with numpy
            '''p1 = [x1, y1, z1]
            p2 = [x2, y2, z2]
            p3 = [x3, y3, z3]
            u = np.subtract(p2, p1)
            v = np.subtract(p3, p1)
            n = np.cross(u,v)
            n = n/np.linalg.norm(n)'''
            #normalize ? http://stackoverflow.com/questions/12049154/python-numpy-vector-math
            #http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors

            #to test (smooth normals by vertex ?): https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy

            #glNormal3f(nx, ny, nz)
            #glNormal3f(*n)
            glNormal3f(norm[faceslist[k][0]][0], norm[faceslist[k][0]][1], norm[faceslist[k][0]][2]);
            glTexCoord2f(faceuv[k][0][0], faceuv[k][0][1])
            #glColor3f(0.85, 0.75, 0.7)
            glVertex3f(x1, y1, z1)
            #glVertex3f(*p1)

            #glNormal3f( nx, ny, nz)
            glNormal3f(norm[faceslist[k][1]][0], norm[faceslist[k][1]][1], norm[faceslist[k][1]][2]);
            glTexCoord2f(faceuv[k][1][0], faceuv[k][1][1])
            #glColor3f(1, 0.66, 0.67)
            glVertex3f(x2, y2, z2)
            #glVertex3f(*p2)

            #glNormal3f( nx, ny, nz)
            glNormal3f(norm[faceslist[k][2]][0], norm[faceslist[k][2]][1], norm[faceslist[k][2]][2]);
            glTexCoord2f(faceuv[k][2][0], faceuv[k][2][1])
            #glColor3f(0.75, 0.65, 0.6)
            glVertex3f(x3, y3, z3)
            #glVertex3f(*p3)
        glEnd()
        glEndList();

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display using only one material
    def DrawOneMaterial(self):
        global vavertices, vavertices_gl, var1, vauv, vauv_gl, vanorm, vanorm_gl
        
        if self.myimage1 == None:
            self.CreateDisplayList()
            print "%.3f after display list" % time.clock()
            self.LoadGLTextures()
            print "%.3f after texture loading" % time.clock()
            glVertexPointer(3, GL_FLOAT, 0, vavertices_gl)
            glTexCoordPointer(2, GL_FLOAT, 0, vauv_gl)
            glNormalPointer(GL_FLOAT, 0, vanorm_gl)
        
        ####### one material method #######
        glBindTexture(GL_TEXTURE_2D, self.myimage1.texturedata.id)            
        if self.normalMapOn == 1: 
            shader.bind()
            shader.uniformi('toggletexture', self.texturesOn)
            glActiveTexture(GL_TEXTURE0)
            shader.uniformi('color_texture', 0)            
            glActiveTexture(GL_TEXTURE1)
            shader.uniformi('normal_texture', 1)
            glBindTexture(GL_TEXTURE_2D, self.myimage2.texturedata.id)
        
        #glCallList(self.listId)

        glDrawArrays(GL_TRIANGLES, 0, len(vavertices) // 3)
        #glVertexPointer(3, GL_FLOAT, 0, var1.tostring())
        
        if self.normalMapOn == 1:
            glActiveTexture(GL_TEXTURE1)
            glDisable(GL_TEXTURE_2D)
            glActiveTexture(GL_TEXTURE0)
            #glDisable(GL_TEXTURE_2D)            
            shader.unbind()            
        ####### one material method end #######
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display using only one material
    def DrawMultipleMaterials(self):        
        if self.texturesList == None:
            self.ArrangeMaterials()
            self.CreateDisplayLists()
            print "%.3f after display list" % time.clock()
            self.LoadMaterials()
            print "%.3f after texture loading" % time.clock()
        
        ####### multiple materials method #######
        for m in xrange(len(self.listIds)):
            if self.texturesList[m][0] != None:
                glBindTexture(GL_TEXTURE_2D, self.texturesList[m][0].texturedata.id)            
            if self.normalMapOn == 1: 
                shader.bind()
                shader.uniformi('toggletexture', self.texturesOn)
                glActiveTexture(GL_TEXTURE0)
                shader.uniformi('color_texture', 0)                
                glActiveTexture(GL_TEXTURE1)
                shader.uniformi('normal_texture', 1)
                #glBindTexture(GL_TEXTURE_2D, self.myimage2.texturedata.id)
                if self.texturesList[m][1] != None:
                    glBindTexture(GL_TEXTURE_2D, self.texturesList[m][1].texturedata.id)
            
            glCallList(self.listIds[m])
            
            if self.normalMapOn == 1:
                glActiveTexture(GL_TEXTURE1)
                glDisable(GL_TEXTURE_2D)
                glActiveTexture(GL_TEXTURE0)
                #glDisable(GL_TEXTURE_2D)            
                shader.unbind()
        ####### multiple materials method end #######


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The main drawing function.
    def DrawGLScene(self):        
        milliseconds = time.clock () * 1000.0
        if (milliseconds - self.g_dwLastFPS >= 1000):                    # // When A Second Has Passed...
            # g_dwLastFPS = win32api.GetTickCount();                # // Update Our Time Variable
            self.g_dwLastFPS = time.clock () * 1000.0
            self.g_nFPS = self.g_nFrames;                                        # // Save The FPS
            self.g_nFrames = 0;                                            # // Reset The FPS Counter
            # // Build The Title String
            szTitle = "%d FPS"  % self.g_nFPS;
            self.set_caption(szTitle)
            
        self.g_nFrames += 1                                                 # // Increment Our FPS Counter        
        
#         global idTexture
        # Clear The Screen And The Depth Buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()                  # Reset The View        

        # Move Left 1.5 units and into the screen 6.0 units.
        glTranslatef(0, self.camHeight, self.camDistance)
        self.angle += self.rotateSpeed
        glRotatef(self.angle, 0, 1, 0)
        
        #glRotatef(180, 0, 1, 0)
        #self.LightPosition = self.vec(math.cos(self.angle*3.14/180), 0.0, math.sin(self.angle*3.14/180), 0.0 )
        #glLightfv(GL_LIGHT1, GL_POSITION, self.LightPosition) # set light position.

#       glBindTexture(GL_TEXTURE_2D, idTexture.value)
        glColor4f(0.8, 0.8, 0.8, 1.)
        
        #self.DrawOneMaterial()
        self.DrawMultipleMaterials()
        
        # Since we have smooth color mode on, this will be great for the Phish Heads :-).
                # Draw a triangle
        '''glBegin(GL_POLYGON)                 # Start drawing a polygon
        glNormal3f(0, 0, 1)
        glColor3f(1.0, 0.0, 0.0)            # Red
        glVertex3f(0.0, 1.0, 0.0)           # Top
        glNormal3f(0, 1, 0)
        glColor3f(0.0, 1.0, 0.0)            # Green
        glVertex3f(1.0, -1.0, 0.0)          # Bottom Right
        glNormal3f(0, .7, .7)
        glColor3f(0.0, 0.0, 1.0)            # Blue
        glVertex3f(-1.0, -1.0, 0.0)         # Bottom Left
        glEnd()                             # We are done with the polygon'''

        # Move Right 3.0 units.
        glTranslatef(3.0, 0.0, 0.0)

        # Draw a square (quadrilateral)
        #glColor3f(0.3, 0.5, 1.0)            # Bluish shade
        glBegin(GL_QUADS)                   # Start drawing a 4 sided polygon
        glVertex3f(-1.0, 1.0, 0.0)          # Top Left
        glVertex3f(1.0, 1.0, 0.0)           # Top Right
        glVertex3f(1.0, -1.0, 0.0)          # Bottom Right
        glVertex3f(-1.0, -1.0, 0.0)         # Bottom Left
        glEnd()                             # We are done with the polygon


        #  since this is double buffered, swap the buffers to display what just got drawn.
        #(pyglet provides the swap, so we dont use the swap here)
        #glutSwapBuffers()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.dispatch_event('on_close')
        #enable or disable rotation
        if symbol == key.SPACE:
            if self.rotateSpeed == 1:
                self.rotateSpeed = 0
            else:
                self.rotateSpeed = 1
        #enable or disable textures
        if symbol == key.T:
            if self.texturesOn == 1:
                self.texturesOn = 0
                glDisable(GL_TEXTURE_2D)
            else:
                self.texturesOn = 1
                glEnable(GL_TEXTURE_2D)        
        if symbol == key.N:
            if self.normalMapOn == 1:
                self.normalMapOn = 0
            else:
                self.normalMapOn = 1
        if symbol == key.UP:
            self.camDistance += 1.0
        if symbol == key.DOWN:
            self.camDistance -= 1.0
        if symbol == key.PAGEUP:
            self.camHeight -= 1.0
        if symbol == key.PAGEDOWN:
            self.camHeight += 1.0

# bonesize = 1.0
# md5_bones=[]

def pskimport(infile):
    global vertexlist, faces, faceslist, UVCoords, faceuv, facemat, matlist, dirname
#     global DEBUGLOG
    #print ("--------------------------------------------------")
    #print ("---------SCRIPT EXECUTING PYTHON IMPORTER---------")
    #print ("--------------------------------------------------")
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

    #me_ob = bpy.data.meshes.new(objName)
    #print("objName:",objName)
    #print "New Mesh = " + me_ob.name + "\n"
    #read general header
    indata = unpack('20s3i',pskfile.read(32))
    #not using the general header at this time
    #==================================================================================================
    # vertex point
    #==================================================================================================
    #read the PNTS0000 header
    indata = unpack('20s3i',pskfile.read(32))
    recCount = indata[3]
    print "Nbr of PNTS0000 records: " + str(recCount)
    counter = 0
    vertexlist = []
    while counter < recCount:
        counter = counter + 1
        indata = unpack('3f',pskfile.read(12))
        #print(indata[0],indata[1],indata[2])
        #vertexlist.extend([(indata[0] * scale ,indata[1] * scale ,indata[2] * scale)])
        vertexlist.extend([(-indata[0] * scale ,indata[2] * scale ,indata[1] * scale)])
        #Tmsh.vertices.append(NMesh.Vert(indata[0],indata[1],indata[2]))

    #==================================================================================================
    # UV
    #==================================================================================================
    #read the VTXW0000 header
    indata = unpack('20s3i',pskfile.read(32))
    recCount = indata[3]
    print "Nbr of VTXW0000 records: " + str(recCount)
    counter = 0
    UVCoords = []
    #UVCoords record format = [index to PNTS, U coord, v coord]
    while counter < recCount:
        counter = counter + 1
        indata = unpack('hhffhh',pskfile.read(16))
        UVCoords.append([indata[0],indata[2],indata[3]])
#         print([indata[0],indata[2],indata[3]])
#         print([indata[1],indata[2],indata[3]])

    #==================================================================================================
    # Face
    #==================================================================================================
    #read the FACE0000 header
    indata = unpack('20s3i',pskfile.read(32))
    recCount = indata[3]
    print "Nbr of FACE0000 records: "+ str(recCount)
    #PSK FACE0000 fields: WdgIdx1|WdgIdx2|WdgIdx3|MatIdx|AuxMatIdx|SmthGrp
    #associate MatIdx to an image, associate SmthGrp to a material
    SGlist = []
    counter = 0
    faces = []
    faceslist=[]
    faceuv = []
    facemat = []
    while counter < recCount:
        counter = counter + 1
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
            print("smooth:",indata[5])
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
    indata = unpack('20s3i',pskfile.read(32))
    recCount = indata[3]
    print "Nbr of MATT0000 records: " +    str(recCount)
    #print " - Not importing any material data now. PSKs are texture wrapped! \n"
    matlist = []
    counter = 0
    while counter < recCount:
        matlist.append('')
        indata = unpack('64s6i',pskfile.read(88))
        print counter, indata[0].rstrip('\x00')
        
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
        counter = counter + 1
    print matlist
    ##

    #==================================================================================================
    # Bones (Armature)
    #==================================================================================================
    #read the REFSKEL0 header
#     indata = unpack('20s3i',pskfile.read(32))
#     recCount = indata[3]
#     print "Nbr of REFSKEL0 records: " + str(recCount) + "\n"
#     Bns = []
#     bone = []
#     nobone = 0
    #==================================================================================================
    # Bone Data
    #==================================================================================================
#     counter = 0
#     print ("---PRASE--BONES---")
#     while counter < recCount:
#         indata = unpack('64s3i11f',pskfile.read(120))
#         #print( "DATA",str(indata))
#         bone.append(indata)
#
#         createbone = md5_bone()
#         #temp_name = indata[0][:30]
#         temp_name = indata[0]
#
#         temp_name = bytes.decode(temp_name)
#         temp_name = temp_name.lstrip(" ")
#         temp_name = temp_name.rstrip(" ")
#         temp_name = temp_name.strip()
#         temp_name = temp_name.strip( bytes.decode(b'\x00'))
#         print ("temp_name:", temp_name, "||")
#         createbone.name = temp_name
#         createbone.bone_index = counter
#         createbone.parent_index = indata[3]
#         createbone.bindpos[0] = indata[8]
#         createbone.bindpos[1] = indata[9]
#         createbone.bindpos[2] = indata[10]
#         createbone.scale[0] = indata[12]
#         createbone.scale[1] = indata[13]
#         createbone.scale[2] = indata[14]
#
#         #w,x,y,z
#         if (counter == 0):#main parent
#             print("no parent bone")
#             createbone.bindmat = mathutils.Quaternion((indata[7],indata[4],indata[5],indata[6]))
#             #createbone.bindmat = mathutils.Quaternion((indata[7],-indata[4],-indata[5],-indata[6]))
#         else:#parent
#             print("parent bone")
#             createbone.bindmat = mathutils.Quaternion((indata[7],-indata[4],-indata[5],-indata[6]))
#             #createbone.bindmat = mathutils.Quaternion((indata[7],indata[4],indata[5],indata[6]))
#
#         md5_bones.append(createbone)
#         counter = counter + 1
#         bnstr = (str(indata[0]))
#         Bns.append(bnstr)
#
#     for pbone in md5_bones:
#         pbone.parent =    md5_bones[pbone.parent_index].name
#
#     bonecount = 0
#     for armbone in bone:
#         temp_name = armbone[0][:30]
#         #print ("BONE NAME: ",len(temp_name))
#         temp_name=str((temp_name))
#         #temp_name = temp_name[1]
#         #print ("BONE NAME: ",temp_name)
#         bonecount +=1
#     print ("-------------------------")
#     print ("----Creating--Armature---")
#     print ("-------------------------")

    #================================================================================================
    #Check armature if exist if so create or update or remove all and addnew bone
    #================================================================================================
    #bpy.ops.object.mode_set(mode='OBJECT')
#     meshname ="ArmObject"
#    objectname = "armaturedata"
#     bfound = False
#     arm = None
#     for obj in bpy.data.objects:
#         if (obj.name == meshname):
#             bfound = True
#             arm = obj
#             break

#     if bfound == False:
#         armdata = bpy.data.armatures.new(objectname)
#         ob_new = bpy.data.objects.new(meshname, armdata)
        #ob_new = bpy.data.objects.new(meshname, 'ARMATURE')
        #ob_new.data = armdata
#         bpy.context.scene.objects.link(ob_new)
        #bpy.ops.object.mode_set(mode='OBJECT')
#         for i in bpy.context.scene.objects: i.select = False #deselect all objects
#         ob_new.select = True
        #set current armature to edit the bone
#         bpy.context.scene.objects.active = ob_new
        #set mode to able to edit the bone
#         bpy.ops.object.mode_set(mode='EDIT')
        #newbone = ob_new.data.edit_bones.new('test')
        #newbone.tail.y = 1
#         print("creating bone(s)")
#         for bone in md5_bones:
#             #print(dir(bone))
#             newbone = ob_new.data.edit_bones.new(bone.name)
#             #parent the bone
#             parentbone = None
#             print("bone name:",bone.name)
#             #note bone location is set in the real space or global not local
#             if bone.name != bone.parent:
#
#                 pos_x = bone.bindpos[0]
#                 pos_y = bone.bindpos[1]
#                 pos_z = bone.bindpos[2]
#
#                 #print( "LINKING:" , bone.parent ,"j")
#                 parentbone = ob_new.data.edit_bones[bone.parent]
#                 newbone.parent = parentbone
#                 rotmatrix = bone.bindmat.to_matrix().resize4x4().rotation_part()
#
#                 #parent_head = parentbone.head * parentbone.matrix.to_quat().inverse()
#                 #parent_tail = parentbone.tail * parentbone.matrix.to_quat().inverse()
#                 #location=Vector(pos_x,pos_y,pos_z)
#                 #set_position = (parent_tail - parent_head) + location
#                 #print("tmp head:",set_position)
#
#                 #pos_x = set_position.x
#                 #pos_y = set_position.y
#                 #pos_z = set_position.z
#
#                 newbone.head.x = parentbone.head.x + pos_x
#                 newbone.head.y = parentbone.head.y + pos_y
#                 newbone.head.z = parentbone.head.z + pos_z
#                 print("head:",newbone.head)
#                 newbone.tail.x = parentbone.head.x + (pos_x + bonesize * rotmatrix[1][0])
#                 newbone.tail.y = parentbone.head.y + (pos_y + bonesize * rotmatrix[1][1])
#                 newbone.tail.z = parentbone.head.z + (pos_z + bonesize * rotmatrix[1][2])
#             else:
#                 rotmatrix = bone.bindmat.to_matrix().resize4x4().rotation_part()
#                 newbone.head.x = bone.bindpos[0]
#                 newbone.head.y = bone.bindpos[1]
#                 newbone.head.z = bone.bindpos[2]
#                 newbone.tail.x = bone.bindpos[0] + bonesize * rotmatrix[1][0]
#                 newbone.tail.y = bone.bindpos[1] + bonesize * rotmatrix[1][1]
#                 newbone.tail.z = bone.bindpos[2] + bonesize * rotmatrix[1][2]
#                 #print("no parent")
#
#     bpy.context.scene.update()

    #==================================================================================================
    #END BONE DATA BUILD
    #==================================================================================================
#     VtxCol = []
#     for x in range(len(Bns)):
#         #change the overall darkness of each material in a range between 0.1 and 0.9
#         tmpVal = ((float(x)+1.0)/(len(Bns))*0.7)+0.1
#         tmpVal = int(tmpVal * 256)
#         tmpCol = [tmpVal,tmpVal,tmpVal,0]
#         #Change the color of each material slightly
#         if x % 3 == 0:
#             if tmpCol[0] < 128: tmpCol[0] += 60
#             else: tmpCol[0] -= 60
#         if x % 3 == 1:
#             if tmpCol[1] < 128: tmpCol[1] += 60
#             else: tmpCol[1] -= 60
#         if x % 3 == 2:
#             if tmpCol[2] < 128: tmpCol[2] += 60
#             else: tmpCol[2] -= 60
#         #Add the material to the mesh
#         VtxCol.append(tmpCol)

    #==================================================================================================
    # Bone Weight
    #==================================================================================================
    #read the RAWW0000 header
#     indata = unpack('20s3i',pskfile.read(32))
#     recCount = indata[3]
#     print "Nbr of RAWW0000 records: " + str(recCount) +"\n"
#     #RAWW0000 fields: Weight|PntIdx|BoneIdx
#     RWghts = []
#     counter = 0
#     while counter < recCount:
#         counter = counter + 1
#         indata = unpack('fii',pskfile.read(12))
#         RWghts.append([indata[1],indata[2],indata[0]])
#     #RWghts fields = PntIdx|BoneIdx|Weight
#     RWghts.sort()
#     print "len(RWghts)=" + str(len(RWghts)) + "\n"
    #Tmsh.update()

    #set the Vertex Colors of the faces
    #face.v[n] = RWghts[0]
    #RWghts[1] = index of VtxCol
    """
    for x in range(len(Tmsh.faces)):
        for y in range(len(Tmsh.faces[x].v)):
            #find v in RWghts[n][0]
            findVal = Tmsh.faces[x].v[y].index
            n = 0
            while findVal != RWghts[n][0]:
                n = n + 1
            TmpCol = VtxCol[RWghts[n][1]]
            #check if a vertex has more than one influence
            if n != len(RWghts)-1:
                if RWghts[n][0] == RWghts[n+1][0]:
                    #if there is more than one influence, use the one with the greater influence
                    #for simplicity only 2 influences are checked, 2nd and 3rd influences are usually very small
                    if RWghts[n][2] < RWghts[n+1][2]:
                        TmpCol = VtxCol[RWghts[n+1][1]]
        Tmsh.faces[x].col.append(NMesh.Col(TmpCol[0],TmpCol[1],TmpCol[2],0))
    """
#     if (DEBUGLOG):
#         logf.close()
    #==================================================================================================
    #Building Mesh
    #==================================================================================================
    print "vertex:",len(vertexlist),"faces:",len(faces)
    #me_ob.vertices.add(len(vertexlist))
    #me_ob.faces.add(len(faces)//4)

    #me_ob.vertices.foreach_set("co", unpack_list(vertexlist))

    #me_ob.faces.foreach_set("vertices_raw", faces)
    #me_ob.faces.foreach_set("use_smooth", [False] * len(me_ob.faces))
    #me_ob.update()

    #===================================================================================================
    #UV Setup
    #===================================================================================================
#     texture = []
#     texturename = "text1"
    #print(dir(bpy.data))
#     if (len(faceuv) > 0):
#         uvtex = me_ob.uv_textures.new() #add one uv texture
#         for i, face in enumerate(me_ob.faces):
#             blender_tface= uvtex.data[i] #face
#             blender_tface.uv1 = faceuv[i][0] #uv = (0,0)
#             blender_tface.uv2 = faceuv[i][1] #uv = (0,0)
#             blender_tface.uv3 = faceuv[i][2] #uv = (0,0)
#         texture.append(uvtex)

        #for tex in me_ob.uv_textures:
            #print("mesh tex:",dir(tex))
            #print((tex.name))

    #===================================================================================================
    #Material Setup
    #===================================================================================================
#     materialname = "mat"
#     materials = []
#
#     matdata = bpy.data.materials.new(materialname)
#     #color is 0 - 1 not in 0 - 255
#     #matdata.mirror_color=(float(0.04),float(0.08),float(0.44))
#     matdata.diffuse_color=(float(0.04),float(0.08),float(0.44))#blue color
#     #print(dir(me_ob.uv_textures[0].data))
#     texdata = None
#     texIndex = len(bpy.data.textures) - 1
#     if (texIndex >= 0):
#         texdata = bpy.data.textures[len(bpy.data.textures)-1]
#         if (texdata != None):
#             #print(texdata.name)
#             #print(dir(texdata))
#             texdata.name = "texturelist1"
#             matdata.active_texture = texdata
#     materials.append(matdata)
    #matdata = bpy.data.materials.new(materialname)
    #materials.append(matdata)
    #= make sure the list isnt too big
    #for material in materials:
        #add material to the mesh list of materials
        #me_ob.materials.append(material)
    #===================================================================================================
    #
    #===================================================================================================
    #obmesh = bpy.data.objects.new(objName,me_ob)
    #check if there is a material to set to
    #if len(materials) > 0:
    #    obmesh.active_material = materials[0] #material setup tmp

    #bpy.context.scene.objects.link(obmesh)

    #bpy.context.scene.update()

    print ("PSK2Blender completed")
#End of def pskimport#########################

scale = .07

##################################main
if __name__ == "__main__":
    global plik,dirname#,basename
#     global model_id
    global vertexlist, faces, faceslist, UVCoords, faceuv    
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:        
        filename = "ABGame/6/Elrath_A_Type_Skm.psk"
        filename = "ABGame/7/Elrath_Bikini_Skm.psk"
        filename = "ABGame/8/Elrath_Skm.psk"
        filename = "ABGame/10/Rehiney_Skm.psk"
        filename = "Batman_Rabbit_Head_Posed/Batman_Rabbit_Head_Posed.psk"
        '''filename = "ABGame/15/Ridika_Bikini_Skm.psk"
        filename = "ABGame/16/Ridika_Skm.psk"
        filename = "ABGame/20/Valle_Bikini_Skm.psk"
        filename = "ABGame/21/Valle_Skm.psk"
        filename = "ABGame/23/Ridika02_Sm.pskx"'''
        
        #filename = "Batmobile_MESH_DAMAGE_TEST.pskx"
        #filename = "Talia_posed.psk"
        #filename = "DLC11_Batman_Inc_CV_pose.psk"
        #filename = "catwoman_posed.psk"
        #filename = "BatmanV3_Unmasked_Posed.psk"
        #filename = "Elrath_Bikini_Skm.psk"
    
    filename.replace('\\', '/')
    
    dirname = os.path.dirname(filename)
    if dirname != '':
        dirname += '/'
    
    pskimport(filename)    

#    print 'nb vertex', len(vertexlist), ', nb face', len(faces), ', uv list', len(UVCoords), ', faceuv', len(faceuv), ', facemat', len(facemat)
#    print vertexlist[0], faces[0], faces[1], faces[2]#, faces[0][0], faces[0][1], faces[0][2]
#     print vertexlist[faces[0][0]][0], vertexlist[faces[0][0]][1], vertexlist[faces[0][0]][2]
#    print faceuv[0]    

    window = World()    
    pyglet.app.run()