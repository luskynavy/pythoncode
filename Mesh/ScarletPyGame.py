#!/usr/bin/env python


import numpy as np
from itertools import izip

import math
import time
import sys
import os.path

#import pyglet
import struct
#from struct import *
#from pyglet.gl import *
#from pyglet import image #<==for image calls
#from pyglet.image.codecs.png import PNGImageDecoder
#from pyglet.window import key
#from OpenGL.GLUT import * #<<<==Needed for GLUT calls

#import OpenGL.WGL.EXT.swap_control
#import OpenGL.raw
#import OpenGL
#import OpenGL.WGL

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame#, pygame.image, pygame.key
from pygame.locals import *
#from opengl_tools import *
from collections import namedtuple

#from ctypes import *

#from shader import Shader

#model = "NPC_Gloria"
model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_ClubDancer03/NPC_ClubDancer03"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_ClubDancer02/NPC_ClubDancer02"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_ClubDancer04/NPC_ClubDancer04"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_ClubBanny/NPC_ClubBanny"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_NudeBeach/NPC_NudeBeach"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_ItemSEAL/NPC_ItemSEAL"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_ItemSEAL/NPC_ItemSEAL_Club"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Guild/NPC_Guild_Club"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_BeachGirl/NPC_BeachGirl"

#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_10/DE_BD_CS_10"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_29/DE_BD_CS_29"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_34/DE_BD_CS_34"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_N_000/DE_BD_N_000"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_N_00/DE_BD_N_00"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_N_001/DE_BD_N_001"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_N_01/DE_BD_N_01"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_N_02/DE_BD_N_02"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_N_03/DE_BD_N_03"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/PT/DE_PT_N_15/DE_PT_N_15" # yellow -> black
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/PT/DE_PT_N_13/DE_PT_N_13"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/PT/DE_PT_N_05/DE_PT_N_05"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/PT/DE_PT_N_14/DE_PT_N_14"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Monica/NPC_Monica"
model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Sati/NPC_Sati"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Yuna/NPC_Yuna"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Auction/NPC_Auction"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Bartender/NPC_Bartender"
 
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_32/DE_BD_CS_32"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_30/DE_BD_CS_30"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_20/DE_BD_CS_20"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_18/DE_BD_CS_18"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_17/DE_BD_CS_17"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_13/DE_BD_CS_13"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_12/DE_BD_CS_12"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_11/DE_BD_CS_11"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_09/DE_BD_CS_09"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_08/DE_BD_CS_08"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_06/DE_BD_CS_06"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_05/DE_BD_CS_05"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_04/DE_BD_CS_04"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_03/DE_BD_CS_03"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_02/DE_BD_CS_02"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/DE/BD/DE_BD_CS_01/DE_BD_CS_01"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/PC/WH/BD/WH_BD_NU_B_04"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/NPC_Knight_R/NPC_Knight_R"
#model = "../../../Mesh/Scarlet Blade/Quickbms/out/a/Objects/NPC/npc_Neirin_a/npc_Neirin_a"



#shader = Shader([
'''
varying vec3 colorL;
void main() {
gl_TexCoord[0] = gl_MultiTexCoord0;

vec3 normalDirection = normalize(gl_NormalMatrix * gl_Normal);
vec3 lightDirection = normalize(vec3(gl_LightSource[1].position));
//vec3 lightDirection = normalize(vec3(0.0, 1.0, 1.0));

vec3 diffuseReflection = 
   vec3(gl_LightSource[1].diffuse) 
   * 1.0 //* vec3(gl_FrontMaterial.emission)
   * max(0.0, dot(normalDirection, lightDirection));
   //* dot(normalDirection, lightDirection);

colorL = diffuseReflection; 

// Set the position of the current vertex
gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
'''
#], [
'''
varying vec3 colorL;
uniform sampler2D color_texture;
uniform sampler2D normal_texture;
uniform int toggletexture; // false/true

void main() {

// Extract the normal from the normal map
//vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rgb * 2.0 - 1.0);
//vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rga * 2.0 - 1.0);
vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rga - 0.5);
//vec3 normal = texture2D(normal_texture, gl_TexCoord[0].st).rgb;
//vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rgb);
//vec3 normal = normalize(texture2D(normal_texture, gl_TexCoord[0].st).rgb - 0.5);

// Determine where the light is positioned (this can be set however you like)
vec3 light_pos = normalize(vec3(1.0, 1.0, 1.5));

// Calculate the lighting diffuse value
//float diffuse = max(dot(normal, light_pos), 0.0);
float diffuse = dot(normal, colorL);

vec3 color = (toggletexture == 1
    ? diffuse * texture2D(color_texture, gl_TexCoord[0].st).rgb
    : diffuse * vec3(0.5, 0.5, 0.5));
//vec3 color = diffuse * .5 + diffuse * texture2D(color_texture, gl_TexCoord[0].st).rgb * 0.5;

//if (color.r > 0.3)
//    discard;

// Set the output color of our current pixel
gl_FragColor = vec4(color, 1.0);
}
'''#])

# create our Phong Shader by Jerome GUINOT aka 'JeGX' - jegx [at] ozone3d [dot] net
# see http://www.ozone3d.net/tutorials/glsl_lighting_phong.php

#shader1 = Shader([
'''
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

    gl_Position = ftransform();
        gl_TexCoord[0]  = gl_TextureMatrix[0] * gl_MultiTexCoord0;
}
'''
#], [
'''
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
'''
#])

#http://en.wikibooks.org/wiki/GLSL_Programming/Blender/Lighting_of_Bumpy_Surfaces
vertex_shader = '''
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
'''

fragment_shader = '''
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
    
    //vec3 localCoords = normalize( texture2D(normal_texture, vec2(texCoords)).rga * 2. - 1.);
    //vec3 localCoords   = normalize( texture2D(normal_texture,vec2(texCoords)).rga); //almost no bump with 15 yellow
    vec3 localCoords = normalize( texture2D(normal_texture, vec2(texCoords)).rga - 0.5);
    //vec3 localCoords = normalize(vec3(2.0, 2.0, 1.0) * vec3(encodedNormal) - vec3(1.0, 1.0, 0.0));
    //vec3 localCoords = normalize(vec3(2.0, 2.0, 1.0) * vec3(encodedNormal) - vec3(1.0, 1.0, 0.0)); 
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
        ambientLighting +        
        diffuseReflection +        
        + specularReflection
        , 1.0);
     //gl_FragColor = vec4(vec3(texColor), 1.0);
}
'''

ShaderProgram = namedtuple("ShaderProgram", "program uniforms attributes")

def assemble_shader_program(
        vertex_shader_source,
        fragment_shader_source,
        uniform_names,
        attribute_names):
    vertex_shader = make_shader(GL_VERTEX_SHADER, vertex_shader_source)
    fragment_shader = make_shader(GL_FRAGMENT_SHADER, fragment_shader_source)
    program = make_program(vertex_shader, fragment_shader)
    return ShaderProgram(
        program,
        get_uniforms(program, uniform_names),
        get_attributes(program, attribute_names))

def get_uniforms(program, names):
    return dict((name, glGetUniformLocation(program, name)) for name in names)


def get_attributes(program, names):
    return dict((name, glGetAttribLocation(program, name)) for name in names)


def make_shader(shadertype, source):
    shader = glCreateShader(shadertype)
    glShaderSource(shader, source)
    glCompileShader(shader)
    retval = ctypes.c_uint(GL_UNSIGNED_INT)
    glGetShaderiv(shader, GL_COMPILE_STATUS, retval)
    if not retval:
        print >> sys.stderr, "Failed to compile shader."
        print glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise Exception("Failed to compile shader.")
    return shader


def make_program(vertex_shader, fragment_shader):
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    retval = ctypes.c_int()
    glGetProgramiv(program, GL_LINK_STATUS, retval)
    if not retval:
        print >> sys.stderr, "Failed to link shader program."
        print glGetProgramInfoLog(program)
        glDeleteProgram(program)
        raise Exception("Failed to link shader program.")
    return program


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
    print "enumerate done", time.clock()
    #return res.tostring()
    if pitch > 0:
        return res.tostring()
    else:
        return ''.join(''.join(c) for c in izip(chunks(res.tostring(), img.width*4)[::-1]))
        
##################################Image
class MyImage:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self,x,y,imagedata,texturedata):
        self.x = x
        self.y = y
        self.imagedata = imagedata
        self.texturedata = texturedata

##################################World
#class World(pyglet.window.Window):
class World(): #pyglet.window.Window):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self):
        '''config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True,)
        try:
            super(World, self).__init__(resizable=True, config=config)
        except:
            super(World, self).__init__(resizable=True)'''
        self.setup()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup(self):
        '''self._width = 800        
        self._height = 600
        self.set_size(self._width,self._height)'''
        self.InitGL()#self._width, self._height)
        #self.InitGL(1280, 1024)
        #pyglet.clock.schedule_interval(self.update, 1/60.0) # update at 60Hz
        self.listId = 0
        self.angle = 0
        self.DisplayGlTriangles()
        self.LoadGLTextures()
        self.rotateSpeed = 1
        self.texturesOn = 1
        self.normalMapOn = 0
        self.camHeight = -10.0
        self.camDistance = -25.0
        self.g_nFPS = 0
        self.g_nFrames = 0                                        # FPS and FPS Counter
        self.g_dwLastFPS = 0                                    # Last FPS Check Time

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self,dt):
        #self.DrawGLScene()
        pass

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
    def InitGL(self):#,Width, Height):             # We call this right after our OpenGL window is created.
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
        gluPerspective(45.0,640/480.0,0.1,100.0)    #setup lens
        glMatrixMode(GL_MODELVIEW)

        self.LightAmbient = self.vec(0.5, 0.5, 0.5, 1.0)
        self.LightDiffuse = self.vec(1., 1., 1., 1.0)
        self.LightPosition = self.vec(0.0, 1.0, 2.0, 0.0 )

        glLightfv(GL_LIGHT1, GL_AMBIENT, self.vec(0.3, 0.3, 0.3, 1.0))  # add lighting. (ambient)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, self.vec(0.9, 0.9, 0.9, 1.0))  # add lighting. (diffuse).
        glLightfv(GL_LIGHT1, GL_POSITION, self.vec(1.0, 1.0, 1.5, 0.0)) # set light position.
        glLightfv(GL_LIGHT1, GL_POSITION, self.vec(0.0, 1.0, 2.0, 0.0)) # set light position.
        glLightfv(GL_LIGHT1, GL_SPECULAR, self.vec(1.0, 1.0, 1.0, 1.0))
        glEnable(GL_LIGHT1)                             # turn light 1 on.

        #glEnable(GL_LIGHT0)                              #Quick And Dirty Lighting (Assumes Light0 Is Set Up)
        
        glEnable(GL_LIGHTING)                            #Enable Lighting
        glEnable(GL_COLOR_MATERIAL)                      #Enable Material Coloring'
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.vec(0.5, 0.5, 0.5, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.vec(1, 1, 1, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)

        glEnable(GL_TEXTURE_2D)                     # Enable texture mapping.
        
        '''info = gl_info.GLInfo()
        info.set_active_context()
        print info.get_version()
        print info.get_vendor()
        print info.get_renderer()
        print 'GL_ARB_vertex_shader', info.have_extension('GL_ARB_vertex_shader')
        print 'GL_UNIFORM_BUFFER', info.have_extension('GL_UNIFORM_BUFFER')'''            
        
        self.program = assemble_shader_program(vertex_shader, fragment_shader,
                                          uniform_names=[
                                            'color_texture',
                                            'normal_texture',
                                            'toggletexture'],
                                          attribute_names=[])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ImageLoad(self,filename,reverse=False):
        image = pygame.image.load(filename)
        pixels = pygame.image.tostring(image, "RGBA", True)
        #print len(pixels)
        textureId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, textureId)
        print "get_data done", time.clock()
        return MyImage(image.get_width(),image.get_height(),pixels,textureId)
    
        '''pic = pyglet.image.load(filename) #, decoder=PNGImageDecoder())
        print "load done", time.clock()
        texture = pic.get_texture()        
        print "get_texture done", time.clock()
        ix = pic.width
        iy = pic.height
        #rawimage = pic.get_image_data()
        print "get_image_data done", time.clock()
        #format = 'RGBA' #'RGB' #'RGBA'
#       pitch = rawimage.width * len(format)
        print texture.image_data._current_pitch
        #imagedata2 = rawimage.get_data(format, -pitch)
        #imagedata = rawimage.get_data(rawimage._current_format, rawimage._current_pitch)
        #imagedata = texture.image_data._current_data
        if reverse == True:
            imagedata2 = get_colors1(texture.image_data, 'RGBA', -texture.image_data._current_pitch)
        else:
            imagedata2 = get_colors1(texture.image_data, 'RGBA', texture.image_data._current_pitch)
        print "get_data done", time.clock()
        return MyImage(ix,iy,imagedata2,texture)'''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def LoadGLTextures(self):
        global dirname, basename
        #load color texture
        #self.myimage1 = self.ImageLoad("NPC_Gloria.bmp")
        #self.myimage1 = self.ImageLoad("NPC_Gloria.png")
        #self.myimage1 = self.ImageLoad("NPC_ClubDancer02.png")
        #self.myimage1 = self.ImageLoad("NPC_Yuna.png")
        #self.myimage1 = self.ImageLoad("NPC_Sati.png")        
        try:
            self.myimage1 = self.ImageLoad(model + '.dds', True)
        except:
            try:
                self.myimage1 = self.ImageLoad(model + '.png')
            except:
                try:
                    self.myimage1 = self.ImageLoad(dirname + basename.split('.')[0].replace('NPC_','') + '.dds', True)
                except:
                    self.myimage1 = None #self.ImageLoad("NPC_Gloria.png")        

#         textures = c_uint()
#         texture = glGenTextures(1, byref(textures))
        if self.myimage1 != None:
            # texture 1 (poor quality scaling) GL_NEAREST
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)

            # 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
            # border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
            glTexImage2D(GL_TEXTURE_2D, 0, 4, self.myimage1.x, self.myimage1.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.myimage1.imagedata)
            glBindTexture(GL_TEXTURE_2D, self.myimage1.texturedata)   # 2d texture (x and y size)
            
        #load normal map
        try:
            self.myimage2 = self.ImageLoad(model + '_NR.dds', True)
        except:
            try:
                self.myimage2 = self.ImageLoad(model + '_NR.png')
            except:
                try:
                    self.myimage2 = self.ImageLoad(dirname + basename.split('.')[0].replace('NPC_','') + '_NR.dds', True)                    
                except:
                    self.myimage2 = None #self.ImageLoad("NPC_Gloria_NR.dds")

        if self.myimage2 != None:
#         glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)  # cheap scaling when image bigger than texture
#         glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)  # cheap scaling when image smalled than texture
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)  # cheap scaling when image bigger than texture
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)  # cheap scaling when image smalled than texture
                
            glTexImage2D(GL_TEXTURE_2D, 0, 4, self.myimage2.x, self.myimage2.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.myimage2.imagedata)
            glBindTexture(GL_TEXTURE_2D, self.myimage2.texturedata)   # 2d texture (x and y size)

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
    def DisplayGlTriangles(self):
        print "before normals compute", time.clock()
        vertices = np.array(vertexlist)
        faces = np.array(faceslist)
        #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros( vertices.shape, dtype=vertices.dtype )
        #Create an indexed view into the vertex array using the array of three indices for triangles
        tris = vertices[faces]
        #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
        n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )# n is now an array of normals per triangle. The length of each normal is dependent the vertices, # we need to normalize these, so that our next step weights each normal equally.normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norm[ faces[:,0] ] += n
        norm[ faces[:,1] ] += n
        norm[ faces[:,2] ] += n
        normalize_v3(norm)
        print "after normals compute", time.clock()
        
        self.listId = glGenLists(1)                                # Generate 2 Different Lists
        glNewList(self.listId,GL_COMPILE)            # Start With The Box List
        glBegin(GL_TRIANGLES)
        for k in range(len(faceslist)):
            #normal by face
            x1 = -vertexlist[faceslist[k][0]][0]
            y1 =  vertexlist[faceslist[k][0]][2]
            z1 = vertexlist[faceslist[k][0]][1]
            x2 = -vertexlist[faceslist[k][1]][0]
            y2 =  vertexlist[faceslist[k][1]][2]
            z2 = vertexlist[faceslist[k][1]][1]
            x3 = -vertexlist[faceslist[k][2]][0]
            y3 =  vertexlist[faceslist[k][2]][2]
            z3 = vertexlist[faceslist[k][2]][1]
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

            #glNormal3f( -nx, ny, -nz);
            glNormal3f( -norm[faceslist[k][0]][0], norm[faceslist[k][0]][2], norm[faceslist[k][0]][1]);
            glTexCoord2f(uvlist[faceslist[k][0]][0], uvlist[faceslist[k][0]][1])
            #glColor3f(0.85, 0.75, 0.7)
            glVertex3f(x1, y1, z1)

            glNormal3f( -norm[faceslist[k][1]][0], norm[faceslist[k][1]][2], norm[faceslist[k][1]][1]);
            glTexCoord2f(uvlist[faceslist[k][1]][0], uvlist[faceslist[k][1]][1])
            #glColor3f(1, 0.66, 0.67)
            glVertex3f(x2, y2, z2)
            
            glNormal3f( -norm[faceslist[k][2]][0], norm[faceslist[k][2]][2], norm[faceslist[k][2]][1]);
            glTexCoord2f(uvlist[faceslist[k][2]][0], uvlist[faceslist[k][2]][1])
            #glColor3f(0.75, 0.65, 0.6)
            glVertex3f(x3, y3, z3)
        glEnd()
        glEndList();
        
        print "display list done", time.clock()


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
            pygame.display.set_caption(szTitle)
            #print szTitle
            
        self.g_nFrames += 1                                                 # // Increment Our FPS Counter

        
        
        # Clear The Screen And The Depth Buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()                  # Reset The View
        
        glColor4f(0.8, 0.8, 0.8, .5)

        # Move Left 1.5 units and into the screen 6.0 units.
        glTranslatef(0, self.camHeight, self.camDistance)
        self.angle += self.rotateSpeed
        glRotatef(self.angle, 0, 1, 0)

        #glBindTexture(self.myimage1.texturedata.target, self.myimage1.texturedata.id)
        
        if self.myimage1 != None:
            glBindTexture(GL_TEXTURE_2D, self.myimage1.texturedata)
        if self.normalMapOn == 1:
            #shader.bind()
            glUseProgram(self.program.program)
            uniforms = self.program.uniforms             
            glActiveTexture(GL_TEXTURE0)
            #shader.uniformi('color_texture', 0)
            glUniform1i(uniforms['color_texture'], 0)
            #shader.uniformi('toggletexture', self.texturesOn)
            glUniform1i(uniforms['toggletexture'], self.texturesOn)        
            glActiveTexture(GL_TEXTURE1)
            #shader.uniformi('normal_texture', 1)
            glUniform1i(uniforms['normal_texture'], 1)
            if self.myimage2 != None:
                glBindTexture(GL_TEXTURE_2D, self.myimage2.texturedata)        
    
        #self.DisplayGlTriangles()
        glCallList(self.listId)
        
        glTranslatef(5,0,0)
        
        '''glCallList(self.listId)
        
        glTranslatef(5,0,0)
        
        glCallList(self.listId)
        
        glTranslatef(-15,0,0)
        
        glCallList(self.listId)
        
        glTranslatef(-5,0,0)
        
        glCallList(self.listId)'''
            
        if self.normalMapOn == 1:
            glActiveTexture(GL_TEXTURE1)
            glDisable(GL_TEXTURE_2D)
            glActiveTexture(GL_TEXTURE0)
            #shader.unbind()
            glUseProgram(0)

        # Since we have smooth color mode on, this will be great for the Phish Heads :-).
        # Draw a triangle
        glBegin(GL_POLYGON)                 # Start drawing a polygon
        glColor3f(1.0, 0.0, 0.0)            # Red
        glVertex3f(0.0, 1.0, 0.0)           # Top
        glColor3f(0.0, 1.0, 0.0)            # Green
        glVertex3f(1.0, -1.0, 0.0)          # Bottom Right
        glColor3f(0.0, 0.0, 1.0)            # Blue
        glVertex3f(-1.0, -1.0, 0.0)         # Bottom Left
        glEnd()                             # We are done with the polygon

        # Move Right 3.0 units.
        glTranslatef(3.0, 0.0, 0.0)

        # Draw a square (quadrilateral)
        glColor3f(0.3, 0.5, 1.0)            # Bluish shade
        glBegin(GL_QUADS)                   # Start drawing a 4 sided polygon
        glVertex3f(-1.0, 1.0, 0.0)          # Top Left
        glVertex3f(1.0, 1.0, 0.0)           # Top Right
        glVertex3f(1.0, -1.0, 0.0)          # Bottom Right
        glVertex3f(-1.0, -1.0, 0.0)         # Bottom Left
        glEnd()                             # We are done with the polygon
        

        #  since this is double buffered, swap the buffers to display what just got drawn.
        #(pyglet provides the swap, so we dont use the swap here)
        #glutSwapBuffers()
        pygame.display.flip()


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_key_press(self, symbol):#, modifiers):
        #if symbol == K_ESCAPE:
        #    self.dispatch_event('on_close')
        #enable or disable rotation
        if symbol == K_SPACE:
            if self.rotateSpeed == 1:
                self.rotateSpeed = 0
            else:
                self.rotateSpeed = 1
        #enable or disable textures
        if symbol == K_t:
            if self.texturesOn == 1:
                self.texturesOn = 0
                glDisable(GL_TEXTURE_2D)
            else:
                self.texturesOn = 1
                glEnable(GL_TEXTURE_2D)
        if symbol == K_n:
            if self.normalMapOn == 1:
                self.normalMapOn = 0
            else:
                self.normalMapOn = 1
        if symbol == K_UP:
            self.camDistance += 1.0
        if symbol == K_DOWN:
            self.camDistance -= 1.0
        if symbol == K_PAGEUP:
            self.camHeight -= 1.0
        if symbol == K_PAGEDOWN:
            self.camHeight += 1.0
                        
'''    def make_resources(self):
        resources = Resources()
        return resources
            
class Resources(object):
    pass'''

def word(long):
    s=''
    for j in range(0,long):
        lit =    struct.unpack('c',plik.read(1))[0]
        if ord(lit)!=0:
            s+=lit
            if len(s)>1000:
                break
    return s


def b(n):
    return struct.unpack(n*'b', plik.read(n))
def B(n):
    return struct.unpack(n*'B', plik.read(n))
def h(n):
    return struct.unpack(n*'h', plik.read(n*2))
def H(n):
    return struct.unpack(n*'H', plik.read(n*2))
def i(n):
    return struct.unpack(n*'i', plik.read(n*4))
def f(n):
    return struct.unpack(n*'f', plik.read(n*4))

def matrix():
    #return Matrix(f(4),f(4),f(4),f(4))
    return 1

#def drawmesh(name):
#    global obj,mesh
    #mesh = bpy.data.meshes.new(name)
    #mesh.verts.extend(vertexlist)
    #mesh.faces.extend(faceslist,ignoreDups=True)
##    vertexuv()
##    scene = bpy.data.scenes.active
##    obj = scene.objects.new(mesh,name)
##    mesh.calcNormals()
##    make_vertex_group()
##    Redraw()



def parser():
    global bonesdata,animdata
    #global mesh_id
    bonesdata=[]
    animdata={}
#    print
    plik.seek(0,2)
    filesize=plik.tell()
    plik.seek(0)
    stop=0
    anim_id=0
    #mesh_id=0
    for k in range(200):
        if plik.tell()==filesize:break
        if stop==1:break
#        print
        print k
        if f(1)[0]==0.0:break
        #plik.tell()
#        print
        section=[]
        nSec=i(1)[0];print 'section count =',nSec
        for m in range(nSec):
            section.append(i(4))
            print m,section[m]
        off=plik.tell()
        for m in range(nSec):
            #plik.seek(off+section[m][1])#begin section
            print 'section type =',section[m][0],plik.tell()
            if section[m][0]==1:
                stype=B(4);print 'stype =',stype
                if stype==(0,0,128,63):#as float=1.0
                    var = i(6);print var
                    #break
                elif stype==(82,184,158,63):#as float=1.240000954
                    var = i(2);print var
                    seek=140
                    #break
                elif stype==(154,153,153,63):
                    var = i(4);print var
                    seek=380
                    #break
                elif stype in [(184,30,133,63),(92,143,130,63)]:
                #elif stype==(184,30,133,63):
                    var=(0,1)
                    pass
                elif stype==(123,20,206,63):
                    pass
                elif stype==(205,204,204,63):
                    #var = i(2);print var
                    #seek=140
                    pass
                else:
                    print 'stop'
                    break
            elif section[m][0]==2:
                if stype==(154,153,153,63):
                    for n in range(var[0]):
                        t=plik.tell()
                        bone = word(128)
                        parent = word(128)
                        #print bone,parent
                        plik.seek(t+260)
                        #print matrix()
                        #print f(3)
                        #print f(3)
                        #print f(4)
                        bonesdata.append([bone,parent,matrix()])
                        plik.seek(t+seek)
                else:
                    for n in range(var[0]):
                        t=plik.tell()
                        bone = word(128)
                        parent = word(128)
                        #print bone,parent
                        plik.seek(t+seek)

            elif section[m][0]==3:
                pass
                #back=plik.tell()
                #build_mesh(var[1])
                #stop=1
                #plik.seek(back)
                tell=plik.tell()
                plik.seek(section[m][2],1)
            elif section[m][0]==4:
                print i(var[1])
                plik.seek(section[m][2],1)
            elif section[m][0]==5:
                pass
                print i(1)[0]
                #plik.seek(section[m][2],1)
            elif section[m][0]==6:
                list=[]
                back=plik.tell()
                for n in range(var[0]):
                    t=plik.tell()
                    list.append([word(128),i(4)])
                    plik.seek(t+144)
                plik.seek(back)
                plik.seek(off+section[m][1]+section[m][2])
            elif section[m][0]==7:
                back=plik.tell()
                animdata={}
                animdata['list']=list
                for id in range(var[0]):
                    data=list[id][1]
                    #print data
                    bonename=list[id][0]
                    animdata[bonename]={}
                    animdata[bonename]['loc']=[]
                    for n in range(data[1]):
                        animdata[bonename]['loc'].append(f(4))

                for id in range(var[0]):
                    data=list[id][1]
                    #print data
                    bonename=list[id][0]
                    if bonename not in animdata:
                        animdata[bonename]={}
                    animdata[bonename]['rot']=[]
                    for n in range(data[2]):
                        animdata[bonename]['rot'].append(f(5))

                plik.seek(back)
                plik.seek(off+section[m][1]+section[m][2])
                anim_id+=1
            elif section[m][0]==17:
                back=plik.tell()
                plik.seek(tell)
                build_mesh(var[1])
                #stop=1
                plik.seek(back)
                print i(1)[0]
                plik.seek(section[m][2],1)
            elif section[m][0]==19:
                num= i(1)[0];print 'num at 19 section',num
                #print i(1)
                for p in range(num):
                    #plik.seek(section[m][2],1)
                    plik.seek(540,1)
                    nTex=i(1)[0];print 'tex count =',nTex
                    print B(12)
                    #help=0
                    for m1 in range(nTex):
                        #print f(2)
                        print word(264)
                        for n1 in range(i(1)[0]):
                            print i(1)
                            print word(64)
                            print i(4)
                            #help=1
            else:
                print 'stop'
                break


    print plik.tell()

def build_mesh(num):
    global vertexlist,faceslist,uvlist,weightlist
    global used_bones
    #global mesh_id
    used_bones=[]

    back=plik.tell()
    for k in range(num):
        count=i(1)[0]
        plik.seek(count*92,1)
        count=i(1)[0]
        plik.seek(count*12,1)
        plik.seek(336,1)
    for k in range(i(1)[0]):
        used_bones.append(word(64))
        #print used_bones[k]

    plik.seek(back)
    
    #merge all model
    vertexlist=[]
    faceslist=[]
    uvlist=[]
    weightlist=[]
    index_start = 0

    for n in range(num):
        tell=plik.tell()
#         vertexlist=[]
#         faceslist=[]
#         uvlist=[]
#         weightlist=[]
        for m in range(i(1)[0]):
            t=plik.tell()
            vertexlist.append(f(3))
            #print vertexlist[m]
            plik.seek(t+28)
            uvlist.append([f(1)[0],1-f(1)[0]])
            plik.seek(t+60)
            weightlist.append([f(4),H(8)])
            plik.seek(t+92)
        for m in range(i(1)[0]):
            #t=plik.tell()
            #faceslist.append(i(3))
            fi = i(3)
            faceslist.append([fi[0] + index_start, fi[1] + index_start, fi[2] + index_start])
            #plik.seek(t+12)
        print 'end mesh section at =',plik.tell()
        index_start += len(vertexlist) #merge all model so index of previous model are added 
        #drawmesh(str(model_id)+'-model-'+str(mesh_id))
        #mesh.materials+=[Material.Get(str(model_id)+'-mat')]
        #assign_materials(mesh,mesh_id,filename,1,image_files,model_id)
        #armobj.makeParentDeform([obj],1,0)
        #assign_materials(str(model_id)+'-model-'+str(mesh_id))
        #print 'build_mesh num ', num, ' model ', str(model_id)+'-model-'+str(mesh_id)
        plik.seek(336,1)
        #mesh_id+=1
        


##################################main
if __name__ == "__main__":
    global plik,dirname,basename
    #global model_id
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        filename.replace('\\', '/')
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        if dirname != '':
            dirname += '/'
        model = dirname + filename.split('\\')[-1].split('.')[0]
    else:
        #filename = "NPC_Gloria.mesh"
        #filename = "NPC_Yuna.mesh"
        #filename = "NPC_Sati.mesh"
        filename = model + ".mesh"
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        if dirname != '':
            dirname += '/'
        
    plik=open(filename,'rb')
    parser()
    #skeleton()
    plik.close()

    print 'nb vertex', len(vertexlist), ', nb face', len(faceslist), ', uv list', len(uvlist)
    #print vertexlist[0], faceslist[0], faceslist[0][0], faceslist[0][1], faceslist[0][2]
#    print vertexlist[faceslist[0][0]][0], vertexlist[faceslist[0][0]][1], vertexlist[faceslist[0][0]][2]


    
    #pyglet.app.run()
    
    video_flags = OPENGL|DOUBLEBUF|RESIZABLE#|FULLSCREEN
    pygame.init()
    screen_dimensions = 800, 600
    surface = pygame.display.set_mode(screen_dimensions, video_flags)
    #pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1) #don't work
    #wglext_arb.wglSwapIntervalEXT(0)
    #glxext_arb.glXSwapIntervalSGI(0)
    #OpenGL.raw._WGL_ARB.wglSwapIntervalEXT(0)
    #OpenGL.WGL.EXT.swap_control.wglSwapIntervalEXT(1)
    #OpenGL.raw.wglSwapIntervalEXT(0)
    #OpenGL.WGL.wglSwapIntervalEXT(1)
    
    window = World()
    #resources = window.make_resources()    
    
    '''frames = 0
    done = 0
    zoom = 1.0
    position = [256.0, 256.0]
    dragging = False
    draglast = 0,0'''
    
    while 1:
        event = pygame.event.poll()
        #if event.type == NOEVENT:
        #    break
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        elif (event.type == KEYDOWN):
            window.on_key_press(event.key)
        elif event.type == VIDEORESIZE:            
            window.ReSizeGLScene(event.dict['size'][0], event.dict['size'][1])
        window.DrawGLScene()
        time.sleep(0.005)
        #frames += 1
        
    pygame.quit()
