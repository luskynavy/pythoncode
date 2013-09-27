
#read demoShadows.py

#import openGL
from OpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *

#import extension for shadows..
from OpenGL.GL.ARB.shadow import *
from OpenGL.GL.ARB.depth_texture import *
from OpenGL.GL.ARB.transpose_matrix import *


shadowMapSize = 512
spotFOV = 100.0

def CreateTextureShadow():
    'before loop, crate a texture obj to store shadow map'
    id = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, id)

    glTexImage2D(   GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
                    shadowMapSize, shadowMapSize, 0,
                    GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, None
                    )

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)

    return id
                
def CreateShadowBefore(position):
    """every frame render objs which casts shadows from light point of view
    position --> light position
    return --> projection matrix of the shadow map created"""
    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()
    gluPerspective(spotFOV, 1.0, 0.1, 100.0)

    lightProjectionMatrix = glGetFloatv(GL_PROJECTION_MATRIX)

    glMatrixMode(GL_MODELVIEW)
    
    glLoadIdentity()
    gluLookAt(  position[0], 
                position[1],
                position[2],
                0.0, 10.0, 0.0, 
                0.0, 1.0, 0.0 )
    lightViewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)

    #Use viewport the same size as the shadow map
    glViewport(0, 0, shadowMapSize, shadowMapSize)

    #Draw back faces into the shadow map
    glCullFace(GL_FRONT)

    #Disable lighting, texture, use flat shading for speed
    glShadeModel(GL_FLAT)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)

    #Disable color writes
    glColorMask(0, 0, 0, 0)

    glPolygonOffset(0.5, 0.5)
    glEnable(GL_POLYGON_OFFSET_FILL) 

    glClear( GL_DEPTH_BUFFER_BIT )

    #eval projection matrix
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()

    glLoadMatrixf(  [   [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.5, 0.0],
                        [0.5, 0.5, 0.5, 1.0] ])

    glMultMatrixf(lightProjectionMatrix)
    glMultMatrixf(lightViewMatrix)
 
    resMatrix = glGetFloatv(GL_TRANSPOSE_MODELVIEW_MATRIX)
  
    glPopMatrix()
    return resMatrix

def CreateShadowAfter(shadowMapID):
    'write texture into texture obj and reset gl params'
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, shadowMapID)
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, shadowMapSize, shadowMapSize)

    glCullFace(GL_BACK)
    glShadeModel(GL_SMOOTH)
    glColorMask(1, 1, 1, 1)

    glDisable(GL_POLYGON_OFFSET_FILL)
    glDisable(GL_TEXTURE_2D)

def RenderShadowCompareBefore(shadowMapID, textureMatrix):
    'eval where draw shadows using ARB extension'
    glEnable(GL_TEXTURE_2D)          
    glBindTexture(GL_TEXTURE_2D, shadowMapID)

    glEnable(GL_TEXTURE_GEN_S)
    glEnable(GL_TEXTURE_GEN_T)
    glEnable(GL_TEXTURE_GEN_R)
    glEnable(GL_TEXTURE_GEN_Q)
    
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_S, GL_EYE_PLANE, textureMatrix[0])

    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_T, GL_EYE_PLANE, textureMatrix[1])

    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_R, GL_EYE_PLANE, textureMatrix[2])

    glTexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_Q, GL_EYE_PLANE, textureMatrix[3])

    #Enable shadow comparison
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB)

    #Shadow comparison should be true (in shadow) if r>texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_GREATER)

    #Shadow comparison should generate an INTENSITY result
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_INTENSITY)

    #Set alpha test to discard false comparisons
    glAlphaFunc(GL_EQUAL, 1.0)
    glEnable(GL_ALPHA_TEST)

    glLightfv(GL_LIGHT0, GL_DIFFUSE, ( 0.6,0.6,0.6,1.0 ))   # Diffuse Light for shadows

def RenderShadowCompareAfter():
    'reset gl params after comparison'
    glDisable(GL_TEXTURE_2D)

    glDisable(GL_TEXTURE_GEN_S)
    glDisable(GL_TEXTURE_GEN_T)
    glDisable(GL_TEXTURE_GEN_R)
    glDisable(GL_TEXTURE_GEN_Q)

    glDisable(GL_ALPHA_TEST) 


