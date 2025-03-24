from kivy.app import App
from kivy.clock import Clock
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics import Canvas, Rectangle, Callback, PushMatrix, \
    PopMatrix, Color, Translate, Rotate, Scale, Mesh, ChangeState, \
    UpdateNormalMatrix, BindTexture
#from objloader import ObjFileLoader
#from pskloader import PSKFileLoader
from MeshAsciiLoader import MeshAsciiLoader
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics.fbo import Fbo
from kivy.properties import ObjectProperty
from kivy.core.image import Image
from kivy.utils import platform

from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
import os

#default mesh dir
meshdir = "../../Mesh/"

class Renderer(Widget):
    texture = ObjectProperty(None, allownone=True)
    
    def load(self, *l):
        self.button.disabled = True

        
        if os.path.isdir(meshdir):
            rep = meshdir
            self.fl = FileChooserListView(path = rep, rootpath = rep, filters = ["*.mesh.ascii"],
                 dirselect=False, size=(400,400), center_x = 250, center_y = 250)            
        else:
            rep = os.getcwd()
            self.fl = FileChooserListView(path = rep, filters = ["*.mesh.ascii"],
                 dirselect=False, size=(400,400), center_x = 250, center_y = 250)
                        
        '''if platform == 'android':
            rep = "."
        else:
            rep = "../../Mesh"'''        
        
        self.fl.bind(selection=self.on_selected)
        super(Renderer, self).add_widget(self.fl)
    
    def on_selected(self, filechooser, selection):
        self.button.disabled = False
        
        print 'load', selection
        super(Renderer, self).remove_widget(self.fl)
        
        scale = 3
        self.fbo.remove_group('truc')        
        
        self.scene = MeshAsciiLoader(selection[0], scale)
                    
       
        with self.fbo:
            #ClearBuffers(clear_depth=True)

            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)    
        
    def change_shader(self, *l):
        print 'change_shader'
        if self.shader == 0:            
            self.fbo.shader.source = resource_find('simple.glsl')
            self.shader = 1
        else:
            #self.fbo.shader.source = resource_find('flat.glsl')            
            self.fbo.shader.source = resource_find('normalmap.glsl')
            self.shader = 0
        self.update_glsl()
        
        
    def __init__(self, **kwargs):
        self.model = 0
        self.shader = 0 
        self.canvas = Canvas()
        #self.scene = ObjFileLoader(resource_find("testnurbs.obj"))
        
        Logger.debug('******************************************************')
        scale = 3        
        #dir = "Duke Nukem Forever_Dr_Valencia" # error index out of range
        #dir = "Duke Nukem Forever_Kitty Pussoix"
        dir = "Duke_Nukem_by_Ventrue"
        #dir = "DOA5U_Rachel_Nurse/Model" #pb uv
                
        if not os.path.isdir(meshdir):
            self.scene = MeshAsciiLoader(resource_find(dir + "/Generic_Item.mesh.ascii"), scale)
        else:
            self.scene = MeshAsciiLoader(resource_find(meshdir + dir + "/Generic_Item.mesh.ascii"), scale)
                
        #scale = .03
        #self.texturename = 'Batman_Rabbit_Head_Posed/Batman_V3_Body_D.PNG'
        #self.scene = PSKFileLoader(resource_find("Batman_Rabbit_Head_Posed/Batman_Rabbit_Head_Posed.psk"), scale)
        #self.texturename = 'Gray/Gray_C1.tga'
        #self.scene = PSKFileLoader(resource_find("Gray/Mike_TPP.psk"), scale) #too many indices, good for split test
        #self.texturename = 'CV_Talia/Talia_Body_D.tga'
        #self.texturename = 'CV_Talia/Talia_Legs_D.tga')
        #self.scene = PSKFileLoader(resource_find("CV_Talia/Talia_posed.psk"), scale) #too many indices, good for split test        
        Logger.debug('******************************************************')        

        with self.canvas:
            self.fbo = Fbo(size=self.size,
                           with_depthbuffer=True,
                           compute_normal_mat=True,
                           clear_color=(0, 0, 0, 0.))
            self.viewport = Rectangle(size=self.size, pos=self.pos)
        #self.fbo.shader.source = resource_find('simple.glsl')
        self.fbo.shader.source = resource_find('normalmap.glsl')

        super(Renderer, self).__init__(**kwargs)
        
        self.fbo['texture1'] = 1
        self.fbo['toggletexture'] = 1

        with self.fbo:
            #ClearBuffers(clear_depth=True)            

            self.cb = Callback(self.setup_gl_context)

            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)        
            
        Clock.schedule_interval(self.update_scene, 1 / 60.)

        # ============= All stuff after is for trackball implementation ===========
        self._touches = []
        
        self.button = Button(text='load')
        self.button.bind(on_release=self.load)
        super(Renderer, self).add_widget(self.button)
                
        button1 = Button(text='shader', center_x = 150)
        button1.bind(on_release=self.change_shader)
        super(Renderer, self).add_widget(button1)

    def on_size(self, instance, value):
        self.fbo.size = value
        self.viewport.texture = self.fbo.texture
        self.viewport.size = value
        self.update_glsl()

    def on_pos(self, instance, value):
        self.viewport.pos = value

    def on_texture(self, instance, value):
        self.viewport.texture = value

    def setup_gl_context(self, *args):
        #clear_buffer        
        glEnable(GL_DEPTH_TEST)
        self.fbo.clear_buffer()        
        #glDepthMask(GL_FALSE);

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def update_glsl(self, *largs):
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 500, 1)
        self.fbo['projection_mat'] = proj
        self.fbo['diffuse_light'] = (1.0, 0.0, 0.0)
        self.fbo['ambient_light'] = (0.1, 0.1, 0.1)
        self.fbo['glLightSource0_position'] = (1.0, 1.0, 1.0, 0.0)
        self.fbo['glLightSource0_spotCutoff'] = 360
        self.fbo['glLightModel_ambient'] = (0.2, 0.2, 0.2, 1.0)
        self.fbo['glLightSource0_diffuse'] = (0.7, 0.7, 0.7, 1.0)
        self.fbo['glLightSource0_specular'] = (.1, .1, .1, 1.0)
        self.fbo['glFrontMaterial_specular'] = (.10, .10, .10, 1.0)
        self.fbo['glFrontMaterial_shininess'] = 0.1
        
    def setup_scene(self):
        Color(.5, .5, .5, 0)

        PushMatrix()
        Translate(0, -3, -5)        
        # This Kivy native Rotation is used just for
        # enabling rotation scene like trackball
        self.rotx = Rotate(0, 1, 0, 0)
        # here just rotate scene for best view
        self.roty = Rotate(180, 0, 1, 0)
        self.scale = Scale(1)
        
        UpdateNormalMatrix()

        self.draw_elements()

        PopMatrix()

    def draw_elements(self):
        #Draw separately all meshes on the scene
        def _draw_element(m, texture='',texture1=''):
            #bind the texture BEFORE the draw (Mesh) 
            if texture1:                
                # here, we are binding a custom texture at index 1
                # this will be used as texture1 in shader.
                tex1 = Image(texture1).texture
                tex1.wrap = 'repeat' #enable of uv support >1 or <0                
                BindTexture(texture=tex1, index=1)
            #clear the texture if none
            else:                
                BindTexture(source="", index=1)
                
            mesh = Mesh(
                vertices=m.vertices,
                indices=m.indices,
                fmt=m.vertex_format,
                mode='triangles',
                group='truc',
            )

            if texture:
                try:
                    texture = Image(texture).texture
                    texture.wrap = 'repeat' #enable of uv support >1 or <0
                    mesh.texture = texture
                except: #no texture if not found or not supported
                    pass            

        def _set_color(*color, **kw):
            id_color = kw.pop('id_color', (0, 0, 0))
            return ChangeState(
                Kd=color,
                Ka=color,
                Ks=(.3, .3, .3),
                Tr=1., Ns=1.,
                intensity=1.,
                id_color=[i / 255. for i in id_color],
            )

        for meshid in range(0, len(self.scene.objects)):
            # Draw each element
            mesh = self.scene.objects[meshid]
            #_set_color(0.7, 0.7, 0., id_color=(255, 255, 0))
            if (mesh.diffuse != ""):
                _draw_element(mesh, mesh.diffuse, mesh.normal)
                #_draw_element(mesh, mesh.diffuse)
                #_draw_element(mesh, mesh.normal)
            else:
                _draw_element(mesh, self.texturename)
        
    def update_scene(self, *largs):
        pass
    # ============= All stuff after is for trackball implementation ===========

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx/self.width)*360
        y_angle = -1*(touch.dy/self.height)*360
        return x_angle, y_angle

    def on_touch_down(self, touch):
        self._touch = touch
        touch.grab(self)
        self._touches.append(touch)
        return super(Renderer, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        touch.ungrab(self)
        self._touches.remove(touch)
        return super(Renderer, self).on_touch_up(touch)

    def on_touch_move(self, touch):
        self.update_glsl()
        if touch in self._touches and touch.grab_current == self:
            if len(self._touches) == 1:
                # here do just rotation
                ax, ay = self.define_rotate_angle(touch)

                self.roty.angle += ax
                self.rotx.angle += ay

            elif len(self._touches) == 2:  # scaling here
                #use two touches to determine do we need scal
                touch1, touch2 = self._touches
                old_pos1 = (touch1.x - touch1.dx, touch1.y - touch1.dy)
                old_pos2 = (touch2.x - touch2.dx, touch2.y - touch2.dy)

                old_dx = old_pos1[0] - old_pos2[0]
                old_dy = old_pos1[1] - old_pos2[1]

                old_distance = (old_dx*old_dx + old_dy*old_dy)
                #Logger.debug('Old distance: %s' % old_distance)

                new_dx = touch1.x - touch2.x
                new_dy = touch1.y - touch2.y

                new_distance = (new_dx*new_dx + new_dy*new_dy)

                #Logger.debug('New distance: %s' % new_distance)
                SCALE_FACTOR = 0.01

                if new_distance > old_distance:
                    scale = SCALE_FACTOR
                    #Logger.debug('Scale up')
                elif new_distance == old_distance:
                    scale = 0
                else:
                    scale = -1*SCALE_FACTOR
                    #Logger.debug('Scale down')

                xyz = self.scale.xyz

                if scale:
                    self.scale.xyz = tuple(p + scale for p in xyz)


class RendererApp(App):
    def build(self):
        root = FloatLayout()
        renderer = Renderer()
        root.add_widget(renderer)        
        return root


if __name__ == "__main__":
    RendererApp().run()
