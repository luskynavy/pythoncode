from kivy.app import App
from kivy.clock import Clock
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics import Canvas, Rectangle, Callback, PushMatrix, \
    PopMatrix, Color, Translate, Rotate, Scale, Mesh, ChangeState, \
    UpdateNormalMatrix
#from objloader import ObjFileLoader
from pskloader import PSKFileLoader
from MeshAsciiLoader import MeshAsciiLoader
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics.fbo import Fbo
from kivy.properties import ObjectProperty
from kivy.core.image import Image
from kivy.utils import platform


class Renderer(Widget):
    texture = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        #self.canvas = RenderContext(compute_normal_mat=True)
        #self.canvas.shader.source = resource_find('simple.glsl')
        self.canvas = Canvas()
        #self.scene = ObjFileLoader(resource_find("testnurbs.obj"))
        
        Logger.debug('******************************************************')
        scale = 3
        #dir = "Pyro/Pyro Red"        
        # dir = "Mai Venus Bikini"        
        #dir = "TRACY [B_A_O]"
        # dir = "Mai Shiranui Biniki 2"
        # dir = "Wrench_Girl_Fight/Default"
        # dir, scale = "Scarllet_Lingerie_KiD/Scarllet_Lingerie", .04
        #dir, scale = "Rumble_Roses_XX_Lady_X_Substance_by_darkblueking", 15
        # dir = "RG_Kaori"
        # dir = "Ol.aMANDA_heavy"
        #dir = "Mai Shiranui Bikini Red"
        # dir = "AlphaProt_Uli_Booli_Classy"
        # dir = "AlphaProt_Lazo_Girls_Brawn"
        # dir = "AlphaProt_Lazo_Girls_Red"
        #dir = "Bayonetta_Default_ Bayonetta"
        #dir = "Bayonetta_nude_V2.5"
        # dir = "BnS-Gon-F002_NCSoft"
        #dir, scale = "CANDY [B_A_O]", 1.5
        # dir = "DeadOrAlive5_HelenaX2Venus"
        # dir = "Dixie_RR_XX"
        # dir = "DOA5_Christie_Dominatrix/Wet text"        
        #dir = "DOA5U_Ayane_Intimate_TRDaz"
        # dir = "DOA5_Kasumi_Hot_Getaway/Model"
        #dir, scale = "Devil May Cry 4 - Trish", 1.5
        # dir = "DOA5-X2_Ayane_AquamarineSwimsuit_TRDaz"
        # dir = "DOA5_Kokoro_Cos7"
        # dir = "DOA5_Kokoro_Halloween"
        #dir = "DOA5U_Christie_Halloween_TRDaz/DOA5U_Christie_Halloween_Hair1"
        dir = "DOA5U_Helena_Halloween_TRDaz"        
        # dir = "DOA5U_Kasumi_Casual/Model/Braid"
        #dir = "DOA5U_Rachel_Business/Model"
        # dir = "DOA5U_Rachel_Casual/Model"
        #dir = "Duke Nukem Forever_Dr_Valencia" # error index out of range
        #dir = "Duke Nukem Forever_Kitty Pussoix"
        dir = "Duke_Nukem_by_Ventrue"
        #dir = "Ivy_SCIV_bonus_pack_L2R/Ivy_1"
        #dir = "Soul_Calibur_IV_Ivy by blufan and vega82"
        #dir = "Kagura_Monokini_O_Z_K"
        #dir = "Rachel Tropical DLC by RoxasKennedy/Normal"
        #dir = "T_Rev_Eliza/Bigger Boobs"        
        #dir = "Devil May Cry 4 - Gloria"
        #dir = "Blade&Soul - Gon female 1"
        #dir = "Blade&Soul - Jin female 01"
        # dir = "Blade&Soul - Jin female 03"
        #dir = "Blade&Soul - Jin female Yuran"
        # dir = "BlackDesert_HumanFemale_Base"
        # dir = "DOA5U_Lisa_Hamilton_Tropical/Unmasked"
        # dir = "DOA5U_Tina_Armstrong_DOA2_Suit/Model"
        # dir = "DOA5U_Tina_Armstrong_Legacy/Model"
        #dir = "DOA5U_Rachel_Nurse/Model" #pb uv        
        # dir = "Injustice _Zatanna_Zatara/Zatana_Normal"
        # dir = "MOM_BIKINI"
        #dir, scale = "Natalia_Lingerie_KiD", .04 
        #dir = "Sefi_Naked"
        #dir = "Sefi_FC"
        #dir = "Alt_Sefi_SC"
        # dir = "Rachael_Foley_RE_Revelation"
        # dir = "Ruidia_FC"
        #dir = "Def_Rudia_SC"
        #dir = "Rumble Roses XX - Candy Cane (Superstar)"
        if platform == 'android':
            self.scene = MeshAsciiLoader(resource_find(dir + "/Generic_Item.mesh.ascii"), scale)
        else:
            self.scene = MeshAsciiLoader(resource_find("../../Mesh/" + dir + "/Generic_Item.mesh.ascii"), scale)
                
        #scale = .03
        #self.texturename = 'Batman_Rabbit_Head_Posed/Batman_V3_Body_D.PNG'
        #self.scene = PSKFileLoader(resource_find("Batman_Rabbit_Head_Posed/Batman_Rabbit_Head_Posed.psk"), scale)
        #self.texturename = 'Gray/Gray_C1.tga'
        #self.scene = PSKFileLoader(resource_find("Gray/Mike_TPP.psk"), scale) #too many indices, good for split test
        #self.texturename = 'CV_Talia/Talia_Body_D.tga'
        #self.texturename = 'CV_Talia/Talia_Legs_D.tga')
        #self.scene = PSKFileLoader(resource_find("CV_Talia/Talia_posed.psk"), scale) #too many indices, good for split test
        #self.texturename = 'AlphaProt_Uli_Booli_Classy/Uli_body_02_D2.tga'
        #self.texturename = 'AlphaProt_Uli_Booli_Classy/Uli_new_UV_DAO.tga'
        #self.scene = PSKFileLoader(resource_find("AlphaProt_Uli_Booli_Classy/NPC_Uli_NightDress.psk"), .07)
        #self.scene = PSKFileLoader(resource_find("AlphaProt_Uli_Booli_Classy/NPC_Uli_Head.psk"), scale)
        Logger.debug('******************************************************')

        #self.meshes = []

        with self.canvas:
            self.fbo = Fbo(size=self.size,
                           with_depthbuffer=True,
                           compute_normal_mat=True,
                           clear_color=(0, 0, 0, 0.))
            self.viewport = Rectangle(size=self.size, pos=self.pos)
        self.fbo.shader.source = resource_find('simple.glsl')
        #self.texture = self.fbo.texture

        super(Renderer, self).__init__(**kwargs)        

        with self.fbo:
            #ClearBuffers(clear_depth=True)

            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)

        Clock.schedule_interval(self.update_scene, 1 / 60.)

        self._touches = []

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
        """ Draw separately all objects on the scene
            to setup separate rotation for each object
        """
        def _draw_element(m, texture=''):
            mesh = Mesh(
                vertices=m.vertices,
                indices=m.indices,
                fmt=m.vertex_format,
                mode='triangles',
            )
            if texture:
                try:
                    texture = Image(texture).texture
                    texture.wrap = 'repeat' #enable of uv support > 1 or <1
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
                _draw_element(mesh, mesh.diffuse)
            else:
                _draw_element(mesh, self.texturename)

        # Then draw other elements and totate it in different axis
        #pyramid = self.scene.objects['Pyramid']
        '''pyramid = self.scene.objects[1]
        PushMatrix()
        self.pyramid_rot = Rotate(0, 0, 0, 1)
        _set_color(0., 0., .7, id_color=(0., 0., 255))
        _draw_element(pyramid, 'rain.png')
        PopMatrix()

        #box = self.scene.objects['Box']
        box = self.scene.objects[2]
        PushMatrix()
        self.box_rot = Rotate(0, 0, 1, 0)
        _set_color(.7, 0., 0., id_color=(255, 0., 0))
        _draw_element(box, 'bricks.png')
        PopMatrix()

        #cylinder = self.scene.objects['Cylinder']
        cylinder = self.scene.objects[3]
        PushMatrix()
        self.cylinder_rot = Rotate(0, 1, 0, 0)
        _set_color(0.0, .7, 0., id_color=(0., 255, 0))
        _draw_element(cylinder, 'wood.png')
        PopMatrix()'''

    def update_scene(self, *largs):
        '''self.pyramid_rot.angle += 0.5
        self.box_rot.angle += 0.5
        self.cylinder_rot.angle += 0.5'''

    # ============= All stuff after is for trackball implementation ===========

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx/self.width)*360
        y_angle = -1*(touch.dy/self.height)*360
        return x_angle, y_angle

    def on_touch_down(self, touch):
        self._touch = touch
        touch.grab(self)
        self._touches.append(touch)

    def on_touch_up(self, touch):
        touch.ungrab(self)
        self._touches.remove(touch)

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
        #root.add_widget(btn)
        return root


if __name__ == "__main__":
    RendererApp().run()
