from kivy.app import App
from kivy.clock import Clock
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.core.window import Window
from kivy.core.image import Image
from PylLoader import PylLoader
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics.opengl import *
from kivy.graphics import *


class Renderer(Widget):
    def __init__(self, **kwargs):
        
        #filename = 'points.ply'
        filename = 'skull.ply'
        #filename = 'fragment.ply' # binary error reading
        
        scale = 1.0 / 50
        
        self.scene = PylLoader(resource_find(filename), scale)
        
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.source = resource_find('shaders-opengl-triangle.glsl')

        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        Clock.schedule_interval(self.update_glsl, 1 / 60.)
        
        # ============= All stuff after is for trackball implementation ===========
        self._touches = []

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)
        glEnable(0x8642) #GL_VERTEX_PROGRAM_POINT_SIZE
        #glEnable(0x8861) #GL_POINT_SPRITE

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def update_glsl(self, *largs):
        #proj = Matrix().view_clip(0, self.width, 0, self.height, 1, 100, 0)
        #proj = Matrix().view_clip(0, self.width, 0, self.height, 0, 500, .9)
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 500, 1)
        self.canvas['projection_mat'] = proj

    def setup_scene(self):
        #Color(0, 0, 0, 1)
        Color(.5, .5, .5, 0)

        PushMatrix()
        Translate(0, -3, -5)
        # This Kivy native Rotation is used just for
        # enabling rotation scene like trackball
        self.rotx = Rotate(0, 1, 0, 0)
        # here just rotate scene for best view
        self.roty = Rotate(180, 0, 1, 0)
        self.scale = Scale(1)

        self.draw_elements()
        
        PopMatrix()
        
    def draw_elements(self):
        #Draw separately all meshes on the scene
        def _draw_element(m):                
            mesh = Mesh(
                vertices=m.vertices,
                indices=m.indices,
                fmt=m.vertex_format,
                mode='points',
                group='truc',
            )

        for meshid in range(0, len(self.scene.objects)):
            # Draw each element
            mesh = self.scene.objects[meshid]

            _draw_element(mesh)
        
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
        return Renderer()

if __name__ == "__main__":
    RendererApp().run()