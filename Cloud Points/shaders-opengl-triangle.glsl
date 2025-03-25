---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec4  v_color;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec4 frag_color;

void main (void) {
    gl_PointSize = 1.0;
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
    gl_Position = projection_mat * pos;
    frag_color = v_color;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 frag_color;
varying vec2 uv_vec;

uniform sampler2D tex;

void main (void){
    gl_FragColor = frag_color;
}
