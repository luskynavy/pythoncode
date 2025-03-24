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
	
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
    gl_Position = projection_mat * pos;
    if (gl_Position.w < 5)
    {
        gl_PointSize = 5.0f;
    }
    else
    {
        gl_PointSize = 2.0f;
    }
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
