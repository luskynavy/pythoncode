---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec4  v_color;
attribute vec2  v_tc0;
attribute vec3 v_normal;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec4 normal_vec;
varying vec4 vertex_pos;

varying vec4 frag_color;
varying vec2 uv_vec;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
	vertex_pos = pos;
    gl_Position = projection_mat * pos;
	normal_vec = vec4(v_normal,0.0);
    frag_color = v_color;
    uv_vec = v_tc0;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 frag_color;
varying vec2 uv_vec;

varying vec4 normal_vec;
varying vec4 vertex_pos;

uniform mat4 normal_mat;

// New uniform that will receive texture at index 1
uniform sampler2D texture1;
uniform sampler2D texture0;

void main (void){
	//correct normal, and compute light vector (assume light at the eye)
    vec4 v_normal = normalize( normal_mat * normal_vec ) ;
	vec4 v_light = normalize( vec4(0,0,0,1) - vertex_pos );
    //reflectance based on lamberts law of cosine
    float theta = clamp(dot(v_normal, v_light), 0.0, 1.0);
    gl_FragColor = vec4(theta, theta, theta, 1.0);

    vec4 color1 = texture2D(texture1, uv_vec);
    vec4 color = texture2D(texture0, uv_vec);
    //color = color * color1;
//	color.a = 1.0; //disable alpha transparency
    gl_FragColor = color*theta;
    gl_FragColor.a = 1.0;
}