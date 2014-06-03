---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 v_pos;
attribute vec4 v_color;
attribute vec2 v_tc0;
attribute vec3 v_normal;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec4 normal_vec;
varying vec4 vertex_pos;

varying mat3 localSurface2View; // mapping from
   // local surface coordinates to view coordinates
varying vec2 texCoords; // texture coordinates
varying vec4 position; // position in view coordinates

varying vec4 frag_color;
varying vec2 uv_vec;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
	vertex_pos = pos;
    gl_Position = projection_mat * pos;
	normal_vec = vec4(v_normal,0.0);
    frag_color = v_color;
    uv_vec = v_tc0;
	
	//vec3 tangent = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5)); //maybe
    //vec3 tangent = vec3(0,0,1);
    vec3 tangent = normalize(cross(vec3(0,1,0), gl_Normal.xyz));
    // the signs and whether tangent is in localSurface2View[1]
    // or localSurface2View[0] depends on the tangent
    // attribute, texture coordinates, and the encoding
    // of the normal map
    localSurface2View[0] = normalize(vec3(gl_ModelViewMatrix * vec4(vec3(tangent), 0.0)));
    //localSurface2View[0]= vec3(1,0,0);
    localSurface2View[2] = normalize(gl_NormalMatrix * gl_Normal);
    localSurface2View[1] = normalize(cross(localSurface2View[2], localSurface2View[0]));


    texCoords = v_tc0;
    position = gl_ModelViewMatrix * gl_Vertex;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif


varying vec2 texCoords; // texture coordinates



varying mat3 localSurface2View; // mapping from
// local surface coordinates to view coordinates
varying vec2 uv_vec; // texture coordinates
varying vec4 position; // position in view coordinates

//uniform sampler2D color_texture;
//uniform sampler2D normal_texture;
uniform int toggletexture; // false/true

uniform sampler2D texture1;
uniform sampler2D tex;


void main()
{
    //vec4 texColor = vec4(texture2D(color_texture, gl_TexCoord[0].st).rgb, 1.0);
    vec4 texColor =  texture2D(tex, texCoords);
	
	if (toggletexture == 0)
        texColor = vec4(0.75, 0.75, 0.75, 1.0);//gl_FrontMaterial.ambient;

    vec4 encodedNormal = texture2D(texture1, vec2(texCoords));

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
	
	//if (0.0 == gl_LightSource[0].position.w)
	if (1)
    // directional light?
    {
       attenuation = 1.0; // no attenuation
       lightDirection = normalize(vec3(gl_LightSource[0].position));
    }
    else // point light or spotlight (or other kind of light)
    {
       vec3 positionToLightSource = vec3(gl_LightSource[0].position - position);
       float distance = length(positionToLightSource);
       attenuation = 1.0 / distance; // linear attenuation
       lightDirection = normalize(positionToLightSource);

       if (gl_LightSource[0].spotCutoff <= 90.0) // spotlight?
       {
          float clampedCosine = max(0.0, dot(-lightDirection, gl_LightSource[0].spotDirection));
          if (clampedCosine < gl_LightSource[0].spotCosCutoff)
          // outside of spotlight cone?
          {
             attenuation = 0.0;
          }
          else
          {
             attenuation = attenuation * pow(clampedCosine, gl_LightSource[0].spotExponent);
          }
       }
    }
	
	vec3 ambientLighting = vec3(gl_LightModel.ambient)
       * vec3(texColor);//* vec3(gl_FrontMaterial.emission);

    vec3 diffuseReflection = 1.//attenuation
       * vec3(gl_LightSource[0].diffuse)
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
          * vec3(gl_LightSource[0].specular)
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
