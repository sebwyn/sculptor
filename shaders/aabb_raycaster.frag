#version 450

layout(location=0) out vec4 f_color;

layout(binding=1) uniform uniform_buffer {
  vec2 dimensions; 
  vec3 pos;
  float near;
} uniforms; 


float circle(vec3 pos) {
  return sqrt(dot(pow(pos, vec3(2.0)), vec3(1.0))) - 1;
}

vec4 raymarch(vec3 pos, vec3 ray) {
  int i = 0;
  while (i < 15) {
    float dist = circle(pos);
    if ( dist < 0.0001 ) {
      return vec4(1.0);
    }
    pos += ray * dist;
    i += 1;
  }
  return vec4(0.0);
}


void main() {
  vec3 pos = uniforms.pos; 
  vec3 ray = normalize(vec3((gl_FragCoord.xy - uniforms.dimensions / 2) / uniforms.dimensions / 2, 1.0));
  ray.y = -ray.y;
  
  f_color = raymarch(pos, ray);
  // f_color = raymarch(pos, ray);
}

