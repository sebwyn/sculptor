#version 450

layout(location=0) out vec4 f_color;

layout(binding=0) uniform camera_info {
  vec2 dimensions; 
  vec3 pos;
  float near;
} camera;


float circle(vec3 pos) {
  return sqrt(dot(pow(pos, vec3(2.0)), vec3(1.0)));
}

bool raymarch(vec3 pos, vec3 ray) {
  int i = 0;
  while (i < 30) {
    float dist = circle(pos);
    if ( circle(pos) < 0.01 ) {
      return true;
    }
    pos += ray * dist;
  }
  return false;
}


void main() {
  vec3 pos = camera.pos; 
  vec3 ray = normalize(vec3(gl_FragCoord.xy / camera.dimensions, 1.0));

  if (raymarch(pos, ray)) {
    f_color = vec4(camera.pos.xy, 1.0, 1.0);
  } else {
    f_color = vec4(0.0, 0.0, 0.0, 1.0);
  }
}

