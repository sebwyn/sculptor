#version 450

layout(location=0) out vec4 f_color;

layout(binding=1) uniform uniform_buffer {
  vec2 dimensions; 
  vec3 pos;
  float near;
} uniforms; 


float circle(vec3 sphere_center, float radius, vec3 pos) {
  return distance(pos, sphere_center) - radius;
}

float sdf(vec3 pos) {
  return circle(vec3(0.0), 1.0, pos);
}

vec3 calcNormal(vec3 p) { // for function f(p)
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdf(p+h.xyy) - sdf(p-h.xyy),
                           sdf(p+h.yxy) - sdf(p-h.yxy),
                           sdf(p+h.yyx) - sdf(p-h.yyx) ) );
}

vec4 raymarch(vec3 pos, vec3 ray) {
  const vec3 light_pos = vec3(-5, 5, -5);
  const vec3 light_dir = normalize(vec3(1.0, -1.0, 1.0));
  const vec3 light_color = vec3(1.0);

  int i = 0;
  while (i < 15) {
    float dist = sdf(pos);
    pos += ray * dist;
    if ( dist < 0.0001 ) {
      vec3 surface_normal = calcNormal(pos);
      vec3 light_ray = normalize(light_pos - pos);
      vec3 shading = light_color * (max(dot(surface_normal, light_ray), 0.0) + 0.05);
      return vec4(shading, 1.0);
    }
    i += 1;
  }
  return vec4(0.0);
}


void main() {
  float aspect_ratio = uniforms.dimensions.y / uniforms.dimensions.x;

  vec3 pos = uniforms.pos; 
  vec3 ray = normalize(vec3((gl_FragCoord.xy - uniforms.dimensions / 2) / uniforms.dimensions / 2, 1.0));
  ray.y *= -aspect_ratio;
  
  f_color = raymarch(pos, ray);
}

