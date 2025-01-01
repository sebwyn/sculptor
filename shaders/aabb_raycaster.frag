#version 450

const float PI = 3.14159265;

layout(location=0) out vec4 f_color;

layout(binding=0) uniform uniform_buffer {
  mat4x4 view;
  mat4x4 proj;
  vec2 screen_size;
} uniforms; 

layout(binding=1) uniform sampler3D voxels;

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
  const vec3 light_pos = vec3(-5, 5, 5);
  const vec3 light_color = vec3(1.0);

  int i = 0;
  while (i < 15) {
    float dist = sdf(pos);
    pos += ray * dist;
    if ( dist < 0.0001 ) {
      vec3 surface_normal = calcNormal(pos);
      vec3 light_ray = normalize(light_pos - pos);
      vec3 shading = light_color * (max(dot(light_ray, surface_normal), 0.0) + 0.05);
      return vec4(shading, 1.0);
    }
    i += 1;
  }
  return vec4(0.0);
}


void main() {
  vec4 projected_near = vec4(gl_FragCoord.xy / (uniforms.screen_size/2.0) - vec2(1.0), 0.0, 1.0);
  projected_near.y *= -1.0;
  
  mat4x4 inverse_camera_matrix = inverse(uniforms.proj * uniforms.view);
  vec3 worldspace_near = (inverse_camera_matrix * projected_near).xyz; 

  vec3 camera_pos = inverse(uniforms.view)[3].xyz;
  vec3 ray = normalize(worldspace_near - camera_pos);
  
  f_color = vec4(texture(voxels, vec3((projected_near.x + 1)/2, (projected_near.y + 1)/2, 0.5)).r);
}

