#version 450

const float PI = 3.14159265;

layout(location=0) out vec4 f_color;

layout(set=0, binding=0) uniform uniform_buffer {
  mat4x4 view;
  mat4x4 proj;
  vec2 screen_size;
} uniforms; 

layout(set=1, binding=1) uniform vox_transform {
  mat4x4 mat;
} voxel_transform;

layout(set=1, binding=2) uniform sampler3D voxels;
layout(set=1, binding=3) uniform sampler1D palette; 

bool rayCubeIntersect(vec3 cubeMin, vec3 cubeMax, vec3 ro, vec3 rd_inv, out float outTmin, out vec3 outNormal) {
    vec3 t0s = (cubeMin - ro) * rd_inv;
    vec3 t1s = (cubeMax - ro) * rd_inv;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger = max(t0s, t1s);

    float tmin = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
    float tmax = min(tbigger.x, min(tbigger.y, tbigger.z));
    outTmin = max(tmin,0.0);
    outNormal = max(-2*step(t0s, t1s)+1, 0.2) * (step(tsmaller.zxy, tsmaller.xyz) * step(tsmaller.yzx, tsmaller.xyz));
    return tmax >= max(tmin, 0.0);
}

bool isVoxel(vec3 pos, vec3 gridSize, vec3 halfGridSize) {
  vec3 v = (pos + halfGridSize) / gridSize;
  return -0.0 <= v.x && v.x <= 1.0 &&
  -0.0 <= v.y && v.y <= 1.0 &&
  -0.0 <= v.z && v.z <= 1.0; 
}

float getVoxel(vec3 pos, vec3 gridSize, vec3 halfGridSize) {
  vec3 texCoord = (pos + halfGridSize) / gridSize;
  return texture(voxels, texCoord).r;
}

void main() {
  vec3 gridSize = textureSize(voxels, 0);
  vec3 halfGridSize = gridSize/2;
  
  vec4 projected_near = vec4(gl_FragCoord.xy / (uniforms.screen_size/2.0) - vec2(1.0), 0.0, 1.0);
  projected_near.y *= -1.0;
  
  mat4x4 inverseCameraMatrix = inverse(uniforms.proj * uniforms.view);
  vec4 near4 = (inverseCameraMatrix * projected_near); 
  vec4 far4 = near4 + inverseCameraMatrix[2];
  vec3 near = near4.xyz / near4.w;
  vec3 far = far4.xyz / far4.w;
  vec3 cameraPos = near;
  vec3 ray = normalize(far-near);

  float tMin;
  vec3 bboxNormal;
  if(!rayCubeIntersect(-halfGridSize, halfGridSize, cameraPos, 1/ray, tMin, bboxNormal)) { f_color = vec4(0.0); return; }
  
  vec3 startPos = cameraPos + ray * tMin;
  vec3 voxelPos = floor(startPos);
  vec3 grid_delta = sign(ray);

  vec3 tDelta = min(1/abs(ray), 1000000);
  vec3 tMax = abs(((voxelPos + max(grid_delta, 0)) - startPos) / ray);

  int iterations = 0;
  float voxel = getVoxel(voxelPos, gridSize, halfGridSize);
  vec3 normal = bboxNormal;
  float tIntersection = -1;
  for (int i = 0; i < 300; ++i) {
    vec3 cmp = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
    voxelPos += grid_delta * cmp;
    voxel = getVoxel(voxelPos, gridSize, halfGridSize);
    if (!isVoxel(voxelPos, gridSize, halfGridSize)) {
      break;
    }
    if(voxel > 0) {
      vec3 tVec = tMax * cmp;
      tIntersection = max(max(tVec.x, tVec.y), tVec.z);
      normal = -1 * grid_delta * cmp;
      break;
    }
    tMax += tDelta * cmp;
  }

  vec3 lightPos = vec3(20, 20, 20);
  vec3 pos = startPos + ray * tIntersection;
  
  if (tIntersection > 0) {
    f_color = texture(palette, voxel) + 0.1 * vec4(normal, 0.0);
    // gl_FragDepth = distance(cameraPos, pos) / distance(cameraPos, far);
  } else {
    f_color = vec4(0.0);
  }
}

