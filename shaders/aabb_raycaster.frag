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

bool rayCubeIntersect(vec3 cubeMin, vec3 cubeMax, vec3 ro, vec3 rd_inv, out float outTmin, out float outTmax, out vec3 outNormal) {
    vec3 t0s = (cubeMin - ro) * rd_inv;
    vec3 t1s = (cubeMax - ro) * rd_inv;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger = max(t0s, t1s);

    float tmin = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
    outTmax = min(tbigger.x, min(tbigger.y, tbigger.z));
    outTmin = max(tmin,0.0);
    outNormal = max(-2*step(t0s, t1s)+1, 0.2) * (step(tsmaller.zxy, tsmaller.xyz) * step(tsmaller.yzx, tsmaller.xyz));
    return outTmax >= outTmin;
}

float getVoxel(vec3 pos, vec3 gridSize, vec3 halfGridSize) {
  vec3 texCoord = (pos + halfGridSize) / gridSize;
  return texture(voxels, texCoord).r;
}

struct Intersection {
  vec3 pos;
  vec3 normal;
  float voxel;
}; 

struct MaybeIntersection {
  Intersection i;
  bool intersects;
};

MaybeIntersection castRay(vec3 origin, vec3 ray) {
  vec3 gridSize = textureSize(voxels, 0);
  vec3 halfGridSize = gridSize/2;
  Intersection intersection;
  MaybeIntersection maybe_intersection;
  maybe_intersection.intersects = false;

  float tMin;
  float tMaxOfBoundingBox;
  vec3 bboxNormal;
  if(!rayCubeIntersect(-halfGridSize, halfGridSize, origin, 1/ray, tMin, tMaxOfBoundingBox, bboxNormal)) { return maybe_intersection; }
  
  vec3 startPos = origin + ray * tMin;
  vec3 voxelPos = floor(startPos);
  vec3 grid_delta = sign(ray);

  vec3 tDelta = min(1/abs(ray), 1000000);
  vec3 tMax = abs(((voxelPos + max(grid_delta, 0)) - startPos) / ray);

  int iterations = 0;
  
  vec3 normal = bboxNormal;
  float tIntersection = -1;

  intersection.voxel = getVoxel(voxelPos, gridSize, halfGridSize);
  if (intersection.voxel > 0) {
    maybe_intersection.intersects = true;
    intersection.pos = startPos;
    intersection.normal = bboxNormal;
    maybe_intersection.i = intersection;
  } else {
    for (int i = 0; i < 500; ++i) {
      vec3 cmp = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
      voxelPos += grid_delta * cmp;

      vec3 tVec = tMax * cmp;
      tIntersection = max(max(tVec.x, tVec.y), tVec.z);
      if (tIntersection > tMaxOfBoundingBox - tMin) { break; }

      intersection.voxel = getVoxel(voxelPos, gridSize, halfGridSize);
      if(intersection.voxel > 0) { 
        maybe_intersection.intersects = true;
        intersection.pos = startPos + ray * (tIntersection);
        intersection.normal = -1 * grid_delta * cmp;
        maybe_intersection.i = intersection;
        break; 
      }

      tMax += tDelta * cmp;
    }
  } 

  
  return maybe_intersection;
}

void main() {
  
  vec4 projected_near = vec4(gl_FragCoord.xy / (uniforms.screen_size/2.0) - vec2(1.0), 0.0, 1.0);
  projected_near.y *= -1.0;
  
  mat4x4 inverseCameraMatrix = inverse(uniforms.proj * uniforms.view * voxel_transform.mat);
  vec4 near4 = (inverseCameraMatrix * projected_near); 
  vec4 far4 = near4 + inverseCameraMatrix[2];
  vec3 near = near4.xyz / near4.w;
  vec3 far = far4.xyz / far4.w;
  vec3 cameraPos = near;
  vec3 ray = normalize(far-near);
  
  MaybeIntersection maybe_intersection = castRay(cameraPos, ray);
  if (maybe_intersection.intersects) {
    Intersection intersection = maybe_intersection.i;
    
    vec3 lightPos = vec3(20, 20, 20);
    f_color = texture(palette, intersection.voxel) + 0.1 * vec4(intersection.normal, 0.0);
    gl_FragDepth = distance(cameraPos, intersection.pos) / distance(cameraPos, far);
  } else {
    discard;
  }
}

