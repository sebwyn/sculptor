#version 450

const float PI = 3.14159265;
const uint MAX_STEPS = 500;

layout(location=0) out vec4 f_color;

layout(set=0, binding=0) uniform uniform_buffer { mat4x4 view; mat4x4 proj; vec2 screen_size; } uniforms; 
layout(set=1, binding=1) uniform vox_transform { mat4x4 mat; } voxel_transform;

layout(set=1, binding=2) buffer Voxels { uint data[]; } voxels;
layout(set=1, binding=3) uniform VoxelDetails { vec3 voxelSize; } voxelDetails; 
layout(set=1, binding=4) uniform sampler1D palette; 

struct Ray {
  vec3 origin;
  vec3 dir;
};

struct Intersection {
  float distance;
  vec3 pos;
  vec3 normal;
  vec4 voxel_color;
  uint type;
  uint step_count;
}; 

struct MaybeIntersection {
  Intersection i;
  bool intersects;
};

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

struct ChildDescriptor {
  uint childIndex; 
  bool far; 
  uint validMask; 
  uint leafMask; 
};

ChildDescriptor getDescriptor(uint i) {
  uint descriptorBytes = voxels.data[i];
  
  ChildDescriptor descriptor;
  descriptor.leafMask = descriptorBytes & 0xFF;
  descriptor.validMask = (descriptorBytes & 0xFF00) >> 8;
  descriptor.far = (descriptorBytes & 0x10000) >> 16 == 1;
  descriptor.childIndex = (descriptorBytes & 0xFFFF0000) >> 17;

  return descriptor;
}

MaybeIntersection raycastSVO(Ray ray) {
  Intersection intersection;
  MaybeIntersection maybeIntersection;
  maybeIntersection.intersects = false;

  vec3 invRayDir = 1 / ray.dir;
  
  vec3 halfGridSize = voxelDetails.voxelSize / 2;
  float t_min;
  float t_max;
  vec3 normal;
  if(!rayCubeIntersect(-halfGridSize, halfGridSize, ray.origin, invRayDir, tMin, tMax, normal)) { return maybeIntersection; }

  //nvidia code starts by removing small ray dir components
  //nvidia computes tx, ty, tz coefficients
  //nvidia initializes octant mask based on ray direction, this will prob be used for traversal
    
  int s_max = 23;
  float epsilon = exp2(float(s_max));
  ivec2 stack[s_max + 1];

  vec3 t_coeff = vec3(0.0);
  vec3 t_bias = vec3(0.0);
  
  vec3 pos = ray.origin + tMin * rayDir;
  int parent = 0;
  int child_descriptor = 0;  
  float scale_exp2 = 0.5f; // exp2(scale - s_max)

  int scale = s_max - 1;
  
  //determine which octant was entered initially
  int idx = 0;
  if (1.5f * tx_coef - tx_bias > t_min) idx ˆ= 1, pos.x = 1.5f;
  if (1.5f * ty_coef - ty_bias > t_min) idx ˆ= 2, pos.y = 1.5f;
  if (1.5f * tz_coef - tz_bias > t_min) idx ˆ= 4, pos.z = 1.5f;

  while (scale < s_max) {
    if (child_descriptor == 0) { child_descriptor = voxels.data[parent]; }

    vec3 t_corner = pos * t_coeff - t_bias;

    float tc_max = min(min(t_corner.x, t_corner.y), t_corner.z);

    //TODO try and understand wtf this is doing
    int child_shift = idx ^ octant_mask;
    int child_masks = child_descriptor << child_shift;

    //handle descending into the octree
    if ((child_masks & 0x800) != 0) && t_min < t_max) {
      //terminate if the voxel is small enough
      if (tc_max * ray_size_coeff + ray_size_bias >= scale_exp2) break;

      //intersect active t-span with cube and evaluate
      float tv_max = min(tc_max, t_max);
      float half = scale_exp2 * 0.5;
    
      vec3 t_center = half * t_coeff + t_corner;
      
      if (t_min <= tv_min) {
        if ((child_masks & 0x080) == 0) break;

        if (tc_max < h)
          stack[scale] = ivec2(parent, floatBitsToInt(t_max)); //encode t_max as int to push it, will recover it later 
        h = tc_max;
        
        int ofs = child_descriptor << 17;
        //TODO handle far pointer logic here
        //revisit once I understand child masks
        parent = ofs + bitCount(child_masks & 0x7f);

        idx = 0;
        scale--;
        scale_exp2 = half;
        
        if (t_center.x > t_min) { idx ^= 1; pos.x += scale_exp2; }
        if (t_center.y > t_min) { idx ^= 2; pos.y += scale_exp2; }
        if (t_center.z > t_min) { idx ^= 4; pos.z += scale_exp2; }

        t_max = tv_max;
        child_descriptor = 0;
        continue;
      }
    }

    int step_mask = 0;
    if (t_corner.x <= tc_max) { step_mask ^= 1; pos.x -= scale_exp2; }
    if (t_corner.y <= tc_max) { step_mask ^= 2; pos.y -= scale_exp2; }
    if (t_corner.z <= tc_max) { step_mask ^= 4; pos.z -= scale_exp2; }

    t_min = tc_max; 
    idx ^= step_mask;

    if ((idx & step_mask) != 0) {
      uint differing_bits = 0;
      if ((step_mask & 1) != 0) differing_bits |= floatBitsToInt(pos.x) ^ floatBitsToInt(pos.x + scale_exp2)
      if ((step_mask & 2) != 0) differing_bits |= floatBitsToInt(pos.y) ^ floatBitsToInt(pos.y + scale_exp2)
      if ((step_mask & 4) != 0) differing_bits |= floatBitsToInt(pos.z) ^ floatBitsToInt(pos.z + scale_exp2)

      scale = (floatBitsToInt(float(differing_bits)) >> 23) - 127;
      scale_exp2 = intBitsToFloat((scale - s_max + 127) << 23)

      ivec2 stack_entry = stack[scale]
      parent = stack_entry.x;
      t_max = intBitsToFloat(stack_entry.y)

      int shx = floatBitsToInt(pos.x) >> scale;
      int shy = floatBitsToInt(pos.y) >> scale;
      int shz = floatBitsToInt(pos.z) >> scale;
      pos.x = intBitsToFloat(shx << scale);
      pos.y = intBitsToFloat(shy << scale);
      pos.z = intBitsToFloat(shz << scale);

      idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);

      h = 0.0f;
      child_descriptor = 0;
    }
  }

  return maybeIntersection;
}

Ray computeViewRay() {
  vec4 projected_near = vec4(gl_FragCoord.xy / (uniforms.screen_size/2.0) - vec2(1.0), 0.0, 1.0);
  projected_near.y *= -1.0;
 
  mat4x4 inverseViewMat = inverse(uniforms.view * voxel_transform.mat);

  mat4x4 inverseCameraMatrix = inverseViewMat * inverse(uniforms.proj);
  vec4 near4 = (inverseCameraMatrix * projected_near); 
  vec4 far4 = near4 + inverseCameraMatrix[2];
  vec3 far = far4.xyz / far4.w;
  vec3 cameraPos = inverseViewMat[3].xyz;
  vec3 dir = normalize(far-cameraPos);
 
  Ray ray;
  ray.origin = cameraPos;
  ray.dir = dir;
  return ray;
}


void main() {
  Ray ray = computeViewRay();
  MaybeIntersection maybeIntersection = raycastSVO(ray);

  if (maybeIntersection.intersects) {
    f_color = vec4(1.0);
  } else {
    discard;
  }
}

