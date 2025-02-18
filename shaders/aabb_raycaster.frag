#version 450

const float PI = 3.14159265;
const uint MAX_STEPS = 500;

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

vec4 getVoxel(vec3 pos, vec3 gridSize, vec3 halfGridSize) {
  vec3 texCoord = (pos + halfGridSize) / gridSize;
  float palette_index = texture(voxels, texCoord).r;
  if (palette_index > 0) {
    return texture(palette, palette_index);
  } else {
    return vec4(0.0);
  }
}

const uint Diffuse = 1u;
const uint Refract = 2u;

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

MaybeIntersection castRay(vec3 origin, vec3 ray) {
  vec3 gridSize = textureSize(voxels, 0);
  vec3 halfGridSize = gridSize/2;

  Intersection intersection;
  MaybeIntersection maybeIntersection;
  maybeIntersection.intersects = false;

  float bboxTMin;
  float bboxTMax;
  vec3 bboxNormal;
  if(!rayCubeIntersect(-halfGridSize, halfGridSize, origin, 1/ray, bboxTMin, bboxTMax, bboxNormal)) { return maybeIntersection; }
  
  vec3 startPos = origin + ray * bboxTMin;
  vec3 voxelPos = floor(startPos);
  vec3 grid_delta = sign(ray);

  vec3 tDelta = min(1/abs(ray), 1000000);
  vec3 tMax = abs(((voxelPos + max(grid_delta, 0)) - startPos) / ray);

  vec3 normal = bboxNormal;

  intersection.distance = bboxTMin;
  intersection.voxel_color = getVoxel(voxelPos, gridSize, halfGridSize);
  if (intersection.voxel_color.z > 0.99) {
    intersection.pos = startPos;
    intersection.normal = bboxNormal;

    maybeIntersection.intersects = true;
    maybeIntersection.i = intersection;
    return maybeIntersection;
  }

  for (int i = 0; i < MAX_STEPS; ++i) {
    vec3 step_axis = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
    vec3 tVec = tMax * step_axis;
    intersection.distance = bboxTMin + max(max(tVec.x, tVec.y), tVec.z);
    if (intersection.distance > bboxTMax - 0.0002) { break; }

    tMax += tDelta * step_axis;

    voxelPos += grid_delta * step_axis;
    intersection.voxel_color = getVoxel(voxelPos, gridSize, halfGridSize);

    if(intersection.voxel_color.a > 0.0) { 
      intersection.pos = origin + ray * intersection.distance;
      intersection.normal = -1 * grid_delta * step_axis;
      intersection.type = intersection.voxel_color.a > 0.99 ? Diffuse : Refract;
      intersection.step_count = i;
      maybeIntersection.i = intersection;
      maybeIntersection.intersects = true;
      break;
    }

  }


  return maybeIntersection;
}


MaybeIntersection castRayVolumetric(vec3 origin, vec3 ray) {
  vec3 gridSize = textureSize(voxels, 0);
  vec3 halfGridSize = gridSize/2;

  Intersection intersection;
  MaybeIntersection maybeIntersection;
  maybeIntersection.intersects = false;

  float bboxTMin;
  float bboxTMax;
  vec3 bboxNormal;
  if(!rayCubeIntersect(-halfGridSize, halfGridSize, origin, 1/ray, bboxTMin, bboxTMax, bboxNormal)) { return maybeIntersection; }
  
  vec3 startPos = origin + ray * bboxTMin;
  vec3 voxelPos = floor(startPos);
  vec3 grid_delta = sign(ray);

  vec3 tDelta = min(1/abs(ray), 1000000);
  vec3 tMax = abs(((voxelPos + max(grid_delta, 0)) - startPos) / ray);

  vec3 normal = bboxNormal;

  vec3 accumulatedColor = vec3(0.0);
  float alphaWeights = 0.0;

  intersection.distance = bboxTMin;
  intersection.voxel_color = getVoxel(voxelPos, gridSize, halfGridSize);
  if (intersection.voxel_color.z > 0.99) {
    intersection.pos = startPos;
    intersection.normal = bboxNormal;

    maybeIntersection.intersects = true;
    maybeIntersection.i = intersection;
    return maybeIntersection;
  }

  for (int i = 0; i < MAX_STEPS; ++i) {
    vec3 step_axis = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
    vec3 tVec = tMax * step_axis;
    intersection.distance = bboxTMin + max(max(tVec.x, tVec.y), tVec.z);
    if (intersection.distance > bboxTMax - 0.0002) { break; }

    tMax += tDelta * step_axis;

    voxelPos += grid_delta * step_axis;
    vec4 voxelColor = getVoxel(voxelPos, gridSize, halfGridSize);

    alphaWeights += voxelColor.a;
    accumulatedColor += voxelColor.a * voxelColor.rgb;

    if(voxelColor.a > 0.0) { 
      intersection.pos = origin + ray * intersection.distance;
      intersection.normal = -1 * grid_delta * step_axis;
      maybeIntersection.intersects = true;
    }

    if (voxelColor.a > 0.99) { break; }

  }

  intersection.voxel_color = vec4(accumulatedColor / alphaWeights, 1.0);
  maybeIntersection.i = intersection;

  return maybeIntersection;
}

float FresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incident, float f0, float f90)
{
        // Schlick aproximation
        float r0 = (n1-n2) / (n1+n2);
        r0 *= r0;
        float cosX = -dot(normal, incident);
        if (n1 > n2)
        {
            float n = n1/n2;
            float sinT2 = n*n*(1.0-cosX*cosX);
            // Total internal reflection
            if (sinT2 > 1.0)
                return f90;
            cosX = sqrt(1.0-sinT2);
        }
        float x = 1.0-cosX;
        float ret = r0+(1.0-r0)*x*x*x*x*x;
 
        // adjust reflect multiplier for object reflectivity
        return mix(f0, f90, ret);
}

void main() {
  vec4 projected_near = vec4(gl_FragCoord.xy / (uniforms.screen_size/2.0) - vec2(1.0), 0.0, 1.0);
  projected_near.y *= -1.0;
  
  mat4x4 inverseViewMat = inverse(uniforms.view * voxel_transform.mat);

  mat4x4 inverseCameraMatrix = inverseViewMat * inverse(uniforms.proj);
  vec4 near4 = (inverseCameraMatrix * projected_near); 
  vec4 far4 = near4 + inverseCameraMatrix[2];
  vec3 far = far4.xyz / far4.w;
  vec3 cameraPos = inverseViewMat[3].xyz;
  vec3 ray = normalize(far-cameraPos);
  
  vec3 lightPos = vec3(30, 30, 30);

  MaybeIntersection maybeIntersection = castRay(cameraPos, ray);
  Intersection intersection = maybeIntersection.i;
  float brightness = 0.05;
  MaybeIntersection light_occlusion_intersection = castRay(intersection.pos + intersection.normal * 0.0001, normalize(lightPos - intersection.pos));
  if (!(light_occlusion_intersection.intersects && light_occlusion_intersection.i.distance < distance(intersection.pos, lightPos))) {
    brightness = dot(normalize(lightPos - intersection.pos), intersection.normal);
  }
  if (maybeIntersection.intersects) {
    // for visualizing the number of steps taken in raytracing
    // f_color = vec4(vec3(float(intersection.step_count) / 100.0), 1.0);
    // gl_FragDepth = distance(cameraPos, intersection.pos) / distance(cameraPos, far);

    if (intersection.type == Diffuse) {
      float brightness = 0.05;
      MaybeIntersection light_occlusion_intersection = castRay(intersection.pos + intersection.normal * 0.0001, normalize(lightPos - intersection.pos));
      if (!(light_occlusion_intersection.intersects && light_occlusion_intersection.i.distance < distance(intersection.pos, lightPos))) {
        brightness = dot(normalize(lightPos - intersection.pos), intersection.normal);
      }
      vec4 litColor =vec4(intersection.voxel_color.rgb * brightness, 1.0); 
      f_color = litColor;
    } else {
      MaybeIntersection reflection_ray = castRay(intersection.pos + intersection.normal * 0.0001, reflect(ray, intersection.normal));
      vec4 reflectionColor = reflection_ray.intersects ? reflection_ray.i.voxel_color : vec4(0.0);
     
      MaybeIntersection refraction_ray = castRayVolumetric(intersection.pos - intersection.normal * 0.0001, refract(ray, intersection.normal, 1.0 / 1.333));
      vec4 refractionColor = refraction_ray.intersects ? refraction_ray.i.voxel_color : vec4(0.0);
      MaybeIntersection light_occlusion_intersection = castRay(refraction_ray.i.pos + refraction_ray.i.normal * 0.0001, normalize(lightPos - refraction_ray.i.pos));
      if (!(light_occlusion_intersection.intersects && light_occlusion_intersection.i.distance < distance(intersection.pos, lightPos))) {
        brightness = dot(normalize(lightPos - intersection.pos), intersection.normal);
      } else {
        brightness = 0.05;
      }
     
      float reflectAmount = FresnelReflectAmount(1.0, 1.333, intersection.normal, ray, 0.02, 1.0);
      f_color = vec4(refractionColor.rgb * brightness, 1.0);
     
     
    }
  } else {
    discard;
  }
}

