## TODO

 - Get a fpv camera working
 - Raycast into 3d texture with simple AABB raycasting
 - Load a .vox file, solve whatever problems come up, maybe work with materials

## Voxel Rendering Techniques and papers

 - The essential AABB raycasting algorithm [paper](https://www.researchgate.net/publication/2611491_A_Fast_Voxel_Traversal_Algorithm_for_Ray_Tracing) 
 - Different raycasting approach that might be better for dynamic scenes or non grid aligned voxels [paper](https://jcgt.org/published/0007/03/04/paper-lowres.pdf)

## More Sources of information

 - [Voxel Game Dev](https://www.reddit.com/r/VoxelGameDev/)
 - [Game Engine Sub](https://www.reddit.com/r/gameenginedevs/)
 - [Procedural Generation](https://www.reddit.com/r/proceduralgeneration/) 
 - [John Lin's Blog](https://voxely.net/blog/)

## Other published voxel engines

 - [Iolite](https://iolite-engine.com) 
 - [Unreal does voxels](https://voxelplugin.com)

 - Teardown approach:
    - Render a g-buffer (depth and color) by raycasting into all visible objects, each object is a 3d texture
    rays are first cast from the camera to where the object is positioned in space, and then AABB ray casting is used within the object
    - For lighting all the objects are splatted into an axis aligned voxel octree where all ambient occlusion, lighting, and spectral occlusion rays can
    be cast, this is described as an octree but in reality is just mip levels, the difference is there may be a way to be more efficient then teardown with
    this datastructure
    - Teardown is also a really interesting source for how physics can be implemented



## Youtube channels with voxel engines

 - [Xima](https://www.youtube.com/@xima1): Minecraft clone, but with raytraced looking graphics
 - [Aurailus](https://www.youtube.com/@Aurailus): classic minecraft like rasterization engine with optimizations
 - [Douglas](https://www.youtube.com/@DouglasDwyer): Small voxels w/ physics goes into quite a bit of detail
 - [Grant Kot](https://www.youtube.com/@GrantKot): big into full voxel/particle water simulation
 - [Frozein](https://www.youtube.com/@frozein): Made a super pretty voxel game on a planet, is currently making an improved raycast engine, not too much content
 - [Gabe Rundlett](https://www.youtube.com/@GabeRundlett): some great content on dynamic raycast engines

## Aesthetic Inspiration

 - [Voxel Art Sub](https://www.reddit.com/r/VOXEL/)

## Other inspiration

 - [Rujik the comatose](https://www.youtube.com/@RujiKtheComatose): does some super sick SDF procedural animation all in game maker with his game "Critter Crosser"
