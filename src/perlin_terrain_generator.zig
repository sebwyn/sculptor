const std = @import("std");
const perlin = @import("perlin");
const VoxelObjectStore = @import("voxel_object_store.zig").VoxelObjectStore;


pub fn generateTerrain(
    size: [3]u32,
    allocator: std.mem.Allocator,
    voxel_store: *VoxelObjectStore,
    opts: struct {
        perlin_cutoff: f32 = 0.8,
        perlin_cycles: f32 = 20.0,
    }
) !void {
    const voxel_ref = try voxel_store.createEmpty(size);
    const voxel_object = voxel_store.getObjectMut(voxel_ref);
    
    const gc = voxel_store.gc;

    const staging_buffer = try voxel_object.voxels.createStagingBuffer(gc, allocator);
    defer staging_buffer.deinit(gc);
    
    for (0..size[0]) |x| {
        for (0..size[1]) |y| {
            for (0..size[2]) |z| {
                const fx: f32 = opts.perlin_cycles * @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(size[0]));
                const fy: f32 = opts.perlin_cycles * @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(size[1]));
                const fz: f32 = opts.perlin_cycles * @as(f32, @floatFromInt(z)) / @as(f32, @floatFromInt(size[2]));

                const perlin_value = perlin.noise(f32, .{ .x = fx, .y = fy, .z = fz, }) + 0.5;

                const voxel: u8 = if (perlin_value > opts.perlin_cutoff) 1 else 0;
                staging_buffer.at(&[_]usize{ x, y, z}).* = voxel;
            }
        }
    }
    _ = try voxel_object.voxels.writeStagingBuffer(gc, voxel_store.command_pool, staging_buffer); 
    
    const palette_buffer = try voxel_object.palette.createStagingBuffer(gc, allocator);
    defer palette_buffer.deinit(gc);

    for (0..255) |i| { palette_buffer.slice(&.{ i }).write(&[_]u8{ 255, 255, 255, 255 }); }
    _ = try voxel_object.palette.writeStagingBuffer(gc, voxel_store.command_pool, palette_buffer);
}

pub fn generateHeightmapTerrain(
    size: [3]u32,
    allocator: std.mem.Allocator,
    voxel_store: *VoxelObjectStore,
    opts: struct {
        perlin_cutoff: f32 = 0.8,
        perlin_cycles: f32 = 20.0,
        peak_height: f32 = 0.1,
        seed: f32 = 10.2,
    }
) !void {
    const voxel_ref = try voxel_store.createEmpty(size);
    const voxel_object = voxel_store.getObjectMut(voxel_ref);
    
    const gc = voxel_store.gc;

    const staging_buffer = try voxel_object.voxels.createStagingBuffer(gc, allocator);
    defer staging_buffer.deinit(gc);
    
    for (0..size[0]) |x| {
        for (0..size[2]) |z| {
            const fx: f32 = opts.perlin_cycles * @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(size[0]));
            const fz: f32 = opts.perlin_cycles * @as(f32, @floatFromInt(z)) / @as(f32, @floatFromInt(size[2]));

            const perlin_value = std.math.clamp(perlin.noise(f32, .{ .x = fx, .y = opts.seed, .z = fz, }) + 0.5, 0.0, 1.0);
            // std.debug.print("{d} Too big??: {d}\n", .{ perlin_value, perlin_value * @as(f32, @floatFromInt(size[1])) * opts.peak_height });
            const highest_index: u32 = @intFromFloat(@floor(perlin_value * @as(f32, @floatFromInt(size[1])) * opts.peak_height));
            

            for (0..size[1]) |y| {
                const voxel: u8 = if (y < highest_index) 1 else 0;
                staging_buffer.at(&[_]usize{ x, y, z}).* = voxel;
            }
        }
    }
    _ = try voxel_object.voxels.writeStagingBuffer(gc, voxel_store.command_pool, staging_buffer); 
    
    const palette_buffer = try voxel_object.palette.createStagingBuffer(gc, allocator);
    defer palette_buffer.deinit(gc);

    for (0..255) |i| { palette_buffer.slice(&.{ i }).write(&[_]u8{ 255, 255, 255, 255 }); }
    _ = try voxel_object.palette.writeStagingBuffer(gc, voxel_store.command_pool, palette_buffer);
}

