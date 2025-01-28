const std = @import("std");
const vk = @import("vulkan");
const zlm = @import("zlm").SpecializeOn(f32);
const Texture = @import("texture.zig").Texture;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const VoxelObject = struct {
    transform_buffer: GraphicsContext.Buffer(zlm.Mat4),
    palette: Texture([4]u8, 1),
    voxels: Texture(u8, 3),

    fn deinit(self: *const VoxelObject, gc: *const GraphicsContext) void {
        self.transform_buffer.deinit(gc);
        self.palette.deinit(gc);
        self.voxels.deinit(gc);
    }
};

pub const VoxelObjectStore = struct {
    const MAX_OBJECTS = 100;

    gc: *const GraphicsContext,
    command_pool: vk.CommandPool,
    allocator: std.mem.Allocator,
    descriptor_set_layout: vk.DescriptorSetLayout,

    object_count: usize = 0,
    voxel_objects: [MAX_OBJECTS]VoxelObject,
    descriptor_sets: [MAX_OBJECTS]vk.DescriptorSet,
    descriptor_pool: vk.DescriptorPool,

    pub const Ref = struct {
        index: usize,
    };

    pub fn init(gc: *const GraphicsContext, command_pool: vk.CommandPool, allocator: std.mem.Allocator) !VoxelObjectStore {
        const descriptor_set_layout = try gc.vkd.createDescriptorSetLayout(gc.dev, &.{
            .binding_count = 3,
            .flags = .{},
            .p_bindings = &.{ .{
                .stage_flags = vk.ShaderStageFlags{ .fragment_bit = true },
                .binding = 1,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
            }, .{
                .stage_flags = vk.ShaderStageFlags{ .fragment_bit = true },
                .binding = 2,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = 1,
            }, .{
                .stage_flags = vk.ShaderStageFlags{ .fragment_bit = true },
                .binding = 3,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = 1,
            } },
        }, null);
        errdefer gc.vkd.destroyDescriptorSetLayout(gc.dev, descriptor_set_layout, null);

        const transform_descriptor_size: vk.DescriptorPoolSize = .{ .type = .uniform_buffer, .descriptor_count = MAX_OBJECTS };
        const image_descriptor_size: vk.DescriptorPoolSize = .{ .type = .combined_image_sampler, .descriptor_count = MAX_OBJECTS * 2 };
        const descriptor_pool_info: vk.DescriptorPoolCreateInfo = .{ .max_sets = MAX_OBJECTS, .pool_size_count = 2, .p_pool_sizes = &.{ transform_descriptor_size, image_descriptor_size } };
        const descriptor_pool = try gc.vkd.createDescriptorPool(gc.dev, &descriptor_pool_info, null);
        errdefer gc.vkd.destroyDescriptorPool(gc.dev, descriptor_pool, null);

        return VoxelObjectStore{ .gc = gc, .command_pool = command_pool, .allocator = allocator, .descriptor_set_layout = descriptor_set_layout, .descriptor_pool = descriptor_pool, .voxel_objects = undefined, .descriptor_sets = undefined };
    }

    pub fn deinit(self: *const VoxelObjectStore) void {
        for (0..self.object_count) |i| {
            self.voxel_objects[i].deinit(self.gc);
        }
        self.gc.vkd.destroyDescriptorPool(self.gc.dev, self.descriptor_pool, null);
        self.gc.vkd.destroyDescriptorSetLayout(self.gc.dev, self.descriptor_set_layout, null);
    }

    pub fn getObjectMut(self: *VoxelObjectStore, ref: Ref) *VoxelObject {
        return &self.voxel_objects[ref.index];
    }

    pub fn createEmpty(self: *VoxelObjectStore, size: [3]u32) !VoxelObjectStore.Ref {
        const object_ref = Ref{ .index = self.object_count };

        var transform_buffer = try self.gc.allocateBuffer(zlm.Mat4, 1, .{ .uniform_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        errdefer transform_buffer.deinit(self.gc);
        _ = try transform_buffer.map(self.gc);
        try transform_buffer.write(self.gc, &.{zlm.Mat4.identity});
        const palette = try Texture([4]u8, 1).init(self.gc, .{255}, .{});
        errdefer palette.deinit(self.gc);
        const voxels = try Texture(u8, 3).init(self.gc, size, .{ .format = .r8_unorm });
        errdefer voxels.deinit(self.gc);

        self.voxel_objects[self.object_count] = VoxelObject{
            .palette = palette,
            .voxels = voxels,
            .transform_buffer = transform_buffer,
        };
        const descriptor_allocate_info: vk.DescriptorSetAllocateInfo = .{
            .p_set_layouts = &.{self.descriptor_set_layout},
            .descriptor_pool = self.descriptor_pool,
            .descriptor_set_count = 1,
        };
        try self.gc.vkd.allocateDescriptorSets(self.gc.dev, &descriptor_allocate_info, self.descriptor_sets[self.object_count..].ptr);
        self.updateDescriptorSet(&self.voxel_objects[self.object_count], self.descriptor_sets[self.object_count]);
        self.object_count += 1;

        return object_ref;
    }

    pub fn createSphere(self: *VoxelObjectStore, object_size: [3]u32, radius: f32) !VoxelObjectStore.Ref {
        const object_ref = try self.createEmpty(object_size);
        const object = self.getObjectMut(object_ref);
        var staging_buffer = try object.voxels.createStagingBuffer(self.gc, self.allocator);
        defer staging_buffer.deinit(self.gc);

        var rand = std.rand.DefaultPrng.init(0);
        for (0..staging_buffer.shape()[0]) |x| {
            for (0..staging_buffer.shape()[1]) |y| {
                for (0..staging_buffer.shape()[2]) |z| {
                    const xf = @as(f32, @floatFromInt(x)) + 0.5 - 16;
                    const yf = @as(f32, @floatFromInt(y)) + 0.5 - 16;
                    const zf = @as(f32, @floatFromInt(z)) + 0.5 - 16;
                    const dist_from_origin: f32 = std.math.sqrt(xf * xf + yf * yf + zf * zf);
                    staging_buffer.at(&.{ x, y, z }).* = rand.random().int(u8) * @as(u8, @intFromBool(dist_from_origin < radius));
                }
            }
        }
        try object.voxels.writeStagingBuffer(self.gc, self.command_pool, staging_buffer);
        return object_ref;
    }

    fn updateDescriptorSet(self: *const VoxelObjectStore, object: *const VoxelObject, descriptor_set: vk.DescriptorSet) void {
        const transform_write_descriptor = vk.WriteDescriptorSet{
            .descriptor_type = .uniform_buffer,
            .dst_set = descriptor_set,
            .dst_binding = 1,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .p_buffer_info = &.{object.transform_buffer.getBufferInfo()},
            .p_image_info = &[_]vk.DescriptorImageInfo{},
            .p_texel_buffer_view = &[_]vk.BufferView{},
        };
        const voxels_write_descriptor = vk.WriteDescriptorSet{
            .descriptor_type = .combined_image_sampler,
            .dst_set = descriptor_set,
            .dst_binding = 2,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .p_buffer_info = undefined,
            .p_image_info = @ptrCast(&object.voxels.descriptor),
            .p_texel_buffer_view = @ptrCast(&object.voxels.view),
        };
        const voxel_palette_write_descriptor = vk.WriteDescriptorSet{
            .descriptor_type = .combined_image_sampler,
            .dst_set = descriptor_set,
            .dst_binding = 3,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .p_buffer_info = undefined,
            .p_image_info = @ptrCast(&object.palette.descriptor),
            .p_texel_buffer_view = @ptrCast(&object.palette.view),
        };
        self.gc.vkd.updateDescriptorSets(self.gc.dev, 3, &.{ transform_write_descriptor, voxels_write_descriptor, voxel_palette_write_descriptor }, 0, null);
    }

    pub fn getDescriptorSets(self: *const VoxelObjectStore) []const vk.DescriptorSet {
        return self.descriptor_sets[0..self.object_count];
    }
};
