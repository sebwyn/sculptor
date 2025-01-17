const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const vk = @import("vulkan");

const Texture3dError = error{ UnsupportedDevice, UnsupportedSize, BadDataFormat };

fn textureSizeSupported(comptime dims: comptime_int, gc: *const GraphicsContext, size: [3]u32) bool {
    const device_limits = gc.vki.getPhysicalDeviceProperties(gc.pdev).limits;
    const max_texture_size = switch(dims) {
        3 => device_limits.max_image_dimension_3d,
        2 => device_limits.max_image_dimension_2d,
        1 => device_limits.max_image_dimension_1d,
        else => @panic("Texture was only expecting 1 to 3 dimenions")
    };
    return size[0] < max_texture_size and size[1] < max_texture_size and size[2] < max_texture_size;
}

fn imageTypeForDimensions(comptime dimensions: comptime_int) vk.ImageType {
    return switch (dimensions) { 
        3 => .@"3d",
        2 => .@"2d",
        1 => .@"1d",
        else => @panic("Texture was only expecting 1 to 3 dimenions") 
    };
}

fn imageViewTypeForDimensions(comptime dimensions: comptime_int) vk.ImageViewType {
    return switch (dimensions) { 
        3 => .@"3d",
        2 => .@"2d",
        1 => .@"1d",
        else => @panic("Texture was only expecting 1 to 3 dimenions") 
    };
}

fn extentForSize(comptime dimensions: comptime_int, size: [3]u32) vk.Extent3D {
    var extent: vk.Extent3D = .{ .width = size[0], .height = 1, .depth = 1 };
    switch (dimensions) {
        3 => { extent.height = size[1]; extent.depth = size[2]; },
        2 => { extent.height = size[1]; },
        1 => {},
        else => @panic("Texture was only expecting 1 to 3 dimenions")
    }
    return extent;
}

fn sizeofFormat(format: vk.Format) usize {
    return switch (format) {
        .r8_unorm => 1,
        .r8g8b8a8_srgb => 4,
        else => @panic("sizeofFormat was not expecting that format!"),
    };
}

pub fn Texture(comptime dimensions: comptime_int) type {
    return struct {
        const Self = @This();

        sampler: vk.Sampler,
        image: vk.Image,
        image_layout: vk.ImageLayout,
        descriptor: vk.DescriptorImageInfo,
        view: vk.ImageView,
        memory: vk.DeviceMemory,
        format: vk.Format,

        size: [3]u32,
        
        pub fn init(gc: *const GraphicsContext, format: vk.Format, size: [dimensions]u32) !Self {
            var verbose_size: [3]u32 = .{ 1, 1, 1 };
            for (0..size.len) |i| { verbose_size[i] = size[i]; }

            const initial_layout: vk.ImageLayout = .undefined;

            const physical_properties = gc.vki.getPhysicalDeviceFormatProperties(gc.pdev, format);
            if (!physical_properties.optimal_tiling_features.contains(.{ .transfer_dst_bit = true })) {
                return error.UnsupportedDevice;
            }

            if (!textureSizeSupported(dimensions, gc, verbose_size)) {
                return error.UnsupportedSize;
            }

            const image_create_info: vk.ImageCreateInfo = .{
                .image_type = imageTypeForDimensions(dimensions),
                .format = format,
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{ .@"1_bit" = true },
                .tiling = .optimal,
                .sharing_mode = .exclusive,
                .extent = extentForSize(dimensions, verbose_size),
                .initial_layout = initial_layout,
                .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
            };

            const image = try gc.vkd.createImage(gc.dev, &image_create_info, null);
            errdefer gc.vkd.destroyImage(gc.dev, image, null);

            const memory_requirements = gc.vkd.getImageMemoryRequirements(gc.dev, image);
            const memory = try gc.allocate(memory_requirements, .{ .device_local_bit = true });
            errdefer gc.vkd.freeMemory(gc.dev, memory, null);

            try gc.vkd.bindImageMemory(gc.dev, image, memory, 0);

            const sampler_create_info: vk.SamplerCreateInfo = .{
                .mag_filter = .nearest,
                .min_filter = .nearest,
                .mipmap_mode = .nearest,
                .address_mode_u = .clamp_to_edge,
                .address_mode_v = .clamp_to_edge,
                .address_mode_w = .clamp_to_edge,
                .mip_lod_bias = 0.0,
                .compare_op = .never,
                .compare_enable = @intFromBool(false),
                .min_lod = 0.0,
                .max_lod = 0.0,
                .max_anisotropy = 1.0,
                .anisotropy_enable = @intFromBool(false),
                .border_color = .int_transparent_black,
                .unnormalized_coordinates = @intFromBool(false),
            };
            const sampler = try gc.vkd.createSampler(gc.dev, &sampler_create_info, null);
            errdefer gc.vkd.destroySampler(gc.dev, sampler, null);

            const image_view_create_info: vk.ImageViewCreateInfo = .{ 
                .image = image, .view_type = imageViewTypeForDimensions(dimensions), 
                .format = format, 
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                    .level_count = 1,
                }, 
                .components = .{ .r = .r, .g = .g, .b = .b, .a = .a } };
            const image_view = try gc.vkd.createImageView(gc.dev, &image_view_create_info, null);
            errdefer gc.vkd.destroyImageView(gc.dev, image_view, null);

            const descriptor = vk.DescriptorImageInfo{
                .image_layout = .shader_read_only_optimal,
                .sampler = sampler,
                .image_view = image_view,
            };

            return Self {
                .sampler = sampler,
                .image = image,
                .image_layout = initial_layout,
                .descriptor = descriptor,
                .view = image_view,
                .memory = memory,
                .format = format,
                .size = verbose_size,
            };
        }

        pub fn write(self: *Self, gc: *const GraphicsContext, pool: vk.CommandPool, data: []const u8) !void {
            var expected_size = sizeofFormat(self.format);
            for (self.size) |dimension| { expected_size *= dimension; }

            if (data.len != expected_size) {
                return error.BadDataFormat;
            }

            const staging_buffer = try gc.writeStagingBuffer(u8, data);
            defer staging_buffer.deinit(gc);

            const command_buffer = try gc.beginSingleTimeCommands(pool);

            const subresource_range: vk.ImageSubresourceRange = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
                .level_count = 1,
            };

            try gc.transitionImageLayout(command_buffer, self.image, subresource_range, .undefined, .transfer_dst_optimal);

            const buffer_image_copy: vk.BufferImageCopy = .{
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .image_extent = extentForSize(dimensions, self.size),
                .buffer_offset = 0,
                .buffer_row_length = self.size[0],
                .buffer_image_height = self.size[1],
                .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            };
            gc.vkd.cmdCopyBufferToImage(command_buffer, staging_buffer.vk_handle, self.image, .transfer_dst_optimal, 1, @ptrCast(&buffer_image_copy));

            try gc.transitionImageLayout(command_buffer, self.image, subresource_range, .transfer_dst_optimal, .shader_read_only_optimal);

            try gc.endSingleTimeCommands(pool, command_buffer, gc.graphics_queue.handle);
        }

        pub fn deinit(self: Self, gc: *const GraphicsContext) void {
            gc.vkd.destroyImageView(gc.dev, self.view, null);
            gc.vkd.destroyImage(gc.dev, self.image, null);
            gc.vkd.destroySampler(gc.dev, self.sampler, null);
            gc.vkd.freeMemory(gc.dev, self.memory, null);
        }
    };
}
