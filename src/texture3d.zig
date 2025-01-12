const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const vk = @import("vulkan");

const Texture3dError = error{ UnsupportedDevice, UnsupportedSize, BadDataFormat };

pub const Texture3d = struct {
    sampler: vk.Sampler,
    image: vk.Image,
    image_layout: vk.ImageLayout,
    descriptor: vk.DescriptorImageInfo,
    view: vk.ImageView,
    memory: vk.DeviceMemory,
    format: vk.Format,

    width: u32,
    height: u32,
    depth: u32,

    pub fn init(gc: *const GraphicsContext, width: u32, height: u32, depth: u32) !Texture3d {
        const format: vk.Format = .r8_unorm;
        const initial_layout: vk.ImageLayout = .undefined;

        const physical_properties = gc.vki.getPhysicalDeviceFormatProperties(gc.pdev, format);
        if (!physical_properties.optimal_tiling_features.contains(.{ .transfer_dst_bit = true })) {
            return error.UnsupportedDevice;
        }

        const max_texture_size = gc.vki.getPhysicalDeviceProperties(gc.pdev).limits.max_image_dimension_3d;
        if (width > max_texture_size or height > max_texture_size or depth > max_texture_size) {
            return error.UnsupportedSize;
        }

        const image_create_info: vk.ImageCreateInfo = .{
            .image_type = .@"3d",
            .format = format,
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .sharing_mode = .exclusive,
            .extent = .{ .width = width, .height = height, .depth = depth },
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

        const image_view_create_info: vk.ImageViewCreateInfo = .{ .image = image, .view_type = .@"3d", .format = format, .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
            .level_count = 1,
        }, .components = .{ .r = .r, .g = .g, .b = .b, .a = .a } };
        const image_view = try gc.vkd.createImageView(gc.dev, &image_view_create_info, null);
        errdefer gc.vkd.destroyImageView(gc.dev, image_view, null);

        const descriptor = vk.DescriptorImageInfo{
            .image_layout = .shader_read_only_optimal,
            .sampler = sampler,
            .image_view = image_view,
        };

        return Texture3d{
            .sampler = sampler,
            .image = image,
            .image_layout = initial_layout,
            .descriptor = descriptor,
            .view = image_view,
            .memory = memory,
            .format = format,
            .width = width,
            .height = height,
            .depth = depth,
        };
    }

    pub fn write(self: *Texture3d, gc: *const GraphicsContext, pool: vk.CommandPool, data: []u8) !void {
        const expected_size = self.width * self.height * self.depth;
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
            .image_extent = .{
                .width = self.width,
                .height = self.height,
                .depth = self.depth,
            },
            .buffer_offset = 0,
            .buffer_row_length = self.width,
            .buffer_image_height = self.height,
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        };
        gc.vkd.cmdCopyBufferToImage(command_buffer, staging_buffer.vk_handle, self.image, .transfer_dst_optimal, 1, @ptrCast(&buffer_image_copy));

        try gc.transitionImageLayout(command_buffer, self.image, subresource_range, .transfer_dst_optimal, .shader_read_only_optimal);

        try gc.endSingleTimeCommands(pool, command_buffer, gc.graphics_queue.handle);
    }

    pub fn deinit(self: Texture3d, gc: *const GraphicsContext) void {
        gc.vkd.destroyImageView(gc.dev, self.view, null);
        gc.vkd.destroyImage(gc.dev, self.image, null);
        gc.vkd.destroySampler(gc.dev, self.sampler, null);
        gc.vkd.freeMemory(gc.dev, self.memory, null);
    }
};
