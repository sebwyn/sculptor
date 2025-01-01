const std = @import("std");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const zlm = @import("zlm").SpecializeOn(f32);
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const triangle_vert = @embedFile("triangle_vert");
const triangle_frag = @embedFile("aabb_raycaster_frag");
const Allocator = std.mem.Allocator;

const app_name = "mach-glfw + vulkan-zig = triangle";

const Vertex = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
    };

    pos: [2]f32,
    color: [3]f32,
};

const Texture3dError = error{ UnsupportedDevice, UnsupportedSize, BadDataFormat };

const Texture3d = struct {
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
            .unnormalized_coordinates = @intFromBool(true),
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
        defer staging_buffer.deinit();

        const command_buffer = try gc.beginSingleTimeCommands(pool);

        const subresource_range: vk.ImageSubresourceRange = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
            .level_count = 1,
        };

        const write_image_layout_transition: vk.HostImageLayoutTransitionInfoEXT = .{
            .image = self.image,
            .subresource_range = subresource_range,
            .old_layout = self.image_layout,
            .new_layout = .transfer_dst_optimal,
        };
        try gc.vkd.transitionImageLayoutEXT(gc.dev, 1, @ptrCast(&write_image_layout_transition));

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

        self.image_layout = .shader_read_only_optimal;
        const shader_read_image_transition: vk.HostImageLayoutTransitionInfoEXT = .{
            .image = self.image,
            .subresource_range = subresource_range,
            .old_layout = .transfer_dst_optimal,
            .new_layout = self.image_layout,
        };
        try gc.vkd.transitionImageLayoutEXT(gc.dev, 1, @ptrCast(&shader_read_image_transition));

        try gc.endSingleTimeCommands(pool, command_buffer, gc.graphics_queue.handle);
    }

    pub fn deinit(self: Texture3d, gc: *const GraphicsContext) void {
        gc.vkd.destroyImageView(gc.dev, self.view, null);
        gc.vkd.destroySampler(gc.dev, self.sampler, null);
        gc.vkd.freeMemory(gc.dev, self.memory, null);
        gc.vkd.destroyImage(gc.dev, self.image, null);
    }
};

const vertices = [_]Vertex{
    .{ .pos = .{ -1.0, -1.0 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 1.0, 1.0 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ -1.0, 1.0 }, .color = .{ 0, 0, 1 } },
    .{ .pos = .{ -1.0, -1.0 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 1.0, -1.0 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ 1.0, 1.0 }, .color = .{ 0, 0, 1 } },
};

const Camera = struct {
    view_matrix: zlm.Mat4,
    proj_matrix: zlm.Mat4,
    screen_size: [4]f32,
};

/// Default GLFW error handling callback
fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}\n", .{ error_code, description });
}

const PI = 3.14159265;

pub fn main() !void {
    glfw.setErrorCallback(errorCallback);
    if (!glfw.init(.{})) {
        std.log.err("failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    }
    defer glfw.terminate();

    var extent = vk.Extent2D{ .width = 800, .height = 600 };

    const window = glfw.Window.create(extent.width, extent.height, app_name, null, null, .{
        .client_api = .no_api,
    }) orelse {
        std.log.err("failed to create GLFW window: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    };
    defer window.destroy();

    const allocator = std.heap.page_allocator;

    const gc = try GraphicsContext.init(allocator, app_name, window);
    defer gc.deinit();

    std.debug.print("Using device: {?s}\n", .{gc.props.device_name});

    var swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    const descriptor_set_layout = try gc.vkd.createDescriptorSetLayout(gc.dev, &.{
        .binding_count = 1,
        .flags = .{},
        .p_bindings = &.{.{
            .stage_flags = vk.ShaderStageFlags{ .fragment_bit = true },
            .binding = 1,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
        }},
    }, null);
    defer gc.vkd.destroyDescriptorSetLayout(gc.dev, descriptor_set_layout, null);

    const pipeline_layout = try gc.vkd.createPipelineLayout(gc.dev, &.{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = &.{descriptor_set_layout},
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    defer gc.vkd.destroyPipelineLayout(gc.dev, pipeline_layout, null);

    const render_pass = try createRenderPass(&gc, swapchain);
    defer gc.vkd.destroyRenderPass(gc.dev, render_pass, null);

    const pipeline = try createPipeline(&gc, pipeline_layout, render_pass);
    defer gc.vkd.destroyPipeline(gc.dev, pipeline, null);

    var framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain);
    defer destroyFramebuffers(&gc, allocator, framebuffers);

    var initialized_uniform_buffers: usize = 0;
    var uniform_buffers = try allocator.alloc(GraphicsContext.Buffer(Camera), framebuffers.len);
    defer allocator.free(uniform_buffers);
    defer for (0..initialized_uniform_buffers) |b| {
        uniform_buffers[b].deinit(&gc);
    };

    while (initialized_uniform_buffers < framebuffers.len) {
        var buffer = try gc.allocateBuffer(Camera, 1, .{ .uniform_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
        errdefer buffer.deinit(&gc);
        try buffer.map(&gc);
        uniform_buffers[initialized_uniform_buffers] = buffer;
        initialized_uniform_buffers += 1;
    }

    const descriptor_pool = try gc.vkd.createDescriptorPool(gc.dev, &vk.DescriptorPoolCreateInfo{ .pool_size_count = 1, .max_sets = @intCast(framebuffers.len), .p_pool_sizes = &.{
        vk.DescriptorPoolSize{
            .type = .uniform_buffer,
            .descriptor_count = @intCast(framebuffers.len),
        },
    } }, null);
    defer gc.vkd.destroyDescriptorPool(gc.dev, descriptor_pool, null);

    var descriptor_set_layouts: []vk.DescriptorSetLayout = try allocator.alloc(vk.DescriptorSetLayout, framebuffers.len);
    defer allocator.free(descriptor_set_layouts);
    for (0..framebuffers.len) |i| {
        descriptor_set_layouts[i] = descriptor_set_layout;
    }

    const descriptor_sets: []vk.DescriptorSet = try allocator.alloc(vk.DescriptorSet, framebuffers.len);
    defer allocator.free(descriptor_sets);

    try gc.vkd.allocateDescriptorSets(gc.dev, &vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .p_set_layouts = descriptor_set_layouts.ptr,
        .descriptor_set_count = @intCast(framebuffers.len),
    }, descriptor_sets.ptr);

    for (0..framebuffers.len) |i| {
        const write_descriptor = vk.WriteDescriptorSet{ .descriptor_type = .uniform_buffer, .dst_set = descriptor_sets[i], .dst_binding = 1, .dst_array_element = 0, .descriptor_count = 1, .p_buffer_info = &.{uniform_buffers[i].getBufferInfo()}, .p_image_info = &[_]vk.DescriptorImageInfo{}, .p_texel_buffer_view = &[_]vk.BufferView{} };
        gc.vkd.updateDescriptorSets(gc.dev, 1, &.{write_descriptor}, 0, null);
    }

    const pool = try gc.vkd.createCommandPool(gc.dev, &.{
        .flags = .{},
        .queue_family_index = gc.graphics_queue.family,
    }, null);
    defer gc.vkd.destroyCommandPool(gc.dev, pool, null);

    const vertex_buffer = try gc.allocateBuffer(Vertex, vertices.len, .{ .transfer_dst_bit = true, .vertex_buffer_bit = true }, .{ .device_local_bit = true });
    defer vertex_buffer.deinit(&gc);

    try uploadVertices(&gc, pool, vertex_buffer);

    // var voxels = try Texture3d.init(&gc, 16, 16, 16);
    // defer voxels.deinit(&gc);
    //
    // var voxel_data: [16 * 16 * 16]u8 = undefined;
    // for (0..16) |z| {
    //     for (0..16) |y| {
    //         for (0..16) |x| {
    //             const xf = @as(f32, @floatFromInt(x)) + 0.5;
    //             const yf = @as(f32, @floatFromInt(y)) + 0.5;
    //             const zf = @as(f32, @floatFromInt(z)) + 0.5;
    //             const dist_from_origin: f32 = std.math.sqrt(xf * xf + yf * yf + zf * zf);
    //             voxel_data[16 * 16 * z + 16 * y + x] = @intFromBool(dist_from_origin < 3);
    //         }
    //     }
    // }
    //
    // try voxels.write(&gc, pool, &voxel_data);

    var cmdbufs = try createCommandBuffers(
        &gc,
        pool,
        allocator,
        vertex_buffer.vk_handle,
        swapchain.extent,
        render_pass,
        pipeline,
        descriptor_sets,
        pipeline_layout,
        framebuffers,
    );
    defer destroyCommandBuffers(&gc, pool, allocator, cmdbufs);

    const start_time = std.time.milliTimestamp();
    while (!window.shouldClose()) {
        const cmdbuf = cmdbufs[swapchain.image_index];

        const current_time = std.time.milliTimestamp();
        const angle: f32 = (2 * PI) * @as(f32, @floatFromInt(current_time - start_time)) / (1000 * 60);

        const x = 10 * std.math.cos(angle);
        const z = 10 * std.math.sin(angle);
        const y = 3 * std.math.sin(4 * angle);

        const target = zlm.Vec3.new(0.0, 0.0, 0.0);
        const camera_position = zlm.Vec3.new(x, y, z);

        const window_size = window.getFramebufferSize();
        const window_width = @as(f32, @floatFromInt(window_size.width));
        const window_height = @as(f32, @floatFromInt(window_size.height));
        const aspect_ratio: f32 = window_width / window_height;

        const camera = Camera{ .view_matrix = zlm.Mat4.createLook(camera_position, (target.sub(camera_position)).normalize(), zlm.Vec3.new(0.0, 1.0, 0.0)), .proj_matrix = zlm.Mat4.createPerspective((PI / 180.0) * 120.0, aspect_ratio, 1, 1000), .screen_size = .{ window_width, window_height, 0.0, 0.0 } };

        try uniform_buffers[swapchain.image_index].write(&gc, &.{ camera });

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            const size = window.getSize();
            extent.width = @intCast(size.width);
            extent.height = @intCast(size.height);
            try swapchain.recreate(extent);

            destroyFramebuffers(&gc, allocator, framebuffers);
            framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain);

            destroyCommandBuffers(&gc, pool, allocator, cmdbufs);
            cmdbufs = try createCommandBuffers(
                &gc,
                pool,
                allocator,
                vertex_buffer.vk_handle,
                swapchain.extent,
                render_pass,
                pipeline,
                descriptor_sets,
                pipeline_layout,
                framebuffers,
            );
        }

        glfw.pollEvents();
    }

    try swapchain.waitForAllFences();
}

fn uploadVertices(gc: *const GraphicsContext, pool: vk.CommandPool, buffer: GraphicsContext.Buffer(Vertex)) !void {
    const vertices_size_bytes = @sizeOf(@TypeOf(vertices));
    const staging_buffer = try gc.writeStagingBuffer(Vertex, &vertices);
    defer staging_buffer.deinit(gc);

    try gc.copyBuffer(pool, staging_buffer.vk_handle, buffer.vk_handle, vertices_size_bytes);
}

fn createCommandBuffers(
    gc: *const GraphicsContext,
    pool: vk.CommandPool,
    allocator: Allocator,
    buffer: vk.Buffer,
    extent: vk.Extent2D,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    descriptor_sets: []vk.DescriptorSet,
    pipeline_layout: vk.PipelineLayout,
    framebuffers: []vk.Framebuffer,
) ![]vk.CommandBuffer {
    const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
    errdefer allocator.free(cmdbufs);

    try gc.vkd.allocateCommandBuffers(gc.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = @truncate(cmdbufs.len),
    }, cmdbufs.ptr);
    errdefer gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);

    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @as(f32, @floatFromInt(extent.width)),
        .height = @as(f32, @floatFromInt(extent.height)),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    for (cmdbufs, 0..) |cmdbuf, i| {
        try gc.vkd.beginCommandBuffer(cmdbuf, &.{
            .flags = .{},
            .p_inheritance_info = null,
        });

        gc.vkd.cmdSetViewport(cmdbuf, 0, 1, @as([*]const vk.Viewport, @ptrCast(&viewport)));
        gc.vkd.cmdSetScissor(cmdbuf, 0, 1, @as([*]const vk.Rect2D, @ptrCast(&scissor)));

        // This needs to be a separate definition - see https://github.com/ziglang/zig/issues/7627.
        const render_area = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = extent,
        };

        gc.vkd.cmdBeginRenderPass(cmdbuf, &.{
            .render_pass = render_pass,
            .framebuffer = framebuffers[i],
            .render_area = render_area,
            .clear_value_count = 1,
            .p_clear_values = @as([*]const vk.ClearValue, @ptrCast(&clear)),
        }, .@"inline");

        gc.vkd.cmdBindPipeline(cmdbuf, .graphics, pipeline);
        const offset = [_]vk.DeviceSize{0};
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @as([*]const vk.Buffer, @ptrCast(&buffer)), &offset);
        gc.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 0, 1, descriptor_sets[i..].ptr, 0, null);
        gc.vkd.cmdDraw(cmdbuf, vertices.len, 1, 0, 0);

        gc.vkd.cmdEndRenderPass(cmdbuf);
        try gc.vkd.endCommandBuffer(cmdbuf);
    }

    return cmdbufs;
}

fn destroyCommandBuffers(gc: *const GraphicsContext, pool: vk.CommandPool, allocator: Allocator, cmdbufs: []vk.CommandBuffer) void {
    gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

fn createFramebuffers(gc: *const GraphicsContext, allocator: Allocator, render_pass: vk.RenderPass, swapchain: Swapchain) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (framebuffers) |*fb| {
        fb.* = try gc.vkd.createFramebuffer(gc.dev, &vk.FramebufferCreateInfo{
            .flags = .{},
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swapchain.swap_images[i].view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const GraphicsContext, allocator: Allocator, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);
    allocator.free(framebuffers);
}

fn createRenderPass(gc: *const GraphicsContext, swapchain: Swapchain) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .flags = .{},
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .flags = .{},
        .pipeline_bind_point = .graphics,
        .input_attachment_count = 0,
        .p_input_attachments = undefined,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_resolve_attachments = null,
        .p_depth_stencil_attachment = null,
        .preserve_attachment_count = 0,
        .p_preserve_attachments = undefined,
    };

    return try gc.vkd.createRenderPass(gc.dev, &vk.RenderPassCreateInfo{
        .flags = .{},
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 0,
        .p_dependencies = undefined,
    }, null);
}

fn createPipeline(
    gc: *const GraphicsContext,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try gc.vkd.createShaderModule(gc.dev, &vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = triangle_vert.len,
        .p_code = @ptrCast(@alignCast(triangle_vert)),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, vert, null);

    const frag = try gc.vkd.createShaderModule(gc.dev, &vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = triangle_frag.len,
        .p_code = @ptrCast(@alignCast(triangle_frag)),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, frag, null);

    const graphics_pipeline_create_info = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = 2,
        .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
            .{
                .flags = .{},
                .stage = .{ .vertex_bit = true },
                .module = vert,
                .p_name = "main",
                .p_specialization_info = null,
            },
            .{
                .flags = .{},
                .stage = .{ .fragment_bit = true },
                .module = frag,
                .p_name = "main",
                .p_specialization_info = null,
            },
        },
        .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
            .flags = .{},
            .vertex_binding_description_count = 1,
            .p_vertex_binding_descriptions = @as([*]const vk.VertexInputBindingDescription, @ptrCast(&Vertex.binding_description)),
            .vertex_attribute_description_count = Vertex.attribute_description.len,
            .p_vertex_attribute_descriptions = &Vertex.attribute_description,
        },
        .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
            .flags = .{},
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        },
        .p_tessellation_state = null,
        .p_viewport_state = &vk.PipelineViewportStateCreateInfo{
            .flags = .{},
            .viewport_count = 1,
            .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
            .scissor_count = 1,
            .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
        },
        .p_rasterization_state = &vk.PipelineRasterizationStateCreateInfo{
            .flags = .{},
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        },
        .p_multisample_state = &vk.PipelineMultisampleStateCreateInfo{
            .flags = .{},
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = vk.FALSE,
            .min_sample_shading = 1,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        },
        .p_depth_stencil_state = null,
        .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
            .flags = .{},
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @as([*]const vk.PipelineColorBlendAttachmentState, @ptrCast(&vk.PipelineColorBlendAttachmentState{
                .blend_enable = vk.FALSE,
                .src_color_blend_factor = .one,
                .dst_color_blend_factor = .zero,
                .color_blend_op = .add,
                .src_alpha_blend_factor = .one,
                .dst_alpha_blend_factor = .zero,
                .alpha_blend_op = .add,
                .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            })),
            .blend_constants = [_]f32{ 0, 0, 0, 0 },
        },
        .p_dynamic_state = &vk.PipelineDynamicStateCreateInfo{
            .flags = .{},
            .dynamic_state_count = 2,
            .p_dynamic_states = &[_]vk.DynamicState{ .viewport, .scissor },
        },
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try gc.vkd.createGraphicsPipelines(
        gc.dev,
        .null_handle,
        1,
        @as([*]const vk.GraphicsPipelineCreateInfo, @ptrCast(&graphics_pipeline_create_info)),
        null,
        @as([*]vk.Pipeline, @ptrCast(&pipeline)),
    );
    return pipeline;
}
