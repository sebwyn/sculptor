const std = @import("std");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const zlm = @import("zlm").SpecializeOn(f32);

const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const triangle_vert = @embedFile("triangle_vert");
const triangle_frag = @embedFile("aabb_raycaster_frag");

const Window = @import("window.zig").Window;
const CameraController = @import("camera_controller.zig").CameraController;
const Camera = @import("camera_controller.zig").Camera;
const Texture = @import("texture.zig").Texture;
const Ndarray = @import("ndarray.zig").Ndarray;
const NdarrayView = @import("ndarray.zig").NdarrayView;

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

const vertices = [_]Vertex{
    .{ .pos = .{ -1.0, -1.0 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 1.0, 1.0 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ -1.0, 1.0 }, .color = .{ 0, 0, 1 } },
    .{ .pos = .{ -1.0, -1.0 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 1.0, -1.0 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ 1.0, 1.0 }, .color = .{ 0, 0, 1 } },
};

/// Default GLFW error handling callback
fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}\n", .{ error_code, description });
}

const PI = 3.14159265;

const VoxelObjectStore = struct {
    const MAX_OBJECTS = 100;

    gc: *const GraphicsContext,
    command_pool: vk.CommandPool,
    allocator: std.mem.Allocator,
    descriptor_set_layout: vk.DescriptorSetLayout,

    object_count: usize = 0,
    voxel_objects: [MAX_OBJECTS]VoxelObject,
    descriptor_sets: [MAX_OBJECTS]vk.DescriptorSet,
    descriptor_pool: vk.DescriptorPool,

    const Ref = struct {
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

        const transform_descriptor_size: vk.DescriptorPoolSize = .{ .type = .uniform_buffer, .descriptor_count = @intCast(1) };
        const image_descriptor_size: vk.DescriptorPoolSize = .{ .type = .combined_image_sampler, .descriptor_count = @intCast(2) };
        const descriptor_pool_info: vk.DescriptorPoolCreateInfo = .{ .max_sets = @intCast(MAX_OBJECTS), .pool_size_count = 2, .p_pool_sizes = &.{ transform_descriptor_size, image_descriptor_size } };
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
        const palette = try Texture(1).init(self.gc, .r8g8b8a8_srgb, .{255});
        errdefer palette.deinit(self.gc);
        const voxels = try Texture(3).init(self.gc, .r8_unorm, size);
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

const VoxelObject = struct {
    transform_buffer: GraphicsContext.Buffer(zlm.Mat4),
    palette: Texture(1),
    voxels: Texture(3),

    fn deinit(self: *const VoxelObject, gc: *const GraphicsContext) void {
        self.transform_buffer.deinit(gc);
        self.palette.deinit(gc);
        self.voxels.deinit(gc);
    }
};

pub fn main() !void {
    glfw.setErrorCallback(errorCallback);
    if (!glfw.init(.{})) {
        std.log.err("failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    }
    defer glfw.terminate();

    var extent = vk.Extent2D{ .width = 800, .height = 600 };

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer if (gpa.deinit() == .leak) {
        @panic("Your leaking dog");
    };
    const general_allocator = gpa.allocator();
    var window = try Window.init(general_allocator, 800, 600, "sculptor");
    defer window.deinit();

    var camera_controller = CameraController.init(&window);
    camera_controller.registerWithEvents(&window);

    const allocator = std.heap.page_allocator;
    const gc = try GraphicsContext.init(allocator, app_name, window.glfw_window);
    defer gc.deinit();

    const pool = try gc.vkd.createCommandPool(gc.dev, &.{
        .flags = .{},
        .queue_family_index = gc.graphics_queue.family,
    }, null);
    defer gc.vkd.destroyCommandPool(gc.dev, pool, null);

    var voxel_object_store = try VoxelObjectStore.init(&gc, pool, allocator);
    defer voxel_object_store.deinit();

    std.debug.print("Using device: {?s}\n", .{gc.props.device_name});

    var swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    const camera_descriptor_layout = try gc.vkd.createDescriptorSetLayout(gc.dev, &.{
        .binding_count = 1,
        .flags = .{},
        .p_bindings = &.{
            .{
                .stage_flags = vk.ShaderStageFlags{ .fragment_bit = true },
                .binding = 0,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
            },
        },
    }, null);
    defer gc.vkd.destroyDescriptorSetLayout(gc.dev, camera_descriptor_layout, null);

    const pipeline_layout = try gc.vkd.createPipelineLayout(gc.dev, &.{
        .flags = .{},
        .set_layout_count = 2,
        .p_set_layouts = &.{ camera_descriptor_layout, voxel_object_store.descriptor_set_layout },
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
        _ = try buffer.map(&gc);
        uniform_buffers[initialized_uniform_buffers] = buffer;
        initialized_uniform_buffers += 1;
    }

    const camera_descriptor_size: vk.DescriptorPoolSize = .{ .type = .uniform_buffer, .descriptor_count = @intCast(1 * framebuffers.len) };
    const descriptor_pool_info: vk.DescriptorPoolCreateInfo = .{ .max_sets = @intCast(framebuffers.len), .pool_size_count = 1, .p_pool_sizes = &.{camera_descriptor_size} };
    const camera_descriptor_pool = try gc.vkd.createDescriptorPool(gc.dev, &descriptor_pool_info, null);
    defer gc.vkd.destroyDescriptorPool(gc.dev, camera_descriptor_pool, null);

    const camera_descriptor_layouts = try allocator.alloc(vk.DescriptorSetLayout, framebuffers.len);
    defer allocator.free(camera_descriptor_layouts);
    @memset(camera_descriptor_layouts, camera_descriptor_layout);

    const camera_descriptor_sets = try allocator.alloc(vk.DescriptorSet, framebuffers.len);
    defer allocator.free(camera_descriptor_sets);

    try gc.vkd.allocateDescriptorSets(gc.dev, &vk.DescriptorSetAllocateInfo{
        .descriptor_pool = camera_descriptor_pool,
        .p_set_layouts = camera_descriptor_layouts.ptr,
        .descriptor_set_count = @intCast(framebuffers.len),
    }, camera_descriptor_sets.ptr);

    for (0..framebuffers.len) |i| {
        const camera_write_descriptor = vk.WriteDescriptorSet{ .descriptor_type = .uniform_buffer, .dst_set = camera_descriptor_sets[i], .dst_binding = 0, .dst_array_element = 0, .descriptor_count = 1, .p_buffer_info = &.{uniform_buffers[i].getBufferInfo()}, .p_image_info = &[_]vk.DescriptorImageInfo{}, .p_texel_buffer_view = &[_]vk.BufferView{} };
        gc.vkd.updateDescriptorSets(gc.dev, 1, &.{camera_write_descriptor}, 0, null);
    }

    var rand = std.rand.DefaultPrng.init(0);

    const voxel_object_ref = try voxel_object_store.createSphere(.{ 32, 32, 32 }, 8.0);
    const voxel_object = voxel_object_store.getObjectMut(voxel_object_ref);

    const voxel_object2_ref = try voxel_object_store.createSphere(.{ 32, 32, 32 }, 8.0);
    const voxel_object2 = voxel_object_store.getObjectMut(voxel_object2_ref);
    try voxel_object2.transform_buffer.write(&gc, &.{zlm.Mat4.createTranslation(zlm.Vec3.new(16, 0, 0))});

    {
        const palette_staging_buffer = try voxel_object.palette.createStagingBuffer(&gc, allocator);
        defer palette_staging_buffer.deinit(&gc);
        for (0..255) |i| {
            const color = &.{ rand.random().int(u8), rand.random().int(u8), rand.random().int(u8), 255 };
            palette_staging_buffer.slice(&.{i}).write(color);
        }

        try voxel_object.palette.writeStagingBuffer(&gc, pool, palette_staging_buffer);
        try voxel_object2.palette.writeStagingBuffer(&gc, pool, palette_staging_buffer);
    }

    const vertex_buffer = try gc.allocateBuffer(Vertex, vertices.len, .{ .transfer_dst_bit = true, .vertex_buffer_bit = true }, .{ .device_local_bit = true });
    defer vertex_buffer.deinit(&gc);

    try uploadVertices(&gc, pool, vertex_buffer);

    var cmdbufs = try createCommandBuffers(
        &gc,
        pool,
        allocator,
        vertex_buffer.vk_handle,
        swapchain.extent,
        render_pass,
        pipeline,
        camera_descriptor_sets,
        voxel_object_store.getDescriptorSets(),
        pipeline_layout,
        framebuffers,
    );
    defer destroyCommandBuffers(&gc, pool, allocator, cmdbufs);

    while (!window.glfw_window.shouldClose()) {
        const cmdbuf = cmdbufs[swapchain.image_index];

        try uniform_buffers[swapchain.image_index].write(&gc, &.{camera_controller.camera});

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            window.emitResizeEvent();

            const size = window.glfw_window.getSize();
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
                camera_descriptor_sets,
                voxel_object_store.getDescriptorSets(),
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
    allocator: std.mem.Allocator,
    buffer: vk.Buffer,
    extent: vk.Extent2D,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    camera_descriptor_sets: []vk.DescriptorSet,
    voxel_object_descriptor_sets: []const vk.DescriptorSet,
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
        gc.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 0, 1, camera_descriptor_sets[i..].ptr, 0, null);

        for (0..voxel_object_descriptor_sets.len) |ds| {
            gc.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 1, 1, voxel_object_descriptor_sets[ds..].ptr, 0, null);
            gc.vkd.cmdDraw(cmdbuf, vertices.len, 1, 0, 0);
        }

        gc.vkd.cmdEndRenderPass(cmdbuf);
        try gc.vkd.endCommandBuffer(cmdbuf);
    }

    return cmdbufs;
}

fn destroyCommandBuffers(gc: *const GraphicsContext, pool: vk.CommandPool, allocator: std.mem.Allocator, cmdbufs: []vk.CommandBuffer) void {
    gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

fn createFramebuffers(gc: *const GraphicsContext, allocator: std.mem.Allocator, render_pass: vk.RenderPass, swapchain: Swapchain) ![]vk.Framebuffer {
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

fn destroyFramebuffers(gc: *const GraphicsContext, allocator: std.mem.Allocator, framebuffers: []const vk.Framebuffer) void {
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
