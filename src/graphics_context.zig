const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const Allocator = std.mem.Allocator;

const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

fn requiredDeviceExtensions(comptime os: std.Target.Os) []const [*:0]const u8 {
    return switch (os.tag) {
        .macos => &[_][*:0]const u8{
            vk.extensions.khr_swapchain.name,
            vk.extensions.khr_portability_subset.name,
            vk.extensions.ext_image_robustness.name,
        },
        else => &[_][*:0]const u8{
            vk.extensions.khr_swapchain.name,
        },
    };
}

const optional_device_extensions = [_][*:0]const u8{};

const optional_instance_extensions = [_][*:0]const u8{
    vk.extensions.khr_get_physical_device_properties_2.name,
    vk.extensions.khr_portability_enumeration.name,
    vk.extensions.ext_debug_utils.name,
};

const apis: []const vk.ApiInfo = &.{.{
    .base_commands = .{
        .createInstance = true,
        .enumerateInstanceExtensionProperties = true,
        .getInstanceProcAddr = true,
    },
    .instance_commands = .{
        .destroyInstance = true,
        .createDevice = true,
        .destroySurfaceKHR = true,
        .enumeratePhysicalDevices = true,
        .getPhysicalDeviceProperties = true,
        .enumerateDeviceExtensionProperties = true,
        .getPhysicalDeviceSurfaceFormatsKHR = true,
        .getPhysicalDeviceSurfacePresentModesKHR = true,
        .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
        .getPhysicalDeviceQueueFamilyProperties = true,
        .getPhysicalDeviceSurfaceSupportKHR = true,
        .getPhysicalDeviceMemoryProperties = true,
        .getDeviceProcAddr = true,
        .getPhysicalDeviceFormatProperties = true,
    },
    .device_commands = .{
        .destroyDevice = true,
        .getDeviceQueue = true,
        .createSemaphore = true,
        .createFence = true,
        .createImageView = true,
        .destroyImageView = true,
        .destroySemaphore = true,
        .destroyFence = true,
        .getSwapchainImagesKHR = true,
        .createSwapchainKHR = true,
        .destroySwapchainKHR = true,
        .acquireNextImageKHR = true,
        .deviceWaitIdle = true,
        .waitForFences = true,
        .resetFences = true,
        .queueSubmit = true,
        .queuePresentKHR = true,
        .createCommandPool = true,
        .destroyCommandPool = true,
        .allocateCommandBuffers = true,
        .freeCommandBuffers = true,
        .queueWaitIdle = true,
        .createShaderModule = true,
        .destroyShaderModule = true,
        .createPipelineLayout = true,
        .destroyPipelineLayout = true,
        .createRenderPass = true,
        .destroyRenderPass = true,
        .createDescriptorSetLayout = true,
        .destroyDescriptorSetLayout = true,
        .createGraphicsPipelines = true,
        .createDescriptorPool = true,
        .destroyDescriptorPool = true,
        .allocateDescriptorSets = true,
        .updateDescriptorSets = true,
        .destroyPipeline = true,
        .createFramebuffer = true,
        .destroyFramebuffer = true,
        .beginCommandBuffer = true,
        .endCommandBuffer = true,
        .allocateMemory = true,
        .freeMemory = true,
        .createBuffer = true,
        .destroyBuffer = true,
        .getBufferMemoryRequirements = true,
        .mapMemory = true,
        .unmapMemory = true,
        .bindBufferMemory = true,
        .cmdBeginRenderPass = true,
        .cmdEndRenderPass = true,
        .cmdBindPipeline = true,
        .cmdDraw = true,
        .cmdSetViewport = true,
        .cmdSetScissor = true,
        .cmdBindVertexBuffers = true,
        .cmdCopyBuffer = true,
        .cmdBindDescriptorSets = true,
        .createImage = true,
        .destroyImage = true,
        .bindImageMemory = true,
        .getImageMemoryRequirements = true,
        .createSampler = true,
        .destroySampler = true,
        .cmdCopyBufferToImage = true,
        .cmdPipelineBarrier = true,
    },
}};

const BaseDispatch = vk.BaseWrapper(apis);
const InstanceDispatch = vk.InstanceWrapper(apis);
const DeviceDispatch = vk.DeviceWrapper(apis);

pub const GraphicsContext = struct {
    vkb: BaseDispatch,
    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    instance: vk.Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,

    dev: vk.Device,
    graphics_queue: Queue,
    present_queue: Queue,

    pub fn init(allocator: Allocator, app_name: [*:0]const u8, window: glfw.Window) !GraphicsContext {
        var self: GraphicsContext = undefined;
        self.vkb = try BaseDispatch.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(&glfw.getInstanceProcAddress)));

        const glfw_exts = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();
            std.log.err("failed to get required vulkan instance extensions: error={s}", .{err.description});
            break :blk error.code;
        };

        var instance_extensions = try std.ArrayList([*:0]const u8).initCapacity(allocator, glfw_exts.len + 1);
        defer instance_extensions.deinit();
        try instance_extensions.appendSlice(glfw_exts);

        var count: u32 = undefined;
        _ = try self.vkb.enumerateInstanceExtensionProperties(null, &count, null);

        const propsv = try allocator.alloc(vk.ExtensionProperties, count);
        defer allocator.free(propsv);

        _ = try self.vkb.enumerateInstanceExtensionProperties(null, &count, propsv.ptr);

        for (optional_instance_extensions) |extension_name| {
            for (propsv) |prop| {
                const len = std.mem.indexOfScalar(u8, &prop.extension_name, 0).?;
                const prop_ext_name = prop.extension_name[0..len];
                if (std.mem.eql(u8, prop_ext_name, std.mem.span(extension_name))) {
                    try instance_extensions.append(@ptrCast(extension_name));
                    break;
                }
            }
        }

        const app_info = vk.ApplicationInfo{
            .p_application_name = app_name,
            .application_version = vk.makeApiVersion(0, 0, 0, 0),
            .p_engine_name = app_name,
            .engine_version = vk.makeApiVersion(0, 0, 0, 0),
            .api_version = vk.makeApiVersion(0, 1, 1, 0),
        };

        self.instance = try self.vkb.createInstance(&vk.InstanceCreateInfo{ .flags = if (builtin.os.tag == .macos) .{
            .enumerate_portability_bit_khr = true,
        } else .{}, .p_application_info = &app_info, .enabled_layer_count = 1, .pp_enabled_layer_names = &validation_layers, .enabled_extension_count = @intCast(instance_extensions.items.len), .pp_enabled_extension_names = @ptrCast(instance_extensions.items), .p_next = &vk.DebugUtilsMessengerCreateInfoEXT{
            .flags = .{},
            .message_severity = .{
                .verbose_bit_ext = true,
                .warning_bit_ext = true,
                .error_bit_ext = true,
            },
            .message_type = .{
                .general_bit_ext = true,
                .validation_bit_ext = true,
                .performance_bit_ext = true,
            },
            .pfn_user_callback = debugCallback,
        } }, null);

        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
        errdefer self.vki.destroyInstance(self.instance, null);

        self.surface = try createSurface(self.instance, window);
        errdefer self.vki.destroySurfaceKHR(self.instance, self.surface, null);

        const candidate = try pickPhysicalDevice(self.vki, self.instance, allocator, self.surface);
        self.pdev = candidate.pdev;
        self.props = candidate.props;
        self.dev = try initializeCandidate(allocator, self.vki, candidate);
        self.vkd = try DeviceDispatch.load(self.dev, self.vki.dispatch.vkGetDeviceProcAddr);
        errdefer self.vkd.destroyDevice(self.dev, null);

        self.graphics_queue = Queue.init(self.vkd, self.dev, candidate.queues.graphics_family);
        self.present_queue = Queue.init(self.vkd, self.dev, candidate.queues.present_family);

        self.mem_props = self.vki.getPhysicalDeviceMemoryProperties(self.pdev);

        return self;
    }

    pub fn deinit(self: GraphicsContext) void {
        self.vkd.destroyDevice(self.dev, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);
    }

    pub fn deviceName(self: GraphicsContext) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.props.device_name, 0).?;
        return self.props.device_name[0..len];
    }

    pub fn findMemoryTypeIndex(self: GraphicsContext, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @as(u5, @truncate(i))) != 0 and mem_type.property_flags.contains(flags)) {
                return @as(u32, @truncate(i));
            }
        }

        return error.NoSuitableMemoryType;
    }

    pub fn allocate(self: GraphicsContext, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.vkd.allocateMemory(self.dev, &.{
            .allocation_size = requirements.size,
            .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }

    pub fn beginSingleTimeCommands(self: GraphicsContext, pool: vk.CommandPool) !vk.CommandBuffer {
        const command_buffer_info: vk.CommandBufferAllocateInfo = .{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        };

        var command_buffer: vk.CommandBuffer = undefined;
        try self.vkd.allocateCommandBuffers(self.dev, &command_buffer_info, @ptrCast(&command_buffer));

        const command_buf_begin_info: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.vkd.beginCommandBuffer(command_buffer, &command_buf_begin_info);

        return command_buffer;
    }

    pub fn endSingleTimeCommands(self: GraphicsContext, pool: vk.CommandPool, command_buffer: vk.CommandBuffer, queue: vk.Queue) !void {
        try self.vkd.endCommandBuffer(command_buffer);

        const submit_command_info = vk.SubmitInfo{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = undefined,
            .p_wait_dst_stage_mask = undefined,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&command_buffer),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = undefined,
        };
        try self.vkd.queueSubmit(queue, 1, @ptrCast(&submit_command_info), .null_handle);
        try self.vkd.queueWaitIdle(queue);

        self.vkd.freeCommandBuffers(self.dev, pool, 1, @ptrCast(&command_buffer));
    }

    pub fn transitionImageLayout(
        self: GraphicsContext,
        command_buffer: vk.CommandBuffer,
        image: vk.Image,
        subresource_range: vk.ImageSubresourceRange,
        old_layout: vk.ImageLayout,
        new_layout: vk.ImageLayout
    ) !void {
        var barrier: vk.ImageMemoryBarrier = .{
            .old_layout = old_layout,
            .new_layout = new_layout,
            .src_queue_family_index = self.graphics_queue.family,   
            .dst_queue_family_index = self.graphics_queue.family,   
            .image = image,
            .subresource_range = subresource_range,
            .src_access_mask = .{ .shader_read_bit =  true },
            .dst_access_mask = .{ .shader_read_bit =  true },
        };

        var source_stage: ?vk.PipelineStageFlags = null;
        var destination_stage: ?vk.PipelineStageFlags = null;
        if (old_layout == .undefined and new_layout == .transfer_dst_optimal) {
            barrier.src_access_mask = .{};
            barrier.dst_access_mask = .{ .transfer_write_bit = true };

            source_stage = .{ .top_of_pipe_bit =  true};
            destination_stage = .{ .transfer_bit =  true };
        } else if (old_layout == .transfer_dst_optimal and new_layout == .shader_read_only_optimal) {
            barrier.src_access_mask = .{ .transfer_write_bit = true };
            barrier.dst_access_mask = .{ .shader_read_bit =  true };
            source_stage = .{ .transfer_bit =  true };
            destination_stage = .{ .fragment_shader_bit = true };
        } else {
            return error.Unknown;
        }

        self.vkd.cmdPipelineBarrier(
            command_buffer, 
            source_stage.?, 
            destination_stage.?, 
            .{ .by_region_bit =  true },
            0, null,
            0, null,
            1, @ptrCast(&barrier)
        );
    }

    pub fn Buffer(Data: type) type {
        return struct {
            vk_handle: vk.Buffer,
            memory: vk.DeviceMemory,
            length: usize,
            data_ptr: ?[*]Data = null,

            const Self = @This();
            pub fn getBufferInfo(self: *const Self) vk.DescriptorBufferInfo {
                return vk.DescriptorBufferInfo{ .buffer = self.vk_handle, .offset = 0, .range = @sizeOf(Data) * @as(u64, self.length) };
            }
            
            pub fn map(self: *Self, gc: *const GraphicsContext) ![]Data {
                 if (self.data_ptr == null) {
                     self.data_ptr = @ptrCast(@alignCast(try gc.vkd.mapMemory(gc.dev, self.memory, 0, vk.WHOLE_SIZE, .{})));
                 }
                 return self.data_ptr.?[0..self.length];
            }

            pub fn unmap(self: *Self, gc: *const GraphicsContext) void {
                if (self.data_ptr) |_| {
                    gc.vkd.unmapMemory(gc.dev, self.memory);
                }
                self.data_ptr = null;
            }

            //This function might not always make the most sense to use as it doubles memory usage temporarily
            pub fn write(self: *Self, gc: *const GraphicsContext, data: []const Data) !void {
                if (self.data_ptr) |data_ptr| {
                    @memcpy(data_ptr, data);
                } else {
                    _ = try self.map(gc);
                    @memcpy(self.data_ptr.?, data);
                    self.unmap(gc);
                }
            }

            pub fn deinit(self: *const Self, gc: *const GraphicsContext) void {
                if (self.data_ptr) |_| { gc.vkd.unmapMemory(gc.dev, self.memory); }
                gc.vkd.freeMemory(gc.dev, self.memory, null);
                gc.vkd.destroyBuffer(gc.dev, self.vk_handle, null);
            }
        };
    }

    pub fn allocateBuffer(self: GraphicsContext, comptime T: type, length: usize, usage_flags: vk.BufferUsageFlags, memory_type: vk.MemoryPropertyFlags) !Buffer(T) {
        const size = @sizeOf(T) * length;
        const buffer = try self.vkd.createBuffer(self.dev, &.{
            .flags = .{},
            .size = size,
            .usage = usage_flags,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        }, null);
        errdefer self.vkd.destroyBuffer(self.dev, buffer, null);

        const mem_reqs = self.vkd.getBufferMemoryRequirements(self.dev, buffer);
        const memory = try self.allocate(mem_reqs, memory_type);
        errdefer self.vkd.freeMemory(self.dev, memory, null);

        try self.vkd.bindBufferMemory(self.dev, buffer, memory, 0);
        
        return Buffer(T) {
            .vk_handle = buffer,
            .memory = memory,
            .length = length,
        };
    }

    pub fn copyBuffer(self: GraphicsContext, pool: vk.CommandPool, src: vk.Buffer, dst: vk.Buffer, size: vk.DeviceSize) !void {
        const cmdbuf = try self.beginSingleTimeCommands(pool);

        const region = vk.BufferCopy{ .src_offset = 0, .dst_offset = 0, .size = size };
        self.vkd.cmdCopyBuffer(cmdbuf, src, dst, 1, @ptrCast(&region));

        try self.endSingleTimeCommands(pool, cmdbuf, self.graphics_queue.handle);
    }

    pub fn writeStagingBuffer(self: *const GraphicsContext, T: type, data: []const T) !Buffer(T) {
        const usage_flags: vk.BufferUsageFlags = .{ .transfer_src_bit = true };
        const memory_properties: vk.MemoryPropertyFlags = .{ .host_visible_bit = true, .host_coherent_bit = true };
        var buffer = try self.allocateBuffer(T, data.len, usage_flags, memory_properties);
        const ptr = try buffer.map(self);
        @memcpy(ptr, data);
        return buffer;
    }
};

fn debugCallback(_: vk.DebugUtilsMessageSeverityFlagsEXT, _: vk.DebugUtilsMessageTypeFlagsEXT, p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT, _: ?*anyopaque) callconv(vk.vulkan_call_conv) vk.Bool32 {
    if (p_callback_data != null) {
        std.log.debug("validation layer: {?s}", .{p_callback_data.?.p_message});
    }

    return vk.FALSE;
}

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    fn init(vkd: DeviceDispatch, dev: vk.Device, family: u32) Queue {
        return .{
            .handle = vkd.getDeviceQueue(dev, family, 0),
            .family = family,
        };
    }
};

fn createSurface(instance: vk.Instance, window: glfw.Window) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    if ((glfw.createWindowSurface(instance, window, null, &surface)) != @intFromEnum(vk.Result.success)) {
        return error.SurfaceInitFailed;
    }

    return surface;
}

fn initializeCandidate(allocator: Allocator, vki: InstanceDispatch, candidate: DeviceCandidate) !vk.Device {
    const priority = [_]f32{1};
    const qci = [_]vk.DeviceQueueCreateInfo{
        .{
            .flags = .{},
            .queue_family_index = candidate.queues.graphics_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
        .{
            .flags = .{},
            .queue_family_index = candidate.queues.present_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
    };

    const queue_count: u32 = if (candidate.queues.graphics_family == candidate.queues.present_family)
        1 // nvidia
    else
        2; // amd

    const required_device_extensions = comptime requiredDeviceExtensions(builtin.os);

    var device_extensions = try std.ArrayList([*:0]const u8).initCapacity(allocator, required_device_extensions.len);
    defer device_extensions.deinit();

    try device_extensions.appendSlice(required_device_extensions[0..required_device_extensions.len]);

    var count: u32 = undefined;
    _ = try vki.enumerateDeviceExtensionProperties(candidate.pdev, null, &count, null);

    const propsv = try allocator.alloc(vk.ExtensionProperties, count);
    defer allocator.free(propsv);

    _ = try vki.enumerateDeviceExtensionProperties(candidate.pdev, null, &count, propsv.ptr);

    for (optional_device_extensions) |extension_name| {
        for (propsv) |prop| {
            if (std.mem.eql(u8, prop.extension_name[0..prop.extension_name.len], std.mem.span(extension_name))) {
                try device_extensions.append(extension_name);
                break;
            }
        }
    }

    return try vki.createDevice(candidate.pdev, &.{
        .flags = .{},
        .queue_create_info_count = queue_count,
        .p_queue_create_infos = &qci,
        .enabled_layer_count = 0,
        .pp_enabled_layer_names = undefined,
        .enabled_extension_count = @as(u32, @intCast(device_extensions.items.len)),
        .pp_enabled_extension_names = @as([*]const [*:0]const u8, @ptrCast(device_extensions.items)),
        .p_enabled_features = null,
    }, null);
}

const DeviceCandidate = struct {
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    queues: QueueAllocation,
};

const QueueAllocation = struct {
    graphics_family: u32,
    present_family: u32,
};

fn pickPhysicalDevice(
    vki: InstanceDispatch,
    instance: vk.Instance,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) !DeviceCandidate {
    var device_count: u32 = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &device_count, null);

    const pdevs = try allocator.alloc(vk.PhysicalDevice, device_count);
    defer allocator.free(pdevs);

    _ = try vki.enumeratePhysicalDevices(instance, &device_count, pdevs.ptr);

    for (pdevs) |pdev| {
        if (try checkSuitable(vki, pdev, allocator, surface)) |candidate| {
            return candidate;
        }
    }

    return error.NoSuitableDevice;
}

fn checkSuitable(
    vki: InstanceDispatch,
    pdev: vk.PhysicalDevice,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) !?DeviceCandidate {
    const props = vki.getPhysicalDeviceProperties(pdev);

    if (!try checkExtensionSupport(vki, pdev, allocator)) {
        return null;
    }

    if (!try checkSurfaceSupport(vki, pdev, surface)) {
        return null;
    }

    if (try allocateQueues(vki, pdev, allocator, surface)) |allocation| {
        return DeviceCandidate{
            .pdev = pdev,
            .props = props,
            .queues = allocation,
        };
    }

    return null;
}

fn allocateQueues(vki: InstanceDispatch, pdev: vk.PhysicalDevice, allocator: Allocator, surface: vk.SurfaceKHR) !?QueueAllocation {
    var family_count: u32 = undefined;
    vki.getPhysicalDeviceQueueFamilyProperties(pdev, &family_count, null);

    const families = try allocator.alloc(vk.QueueFamilyProperties, family_count);
    defer allocator.free(families);
    vki.getPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families.ptr);

    var graphics_family: ?u32 = null;
    var present_family: ?u32 = null;

    for (families, 0..) |properties, i| {
        const family = @as(u32, @intCast(i));

        if (graphics_family == null and properties.queue_flags.graphics_bit) {
            graphics_family = family;
        }

        if (present_family == null and (try vki.getPhysicalDeviceSurfaceSupportKHR(pdev, family, surface)) == vk.TRUE) {
            present_family = family;
        }
    }

    if (graphics_family != null and present_family != null) {
        return QueueAllocation{
            .graphics_family = graphics_family.?,
            .present_family = present_family.?,
        };
    }

    return null;
}

fn checkSurfaceSupport(vki: InstanceDispatch, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
    var format_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);

    var present_mode_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);

    return format_count > 0 and present_mode_count > 0;
}

fn checkExtensionSupport(
    vki: InstanceDispatch,
    pdev: vk.PhysicalDevice,
    allocator: Allocator,
) !bool {
    var count: u32 = undefined;
    _ = try vki.enumerateDeviceExtensionProperties(pdev, null, &count, null);

    const propsv = try allocator.alloc(vk.ExtensionProperties, count);
    defer allocator.free(propsv);

    _ = try vki.enumerateDeviceExtensionProperties(pdev, null, &count, propsv.ptr);

    const required_device_extensions = comptime requiredDeviceExtensions(builtin.os);

    for (required_device_extensions) |ext| {
        for (propsv) |props| {
            const len = std.mem.indexOfScalar(u8, &props.extension_name, 0).?;
            const prop_ext_name = props.extension_name[0..len];
            if (std.mem.eql(u8, std.mem.span(ext), prop_ext_name)) {
                break;
            }
        } else {
            return false;
        }
    }

    return true;
}
