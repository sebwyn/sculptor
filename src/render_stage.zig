const vk = @import("vulkan");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Window = @import("window.zig").Window;
const std = @import("std");
const Texture = @import("texture.zig").Texture;

const Renderer = struct {
    render_stages: []RenderStage,
    swapchain: Swapchain, 
    command_buffers: []vk.CommandBuffer,
    window: *Window,

    pub fn render(self: *Renderer, gc: *const GraphicsContext) void {
        
        for (self.render_stages) |render_stage| {
            const command_buffer = self.command_buffers[self.swapchain.image_index];
            
            render_stage.beginRenderPass(gc, self.swapchain.image_index);
            
            for (render_stage.subpasses) |subpass| {
                subpass.render(gc, command_buffer);
                gc.vkd.vkCmdNextSubpass(command_buffer, vk.VK_SUBPASS_CONTENTS_INLINE);
            }

            gc.vkd.cmdEndRenderPass(self.command_buffers[self.swapchain.image_index]);

            if (render_stage.swapchain_attachment != null) {

                try gc.vkd.endCommandBuffer(self.command_buffers[self.swapchain.image_index]);
                
                if (render_stage.swapchain_attachment == null) { return; }

                const state = self.swapchain.present() catch |err| switch(err) {
                    error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
                    else => |narrow| return narrow,
                };

                if (state == .suboptimal) {
                    
                    const size = self.window.glfw_window.getSize();
                    try self.swapchain.recreate(vk.Extent2D {
                        .width = size.width,
                        .height = size.height
                    });
                    
                    // RECREATE render stage
                    // depth_texture.deinit(&gc);
                    // depth_texture = try Texture(2).init(&gc, .{ swapchain.extent.width, swapchain.extent.height }, depth_texture_options);
                    
                    // destroyFramebuffers(&gc, allocator, framebuffers);
                    // framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain, &depth_texture);
                    

                    // RECREATE command buffers
                    // destroyCommandBuffers(&gc, pool, allocator, cmdbufs);
                    // cmdbufs = try createCommandBuffers(
                    //     &gc,
                    //     pool,
                    //     allocator,
                    //     vertex_buffer.vk_handle,
                    //     swapchain.extent,
                    //     render_pass,
                    //     pipeline,
                    //     camera_descriptor_sets,
                    //     voxel_object_store.getDescriptorSets(),
                    //     pipeline_layout,
                    //     framebuffers,
                    // );
                }
            } 
        }
    }
};

const GraphicsPipeline = struct {
    pipeline: vk.Pipeline,
    descriptor_pool: vk.DescriptorPool,
    pipeline_bindpoint: vk.PipelineBindPoint,
    
    pub fn bind(self: *GraphicsPipeline, gc: *const GraphicsContext, command_buffer: *vk.CommandBuffer) bool {
        gc.vkd.cmdBindPipeline(command_buffer, self.pipeline_bindpoint, self.pipeline);

        // for () {
        //     gc.vkd.cmdBindDescriptorSets()
        // }
    }
};

const Subpass = struct {
    pipeline: GraphicsPipeline,

    pub fn render(self: *GraphicsPipeline, gc: *const GraphicsContext, command_buffer: *vk.CommandBuffer) bool {
        self.pipeline.bind(gc, command_buffer);
    }
};

const Attachment = struct {
    name: []const u8,
    kind: Kind,
    format: vk.Format, 
    clear_color: vk.ClearValue,

    const Kind = enum {
        Image,
        Swapchain,
        Depth
    };
};

const RenderStage = struct {
    attachments: []Attachment,
    framebuffers: []vk.Framebuffer,
    subpasses: []Subpass,

    render_area: vk.Rect2D,
    render_pass: vk.RenderPass,
    clear_values: []vk.ClearValue,

    swapchain_attachment: ?vk.ImageView,
    depth_attachment: ?vk.ImageView,
    allocator: std.mem.Allocator,


    pub fn init(allocator: std.mem.Allocator, swapchain: *const Swapchain) RenderStage {

        return RenderStage {
            .attachments = attachments,
            .framebuffers = framebuffers,
            .subpasses = subpasses,
            .render_area = render_area,
            .render_pass = render_pass,
            .clear_values = clear_values,
            .swapchain_attachment = swapchain_attachment,
            .depth_attachment = depth_attachment,
            .allocator = allocator,
        };    
    }

    pub fn deinit(self: *RenderStage, gc: *const GraphicsContext) void {
        for (self.framebuffers) |framebuffer| { gc.vkd.destroyFramebuffer(gc.dev, framebuffer, null); }
        self.allocator.free(self.framebuffers);
        gc.vkd.destroyRenderPass(gc.dev, self.render_pass, null);
    }
    
    pub fn beginRenderPass(self: *RenderStage, gc: *const GraphicsContext, command_buffer: *vk.CommandBuffer, swapchain_image_index: u32) void { 
        const viewport = vk.Viewport {
            .x = 0.0,
            .y = 0.0,
            .width = self.render_area.extent.width,
            .height = self.render_area.height,
            .min_depth = 0.0,
            .max_depth = 1.0,
        };
        gc.vkd.cmdSetViewport(command_buffer, 0, 1, @as([*]const vk.Viewport, @ptrCast(&viewport)));
        gc.vkd.cmdSetScissor(command_buffer, 0, 1, @as([*]const vk.Rect2D, @ptrCast(&self.render_area)));

        gc.vkd.cmdBeginRenderPass(command_buffer, &.{
            .render_pass = self.render_pass,
            .framebuffer = self.framebuffers[swapchain_image_index],
            .render_area = self.render_area,
            .clear_value_count = self.clear_value.len,
            .p_clear_values = self.clear_values.ptr,
        }, .@"inline");
    }

    pub fn createRenderpass(self: *RenderStage, gc: *const GraphicsContext, depth_format: vk.Format, surface_format: vk.Format) void {
        const attachment_descriptions = self.allocator.alloc(vk.AttachmentDescription, self.attachments.len);
        defer self.allocator.free(attachment_descriptions);

        for (0.., self.attachments) |i, attachment| {
            var attachment_description = vk.AttachmentDescription{
                .samples = .{ .@"1_bit" = true },
                .load_op = .clear,
                .store_op = .store,
                .stencil_load_op = .dont_care,
                .stencil_store_op = .dont_care,
                .initial_layout = .undefined,
            };

            switch (attachment.kind) {
                Attachment.Kind.Image => {
                    attachment_description.final_layout = .color_attachment_optimal; 
                    attachment_description.format = attachment.format;
                },
                Attachment.Kind.Depth => {
                    attachment_description.final_layout = .depth_stencil_attachment_optimal;
                    attachment_description.format = depth_format;
                },
                Attachment.Kind.Swapchain => {
                    attachment_description.final_layout = .present_src_khr;
                    attachment_description.format = surface_format;
                }
            }
            attachment_descriptions[i] = attachment_description;
        }

        const subpass_descriptions = self.allocator.alloc(vk.SubpassDescription, self.subpasses.len);
        defer self.allocator.free(subpass_descriptions);

        const subpass_dependencies = self.allocator.alloc(vk.SubpassDependency, self.subpasses.len);
        defer self.allocator.free(subpass_dependencies);
        
        for (0.., self.subpasses) |i, subpass| {
            const subpass_color_attachments = std.ArrayList(vk.AttachmentReference).init(self.allocator); 
            const depth_attachment: ?vk.AttachmentReference = null;
            
            for (subpass.attachment_bindings) |attachment_binding| {
                const attachment = self.getAttachment(attachment_binding);

                if (attachment == null) {
                    std.debug.print("Failed to find attachment with binding {any} on render stage", attachment_binding);
                    continue;
                }
                
                if (attachment.kind == Attachment.Kind.Depth) {
                    depth_attachment = vk.AttachmentReference {
                        .attachment = attachment.binding,
                        .layout = .depth_stencil_attachment_optimal,
                    };
                    continue;
                }

                subpass_color_attachments.append(vk.AttachmentReference{
                    .attachment = attachment.binding,
                    .layout = .color_attachment_optimal,
                });
            }
            
            subpass_descriptions[i] = vk.SubpassDescription {
                .pipeline_bind_point = .graphics,
                .color_attachment_count = subpass_color_attachments.len,
                .p_color_attachments = @ptrCast(subpass_color_attachments.items.ptr),
            };
            if (depth_attachment) |dp| { subpass_descriptions[i].p_depth_stencil_attachment = &dp; }

            var subpass_dependency = vk.SubpassDependency {
                .src_stage_mask = .{ .color_attachment_output_bit = true },
                .dst_stage_mask = .{ .fragment_shader_bit = true },
                .src_access_mask = .{ .shader_read_bit =  true },
                .dependency_flags = .{ .by_region_bit =  true },
            };

            if (subpass.binding == (self.subpasses.len - 1)) {
                subpass_dependency.dst_subpass = vk.SUBPASS_EXTERNAL;
                subpass_dependency.dst_stage_mask = .{ .bottom_of_pipe_bit =  true };
                subpass_dependency.src_access_mask = .{ .color_attachment_read_bit =  true, .color_attachment_write_bit = true };
                subpass_dependency.dst_access_mask = .{ .memory_read_bit =  true };
            } else {
                subpass_dependency.dst_subpass = subpass.binding;
            }

            if (subpass.binding == 0) {
                subpass_dependency.src_subpass = vk.SUBPASS_EXTERNAL;
                subpass_dependency.src_stage_mask = .{ .bottom_of_pipe_bit =  true };
                subpass_dependency.dst_stage_mask = .{ .color_attachment_output_bit =  true };
                subpass_dependency.src_access_mask = .{ .memory_read_bit =  true };
                subpass_dependency.dst_access_mask = .{ .color_attachment_read_bit =  true, .color_attachment_write_bit = true };
            }

            subpass_dependencies[i] = subpass_dependency;
        }

        return try gc.vkd.createRenderPass(gc.dev, &vk.RenderPassCreateInfo{
            .attachment_count = attachment_descriptions.len,
            .p_attachments = attachment_descriptions.ptr,
            .subpass_count = subpass_descriptions.len,
            .p_subpasses = subpass_descriptions.ptr,
            .dependency_count = subpass_dependencies.len,
            .p_dependencies = subpass_dependencies.ptr,
        }, null);
    }

    pub fn createFramebuffers(self: *RenderStage, gc: *const GraphicsContext, swapchain: *const Swapchain) void {
        const image_textures = self.allocator.alloc([]?Texture(2), self.attachments.len);
        var found_depth = false;
        
        const attachment_image_size = .{ self.render_area.extent.width, self.render_area.extent.height};
        for (0.., self.attachments) |i, attachment| {
            switch(attachment.kind) {
                Attachment.Kind.Image => {
                    image_textures[i] = (Texture(2).init(gc, attachment_image_size, Texture.TextureOptions {
                        .format = attachment.format, 
                        .usage = .{ .color_attachment_bit = true },
                        .initial_layout = vk.ImageLayout { .color_attachment_optimal } 
                    }));
                },
                Attachment.Kind.Depth  => if (!found_depth) {
                    image_textures[i] = try Texture(2).init(gc, attachment_image_size, Texture.TextureOptions { 
                        .format = .d32_sfloat,
                        .usage = .{ .depth_stencil_attachment_bit = true },
                        .aspect_mask = .{ .depth_bit = true } 
                    });
                    found_depth = true;
                }
                else {},
            }
        }

        const framebuffers = try self.allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
        errdefer self.allocator.free(framebuffers);

        var i: usize = 0;
        errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

        for (0.., swapchain.swap_images) |swap_image_index, swap_image| {
            const attachments = try self.allocator.alloc(vk.ImageView, self.attachments.len);
            for (self.attachments) |*attachment| {
                attachments.* = switch (attachment.kind)  {
                    Attachment.Kind.Image | Attachment.Kind.Depth => image_textures[i].view,
                    Attachment.Kind.Swapchain => swap_image.view,
                };
            }

            self.framebuffers[swap_image_index] = try gc.vkd.createFramebuffer(gc.dev, &vk.FramebufferCreateInfo{
                .render_pass = self.render_pass,
                .attachment_count = attachments.len,
                .p_attachments = attachments.ptr,
                .width = self.render_area.extent.width,
                .height = self.render_area.extent.height,
                .layers = 1,
            }, null);
            i += 1;
        }
    }
};
