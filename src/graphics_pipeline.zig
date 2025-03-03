const vk = @import("vulkan");
const Shader = @import("shader.zig").Shader;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const std = @import("std");

const VertexInputDescription = struct {

};

const GraphicsPipeline = struct {
    shader: Shader,

    pub fn init(gc: *const GraphicsContext, allocator: std.mem.Allocator, shader: Shader) GraphicsPipeline {
        
        const descriptor_set_layouts = shader.createDescriptorSetLayouts(gc, allocator);

        const pipeline_layout = gc.vkd.createPipelineLayout(gc.dev, vk.PipelineLayoutCreateInfo {
            .set_layout_count = descriptor_set_layouts.len, 
            .p_set_layouts = descriptor_set_layouts.ptr, 
        }, null);



        GraphicsPipeline {
            .shader = shader,
        };
    }
};
