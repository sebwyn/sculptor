const shader = @import("shader.zig");
const std = @import("std");

const VertexInputDescription = struct {

};

const GraphicsPipeline = struct {
    shader: shader.Shader,

    pub fn init(shaders: []std.fs.path) GraphicsPipeline {
        _ = shaders;

        GraphicsPipeline {
            .shader = shader.Shader.init(.{})
        };
    }
};
