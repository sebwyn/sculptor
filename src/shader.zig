const vk = @import("vulkan");
const std = @import("std");
const spirv_reflect = @cImport({
    @cInclude("spirv_reflect.h");
});

pub const Shader = struct {
    files: []std.fs.path,
    uniforms: std.ArrayList(Uniform),
    allocator: std.mem.Allocator,

    pub const Uniform = struct {
        name: []u8,
        set: u32,
        binding: u32,
        descriptor_type: vk.DescriptorType,
        shader_stage: vk.ShaderStageFlags,
    };
    
    pub const ShaderCreateInfo = struct {
        entry_point: []const u8,
        spirv: []const u8,
    };
    pub fn init(allocator: std.mem.Allocator, shader_stages: []const ShaderCreateInfo) !Shader {
        var uniforms = std.ArrayList(Uniform).init(allocator);
        errdefer uniforms.deinit();

        for (shader_stages) |shader_stage| {

            var module: spirv_reflect.SpvReflectShaderModule = undefined; 
            const result = spirv_reflect.spvReflectCreateShaderModule(shader_stage.spirv.len, shader_stage.spirv.ptr, &module);
            if (result != spirv_reflect.SPV_REFLECT_RESULT_SUCCESS) {
                return ShaderError.FailedToReflect;
            }
            defer spirv_reflect.spvReflectDestroyShaderModule(&module);
            
            const shader_stage_bit = switch(module.shader_stage) {
                spirv_reflect.SPV_REFLECT_SHADER_STAGE_VERTEX_BIT => vk.ShaderStageFlags { .vertex_bit = true },
                spirv_reflect.SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT => vk.ShaderStageFlags { .fragment_bit = true },
                else => return ShaderError.UnsupportedShaderStage,
            };

            for (0..module.descriptor_binding_count) |i| {
                const descriptor_binding = module.descriptor_bindings[i];
                
                const descriptor_type: vk.DescriptorType = switch (descriptor_binding.descriptor_type) {
                    spirv_reflect.SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER => .combined_image_sampler,
                    spirv_reflect.SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER => .sampler,
                    spirv_reflect.SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE => .sampled_image,
                    spirv_reflect.SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER => .uniform_buffer,
                    spirv_reflect.SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT => .input_attachment,
                    else => return ShaderError.UnsupportedDescriptorType,
                };
                
                const name = allocator.dupe(u8, std.mem.span(descriptor_binding.name)) catch unreachable;

                uniforms.append(Uniform {
                    .name = name,
                    .set = descriptor_binding.set,
                    .binding = descriptor_binding.binding,
                    .shader_stage = shader_stage_bit,
                    .descriptor_type = descriptor_type,
                }) catch unreachable;

            }
        }

        return Shader {
            .files = &.{},
            .uniforms = uniforms,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *const Shader) void {
        for (self.uniforms.items) |uniform| {
            self.allocator.free(uniform.name);
        }
        self.uniforms.deinit();
    }
};

pub const ShaderError = error { UnexpectedShaderType, FailedToReflect, UnsupportedDescriptorType, UnsupportedShaderStage };

pub fn get_shader_stage(shader_path: std.fs.path) !vk.ShaderStageFlags {
    const extension = shader_path.extension;
    return if (std.mem.eql(extension, ".comp")) {
        vk.ShaderStageFlags { .compute_bit = true };
    } else if (std.mem.eql(extension, ".vert")) {
        vk.ShaderStageFlags { .vertex_bit = true };
    } else if (std.mem.eql(extension, ".frag")) {
        vk.ShaderStageFlags { .fragment_bit = true };
    } else {
        ShaderError.UnexpectedShaderType;
    };
}

fn reflect_shader_module(spirv_bytes: []const u8) void {
    var module: spirv_reflect.SpvReflectShaderModule = undefined;
    const result = spirv_reflect.spvReflectCreateShaderModule(spirv_bytes.len, spirv_bytes.ptr, &module);
    if (result != spirv_reflect.SPV_REFLECT_RESULT_SUCCESS) { return; }
    defer spirv_reflect.spvReflectDestroyShaderModule(&module);
    
    for (0..module.descriptor_binding_count) |i| {
        const descriptor_binding = module.descriptor_bindings[i];
        std.debug.print("{s}: {d} {d}\n", .{ descriptor_binding.name, descriptor_binding.set, descriptor_binding.binding });
    }
}

test "can load shader module" {
    const temp = @import("temp");

    const shader_source = &.{ 
        "#version 450",
        "",
        "",
        "layout(set=0, binding=1) uniform uniform_buffer {",
        "   mat4x4 view;",
        "   mat4x4 proj;",
        "   vec2 screen_size;",
        "} camera;",
        "",
        "layout(location = 0) in vec2 a_pos;",
        "layout(location = 1) in vec3 a_color;",
        "",
        "layout(location = 0) out vec3 v_color;",
        "",
        "void main() {",
        "    gl_Position = camera.view * camera.proj * vec4(a_pos, 0.0, 1.0);",
        "    v_color = a_color;",
        "}",
    };
    const shader_text = try std.mem.join(std.testing.allocator, "\n", shader_source);
    defer std.testing.allocator.free(shader_text);
    
    const compiled_bytes = block: {
        var tempdir = try temp.TempDir.create(std.testing.allocator, .{});
        defer tempdir.deinit();
        
        var d = try tempdir.open(.{});
        defer d.close();

        {
            const vert_shader_source_file = try d.createFile("test_shader.vert", .{});
            defer vert_shader_source_file.close();

            _ = try vert_shader_source_file.writeAll(shader_text);
        }
        
        const source_path = try d.realpathAlloc(std.testing.allocator, "test_shader.vert");
        defer std.testing.allocator.free(source_path);
        
        const spv_shader_file = try d.createFile("test_shader.spv", .{});
        spv_shader_file.close();

        const spv_path = try d.realpathAlloc(std.testing.allocator, "test_shader.spv");
        defer std.testing.allocator.free(spv_path);

        const result = try std.process.Child.run(.{
            .allocator = std.testing.allocator, 
            .argv = &.{ "glslc", source_path, "-o", spv_path }
        });
        try std.testing.expectEqual(std.process.Child.Term { .Exited = 0 }, result.term);


        const spv_file = try std.fs.openFileAbsolute(spv_path, .{});
        defer spv_file.close();

        const compiled_bytes = try spv_file.readToEndAlloc(std.testing.allocator, 10000);
        break :block compiled_bytes;
    };
    defer std.testing.allocator.free(compiled_bytes);

    const shader = try Shader.init(std.testing.allocator, &[_]Shader.ShaderCreateInfo{ 
        Shader.ShaderCreateInfo { .entry_point = "main", .spirv = compiled_bytes } 
    });
    defer shader.deinit();
    
    const name = try std.testing.allocator.dupe(u8, "camera");
    defer std.testing.allocator.free(name);

    const expected_uniforms = [_]Shader.Uniform{
        Shader.Uniform { .set = 0, .binding = 1, .name = name, .descriptor_type = .uniform_buffer, .shader_stage = .{ .vertex_bit = true }}
    };

    try std.testing.expectEqualDeep(&expected_uniforms, shader.uniforms.items);
}

test "can load real shader" {
    const aabb_raycaster_frag = @embedFile("aabb_raycaster_frag.spv");

    const shader = try Shader.init(std.testing.allocator, &.{
        .{ .entry_point = "main", .spirv = aabb_raycaster_frag }
    });
    defer shader.deinit();

    try std.testing.expectEqual(4, shader.uniforms.items.len);
    
    for (shader.uniforms.items) |item| {
        try std.testing.expectEqual(vk.ShaderStageFlags { .fragment_bit = true }, item.shader_stage);
    }
}
