const vk = @import("vulkan");
const std = @import("std");
const spirv_reflect = @cImport({
    @cInclude("spirv_reflect.h");
});

pub const Shader = struct {
    files: []std.fs.path,
    uniforms: []Uniform,

    pub const Uniform = struct {
        binding: u32,
        offset: u32,
        size: u32,
        gl_type: u32, 
        read_only: bool,
        write_only: bool,
        stage_flags: vk.ShaderStageFlags,
    };
    
    pub const ShaderCreateInfo = struct {
        entry_point: []const u8,
        spirv: []const u8,
    };
    pub fn init(shader_stages: []const ShaderCreateInfo) Shader {
        _ = shader_stages;
    }
};

pub const ShaderError = error { UnexpectedShaderType };

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

    const shader_source = []const []const u8{ 
        "#version 450" ++ "\n",
        "",
        "layout(location = 0) in vec2 a_pos;",
        "layout(location = 1) in vec3 a_color;",
        "",
        "layout(location = 0) out vec3 v_color;",
        "",
        "void main() {",
        "    gl_Position = vec4(a_pos, 0.0, 1.0);",
        "    v_color = a_color;",
        "}",
    };
    const shader_text = std.mem.join(std.testing.allocator, "\n", shader_source);
    
    {
        const tempdir = try temp.TempDir.create(std.testing.allocator, .{});
        defer tempdir.deinit();
        
        const d = try tempdir.open();
        defer d.close();

        {
            const vert_shader_source_file = try d.createFile("test_shader.vert");
            defer vert_shader_source_file.close();

            _ = try vert_shader_source_file.writeAll(shader_text);
        }
        
        const source_path = d.realpath("test_shader.vert");
        const spv_source_path = d.realpath("test_shader.spv");
        _ = try std.process.Child.run(std.testing.allocator, .{ "glslc", source_path, "-o", spv_source_path });

        const spv_file = try std.fs.openFileAbsolute(spv_source_path);
        defer spv_file.close();

        const compiled_bytes = try spv_file.readToEndAlloc(std.testing.allocator, 10000);
        std.testing.expectEqual("", compiled_bytes);
    }
}

