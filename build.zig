const builtin = @import("builtin");
const std = @import("std");

const SculptorBuildError = error{CouldNotFindVulkan};

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "sculptor",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    const exe = try create_compile_step(b, target, optimize);
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
    
    const exe_check = try create_compile_step(b, target, optimize);
    const check = b.step("check", "Check if foo compiles");
    check.dependOn(&exe_check.step);
}

fn embed_shaders(b: *std.Build, module: *std.Build.Module) !void {
    var shader_dir = try std.fs.openDirAbsolute(b.path("shaders").getPath(b), .{ .iterate = true });
    defer shader_dir.close();

    var shader_paths = shader_dir.iterate();
    while (shader_paths.next() catch null) |input_file| {
        if (input_file.kind != .file) continue;
        
        var name_extension = std.mem.splitSequence(u8, input_file.name, ".");
        _ = name_extension.next() orelse continue;
        const extension = name_extension.next() orelse continue;

        if (std.mem.eql(u8, extension, "vert") or
            std.mem.eql(u8, extension, "frag") or
            std.mem.eql(u8, extension, "glsl"))
        {
            const output_file_name = try b.allocator.dupe(u8, input_file.name);
            _ = std.mem.replace(u8, input_file.name, ".", "_", output_file_name);

            const compiled_file_name = b.fmt("{s}.spv", .{ output_file_name} );

            const compile_shader = b.addSystemCommand(&.{"glslc"});
            compile_shader.addFileArg(b.path("shaders").path(b, input_file.name));
            compile_shader.addArgs(&.{ "--target-env=vulkan1.1", "-o" });
            const compiled_path = compile_shader.addOutputFileArg(compiled_file_name);
            
            module.addAnonymousImport(compiled_file_name, .{ .root_source_file = compiled_path });
        }
    }
}

fn create_compile_step(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !*std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "sculptor",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const glfw_dep = b.dependency("mach-glfw", .{ .target = target, .optimize = optimize, });

    const env = try std.process.getEnvMap(b.allocator);
    const vulkan_sdk_path = try (env.get("VULKAN_SDK") orelse error.CouldNotFindVulkan);
    exe.addLibraryPath(std.Build.LazyPath{ .cwd_relative = try std.fs.path.join(b.allocator, &[_][]const u8{ vulkan_sdk_path, "Lib" }) });
    const vulkan_library_name = switch (builtin.os.tag) { .windows => "vulkan-1", else => "vulkan" };
    exe.linkSystemLibrary(vulkan_library_name);

    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");
    const vk_gen = b.dependency("vulkan_zig", .{}).artifact("vulkan-zig-generator");
    const vk_generate_cmd = b.addRunArtifact(vk_gen);
    vk_generate_cmd.addFileArg(registry);
    const vulkan_zig = b.addModule("vulkan-zig", .{ .root_source_file = vk_generate_cmd.addOutputFileArg("vk.zig"), });

    exe.root_module.addImport("mach-glfw", glfw_dep.module("mach-glfw"));
    exe.root_module.addImport("perlin", b.dependency("perlin", .{}).module("perlin"));
    exe.root_module.addImport("vulkan", vulkan_zig);
    exe.root_module.addImport("zlm", b.dependency("zlm", .{}).module("zlm"));
    
    _ = try embed_shaders(b, &exe.root_module);

    return exe;
}
