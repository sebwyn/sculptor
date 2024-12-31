const std = @import("std");

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

    const exe = b.addExecutable(.{
        .name = "sculptor",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // exe.addLibraryPath(b.path("dependencies/VulkanSDK/1.3.290.0/macOS/lib/"));
    exe.linkSystemLibrary("vulkan");

    // Use mach-glfw
    const glfw_dep = b.dependency("mach-glfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("mach-glfw", glfw_dep.module("mach-glfw"));

    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");
    const vk_gen = b.dependency("vulkan_zig", .{}).artifact("vulkan-zig-generator");
    const vk_generate_cmd = b.addRunArtifact(vk_gen);
    vk_generate_cmd.addFileArg(registry);
    const vulkan_zig = b.addModule("vulkan-zig", .{
        .root_source_file = vk_generate_cmd.addOutputFileArg("vk.zig"),
    });
    exe.root_module.addImport("vulkan", vulkan_zig);

    var shader_dir = try std.fs.openDirAbsolute(b.path("shaders").getPath(b), .{ .iterate = true });
    defer shader_dir.close();

    var shader_paths = shader_dir.iterate();
    while (shader_paths.next() catch null) |input_file| {
        if (input_file.kind != .file) continue;
        var name_extension = std.mem.splitSequence(u8, input_file.name, ".");
        const name = name_extension.next() orelse continue;
        const extension = name_extension.next() orelse continue;

        if (std.mem.eql(u8, extension, "vert") or
            std.mem.eql(u8, extension, "frag") or
            std.mem.eql(u8, extension, "glsl"))
        {
            const output_file = try b.allocator.alloc(u8, input_file.name.len);
            defer b.allocator.free(output_file);
            _ = try std.fmt.bufPrint(output_file, "{s}_{s}", .{ name, extension });

            const compiled_file_name = try b.allocator.alloc(u8, input_file.name.len + 4);
            defer b.allocator.free(compiled_file_name);
            _ = try std.fmt.bufPrint(compiled_file_name, "{s}.spv", .{output_file});

            const compile_vert_shader = b.addSystemCommand(&.{"glslc"});
            compile_vert_shader.addFileArg(b.path(b.pathJoin(&.{ "shaders", input_file.name })));
            compile_vert_shader.addArgs(&.{ "--target-env=vulkan1.1", "-o" });
            const compiled_file = compile_vert_shader.addOutputFileArg(compiled_file_name);
            exe.root_module.addAnonymousImport(output_file, .{
                .root_source_file = compiled_file,
            });
        }
    }

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
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
}
