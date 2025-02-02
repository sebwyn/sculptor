const zlm_helpers = @import("zlm");
const std = @import("std");
const zlm = @import("zlm").SpecializeOn(f32);
const VoxelObjectStore = @import("voxel_object_store.zig").VoxelObjectStore;
const VoxelObject = @import("voxel_object_store.zig").VoxelObject;

const VoxReaderError = error{ UnexpectedHeader, UnsupportedVersion, UnexpectedChunkType, ExpectedXYZIChunk, NoObjectsFound };

pub fn read_vox_file(file_path: []const u8, allocator: std.mem.Allocator, voxel_store: *VoxelObjectStore) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_reader = file.reader();

    var header: [4]u8 = undefined;
    _ = try file_reader.read(&header);
    if (!std.mem.eql(u8, &header, "VOX ")) {
        return VoxReaderError.UnexpectedHeader;
    }

    const version = try file_reader.readInt(i32, std.builtin.Endian.little);
    if (version != 150) {
        return VoxReaderError.UnsupportedVersion;
    }

    try read_all_chunks(file_reader, allocator, voxel_store);
}

fn read_chunk_header(file_reader: std.fs.File.Reader) !struct { id: [4]u8, chunk_len: i32, child_chunk_len: i32 } {
    var id: [4]u8 = undefined;
    _ = try file_reader.read(&id);

    const chunk_len = try file_reader.readInt(i32, std.builtin.Endian.little);
    const child_chunk_len = try file_reader.readInt(i32, std.builtin.Endian.little);

    return .{ .id = id, .chunk_len = chunk_len, .child_chunk_len = child_chunk_len };
}

fn read_all_chunks(file_reader: std.fs.File.Reader, allocator: std.mem.Allocator, voxel_store: *VoxelObjectStore) !void {
    var object_refs = std.ArrayList(VoxelObjectStore.Ref).init(allocator);
    defer object_refs.deinit();

    while (true) {
        const header = read_chunk_header(file_reader) catch |err| {
            if (err == std.fs.File.Reader.NoEofError.EndOfStream) {
                return;
            } else return err;
        };

        if (std.mem.eql(u8, &header.id, "SIZE")) {
            const x: u32 = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
            const y: u32 = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
            const z: u32 = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));

            const ref = try voxel_store.createEmpty(.{ x, y, z });
            const voxel_obj = voxel_store.getObjectMut(ref);
            try read_xyzi_chunk_into_texture(file_reader, allocator, voxel_store, voxel_obj);
            object_refs.append(ref) catch unreachable;
        } else if (std.mem.eql(u8, &header.id, "RGBA")) {
            if (object_refs.items.len == 0) {
                return VoxReaderError.NoObjectsFound;
            }

            var first_object = voxel_store.getObjectMut(object_refs.items[0]);

            const palette_staging_buffer = try first_object.palette.createStagingBuffer(voxel_store.gc, allocator);
            defer palette_staging_buffer.deinit(voxel_store.gc);
            for (0..254) |i| {
                var rgba: [4]u8 = undefined;

                _ = try file_reader.read(&rgba);
                //
                //some special casing for monu8 to make the water translucent
                if (rgba[1] > rgba[2] and rgba[1] > rgba[0]) {
                    rgba[3] = 32;
                }
                palette_staging_buffer.slice(&.{i}).write(&rgba);
                std.debug.print("{d}\n", .{rgba[3]});
            }

            for (object_refs.items) |obj| {
                const voxel_object = voxel_store.getObjectMut(obj);
                try voxel_object.palette.writeStagingBuffer(voxel_store.gc, voxel_store.command_pool, palette_staging_buffer);
            }
        } else if (std.mem.eql(u8, &header.id, "MAIN") or std.mem.eql(u8, &header.id, "PACK")) {
            try file_reader.skipBytes(@intCast(header.chunk_len), .{});
        } else {
            return VoxReaderError.UnexpectedChunkType;
        }
    }
}

fn read_xyzi_chunk_into_texture(file_reader: std.fs.File.Reader, allocator: std.mem.Allocator, voxel_store: *VoxelObjectStore, voxel_obj: *VoxelObject) !void {
    const header = try read_chunk_header(file_reader);

    if (!std.mem.eql(u8, &header.id, "XYZI")) {
        return VoxReaderError.ExpectedXYZIChunk;
    }

    const scale = zlm.Mat4.createScale(0.2, 0.2, 0.2);
    const rot = zlm.Mat4.createAngleAxis(zlm.Vec3.new(1, 0, 0), zlm_helpers.toRadians(-90.0));
    try voxel_obj.transform_buffer.write(voxel_store.gc, &.{rot.mul(scale)});

    {
        const staging_buffer = try voxel_obj.voxels.createStagingBuffer(voxel_store.gc, allocator);
        defer staging_buffer.deinit(voxel_store.gc);
        const number_of_voxels: usize = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
        for (0..number_of_voxels) |_| {
            var xyzi: [4]u8 = undefined;
            _ = try file_reader.read(&xyzi);
            const index: [3]usize = .{ xyzi[0], xyzi[1], xyzi[2] };
            staging_buffer.at(&index).* = xyzi[3];
        }
        try voxel_obj.voxels.writeStagingBuffer(voxel_store.gc, voxel_store.command_pool, staging_buffer);
    }
}
