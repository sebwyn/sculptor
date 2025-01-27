const std = @import("std");
const zlm = @import("zlm").SpecializeOn(f32);
const VoxelObjectStore = @import("voxel_object_store.zig").VoxelObjectStore;
const VoxelObject = @import("voxel_object_store.zig").VoxelObject;

const VoxReaderError = error { UnexpectedHeader, UnsupportedVersion, UnexpectedChunkType, ExpectedXYZIChunk };

pub fn read_vox_file(file_path: []const u8, allocator: std.mem.Allocator, voxel_store: *VoxelObjectStore) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close(); 
    
    var file_reader = file.reader(); 
    
    var header: [4]u8 = undefined;
    _ = try file_reader.read(&header);
    if (!std.mem.eql(u8, &header, "VOX ")) { return VoxReaderError.UnexpectedHeader; }

    const version = try file_reader.readInt(i32, std.builtin.Endian.little);
    if (version != 150) { return VoxReaderError.UnsupportedVersion; }

    while (try read_chunk(file_reader, allocator, voxel_store)) {}
}

fn read_chunk_header(file_reader: std.fs.File.Reader) !struct { id: [4]u8, chunk_len: i32, child_chunk_len: i32 } {
    var id: [4]u8 = undefined;
    _ = try file_reader.read(&id);

    const chunk_len = try file_reader.readInt(i32, std.builtin.Endian.little);
    const child_chunk_len = try file_reader.readInt(i32, std.builtin.Endian.little);

    return .{
        .id = id,
        .chunk_len = chunk_len,
        .child_chunk_len = child_chunk_len
    };
}

fn read_chunk(file_reader: std.fs.File.Reader, allocator: std.mem.Allocator, voxel_store: *VoxelObjectStore) !bool {
    const header = read_chunk_header(file_reader) catch |err| {
        if (err == std.fs.File.Reader.NoEofError.EndOfStream) { return false; }
        else return err;
    };

    std.debug.print("Found chunk: {s}, {d}, {d}\n", .{ header.id, header.chunk_len, header.child_chunk_len });
    
    if (header.chunk_len > 0) {
        if (std.mem.eql(u8, &header.id, "SIZE")) {
            const x: u32 = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
            const y: u32 = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
            const z: u32 = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
            std.debug.print("Found size chunk: {d}, {d}, {d}\n", .{ x, y, z });

            const ref = try voxel_store.createEmpty(.{ x, z, y });
            const voxel_obj = voxel_store.getObjectMut(ref);

            try read_xyzi_chunk_into_texture(file_reader, allocator, voxel_store, voxel_obj);
        } else if (std.mem.eql(u8, &header.id, "RGBA")) {
            
            try file_reader.skipBytes(@intCast(header.chunk_len), .{});
        } else if (std.mem.eql(u8, &header.id, "MAIN") or std.mem.eql(u8, &header.id, "PACK")) {
            try file_reader.skipBytes(@intCast(header.chunk_len), .{});
        } else {
            return VoxReaderError.UnexpectedChunkType;
        }

    }

    return true;
}

fn read_xyzi_chunk_into_texture(file_reader: std.fs.File.Reader, allocator: std.mem.Allocator, voxel_store: *VoxelObjectStore, voxel_obj: *VoxelObject) !void {
    const header = try read_chunk_header(file_reader);

    if (!std.mem.eql(u8, &header.id, "XYZI")) {
        return VoxReaderError.ExpectedXYZIChunk;
    }

    std.debug.print("Found chunk: {s}, {d}, {d}\n", .{ header.id, header.chunk_len, header.child_chunk_len });
    
    // const transform = zlm.Mat4.createTranslation(zlm.Vec3.new(17.8, 0, 0)).mul();
    try voxel_obj.transform_buffer.write(voxel_store.gc, &.{zlm.Mat4.createScale(0.2, 0.2, 0.2)});

    {
        const staging_buffer = try voxel_obj.voxels.createStagingBuffer(voxel_store.gc, allocator);
        // @memset(staging_buffer.buffer.data_ptr.?[0..staging_buffer.texture.byteSize()], 0);
        defer staging_buffer.deinit(voxel_store.gc);
        const number_of_voxels: usize = @intCast(try file_reader.readInt(i32, std.builtin.Endian.little));
        // std.debug.print("N: {any}\n", .{number_of_voxels});
        for (0..number_of_voxels) |_| {
            var xyzi: [4]u8 = undefined; 
            _ = try file_reader.read(&xyzi);
            // std.debug.print("{any}\n", .{xyzi});
            const index: [3]usize = .{ xyzi[0], xyzi[2], xyzi[1] };
            staging_buffer.at(&index).* = xyzi[3];
        }
        // @panic("get me out");
        try voxel_obj.voxels.writeStagingBuffer(voxel_store.gc, voxel_store.command_pool, staging_buffer);
    }
    
    {
        var rand = std.rand.DefaultPrng.init(0);
        const palette_staging_buffer = try voxel_obj.palette.createStagingBuffer(voxel_store.gc, allocator);
        defer palette_staging_buffer.deinit(voxel_store.gc);
        for (0..255) |i| {
            const color = &.{ rand.random().int(u8), rand.random().int(u8), rand.random().int(u8), 255 };
            palette_staging_buffer.slice(&.{i}).write(color);
        }
        try voxel_obj.palette.writeStagingBuffer(voxel_store.gc, voxel_store.command_pool, palette_staging_buffer);
    }
}
