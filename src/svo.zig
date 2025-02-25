const std = @import("std");

const ChildDescriptor = packed struct {
    child_ptr: u15,
    far: bool,
    valid_mask: u8,
    leaf_mask: u8
};

const SVOError = error { DataLenMismatch, OutOfBufferSpace };

const SVOBuffer = struct {
    descriptors: [32_767]ChildDescriptor,
    len: u15,

    pub fn init() SVOBuffer {
        return SVOBuffer {
            .descriptors = undefined,
            .len = 0,
        };
    }

    pub fn push(self: *SVOBuffer, node: ChildDescriptor) !u15 {
        const index = self.len;
        if (self.len < self.descriptors.len) {
            self.descriptors[index] = node;
            self.len += 1;
        } else {
            return SVOError.OutOfBufferSpace;
        }
        return index;
    }
};

const SVONode = union(enum) {
    uniform: bool,
    mixed: ChildDescriptor,
};

const SparseVoxelOctree = struct {
    buffer: SVOBuffer, 

    pub fn from_dense(cube_size: usize, data: []const u8) !SparseVoxelOctree {
        const expected_data_len = cube_size * cube_size * cube_size;
        if(expected_data_len != data.len) {
            return SVOError.DataLenMismatch;
        }

        var buffer = SVOBuffer.init();
        
        const root_index = try buffer.push(ChildDescriptor { .child_ptr = 0, .far = false, .valid_mask = 0, .leaf_mask = 0});
        switch (do_from_dense(&buffer, [3]usize{ 0, 0, 0 }, cube_size, cube_size, data)) {
            .mixed => |root_node| { buffer.descriptors[root_index] = root_node; },
            else => {}
        }
        return SparseVoxelOctree { .buffer = buffer };
    }

    fn do_from_dense(svo_buffer_writer: *SVOBuffer, pos: [3]usize, size: usize, data_size: usize, data: []const u8) SVONode {
        var valid_mask: u8 = 0;
        var leaf_mask: u8 = 0;

        const half_size = size / 2;

        var child_i: usize = 0;
        var children: [8]ChildDescriptor = undefined;
        for (0..8) |i| {
            const x = (i & 0b001) >> 0;
            const y = (i & 0b010) >> 1;
            const z = (i & 0b100) >> 2;
            const child_bit: u8 = @as(u8, 1) << @intCast(i);
            
            const node_pos = .{ pos[0] + x * half_size, pos[1] + y * half_size, pos[2] + z * half_size };
            if (half_size == 1) {
                const node_index = node_pos[0] + node_pos[1] * data_size + node_pos[2] * data_size * data_size;
                if (data[node_index] > 0) {
                    leaf_mask |= child_bit;
                    valid_mask |= child_bit;
                }
            } else if (half_size > 1) {
                switch (do_from_dense(svo_buffer_writer, node_pos, half_size, data_size, data)) {
                    .uniform => |filled| if (filled) { valid_mask |= child_bit; leaf_mask |= child_bit; },
                    .mixed => |child| {
                        valid_mask |= child_bit;
                        children[child_i] = child;
                        child_i += 1; 
                    },
                }
            }
        }

        var child_ptr: u15 = 0;
        if (child_i > 0) {
            child_ptr = svo_buffer_writer.push(children[0]) catch unreachable;
            for (1..child_i) |i| { _ = svo_buffer_writer.push(children[i]) catch unreachable; }
        }
        
        if (valid_mask == 0) {
            return SVONode { .uniform = false };
        } else if (leaf_mask == 255) {
            return SVONode { .uniform = true };
        } else {
            const child_descriptor = ChildDescriptor { .child_ptr = child_ptr, .far = false, .valid_mask = valid_mask, .leaf_mask = leaf_mask };
            return SVONode { .mixed = child_descriptor };
        }
    }
};

test "cleans up after itself" {
    var dense_grid = [1]u8{ 0 } ** (8*8*8);
    dense_grid[0] = 1;
    dense_grid[4 + 4 * 8 + 4 * 8 * 8] = 1;

    _ = try SparseVoxelOctree.from_dense(8, &dense_grid); 
}

test "can construct from 8x8x8 dense grid" {
    var dense_grid = [1]u8{ 0 } ** (8*8*8);
    dense_grid[0] = 1;
    dense_grid[1] = 1;
    dense_grid[4 + 4 * 8 + 4 * 8 * 8] = 1;
    dense_grid[4 + 4 * 8 + 5 * 8 * 8] = 1;
    
    const starting_sub_grid_x = 6;
    const starting_sub_grid_y = 4; 
    const starting_sub_grid_z = 4;
    
    for (starting_sub_grid_x..starting_sub_grid_x+2) |x| {
        for (starting_sub_grid_y..starting_sub_grid_y+2) |y| {
            for (starting_sub_grid_z..starting_sub_grid_z+2) |z| {
                dense_grid[x + y * 8 + z * 8 * 8] = 1;
            }
        }
    }

    const svo: SparseVoxelOctree = try SparseVoxelOctree.from_dense(8, &dense_grid); 
     
    try std.testing.expectEqual(5, svo.buffer.len);

    const root_node = svo.buffer.descriptors[0];

    try std.testing.expectEqual(0b10000001, root_node.valid_mask);
    try std.testing.expectEqual(0, root_node.leaf_mask);
    
    const child_1 = svo.buffer.descriptors[root_node.child_ptr];
    const child_2 = svo.buffer.descriptors[root_node.child_ptr + 1];
    
    try std.testing.expectEqual(0b1, child_1.valid_mask);
    try std.testing.expectEqual(0, child_1.leaf_mask);
    try std.testing.expectEqual(0b11, child_2.valid_mask);
    try std.testing.expectEqual(0b10, child_2.leaf_mask);

    const child_3 = svo.buffer.descriptors[child_1.child_ptr];
    const child_4 = svo.buffer.descriptors[child_2.child_ptr]; 
        
    try std.testing.expectEqual(0b11, child_3.valid_mask);
    try std.testing.expectEqual(0b11, child_3.leaf_mask);
    try std.testing.expectEqual(0b10001, child_4.valid_mask);
    try std.testing.expectEqual(0b10001, child_4.leaf_mask);
}
