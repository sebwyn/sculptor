const std = @import("std");

const Node = struct {
    child_mask: u8,
    children: [8]?*Node,

    fn deinit(self: *Node, allocator: std.mem.Allocator) void {
        defer allocator.destroy(self);
        for (0..8) |i| {
            if (self.children[i]) |child| {
                child.deinit(allocator);
            }
        }
    }
};

pub fn construct_svo(allocator: std.mem.Allocator, data_size: usize, data: []const u8) ?*Node {
    return do_construct_svo(allocator, [3]usize{ 0, 0, 0 }, data_size, data_size, data);
}

pub fn do_construct_svo(allocator: std.mem.Allocator, pos: [3]usize, size: usize, data_size: usize, data: []const u8) ?*Node {
    var child_mask: u8 = 0;
    var child_ptrs: [8]?*Node = .{ null } ** 8;
    for (0..2) |x| {
        for (0..2) |y| {
            for (0..2) |z| {
                const i: u3 = @intCast(x + y * 2 + z * 2 * 2);
                const child_bit = @as(u8, 0b10000000) >> i;
                if (size > 2) {
                    const half_size = size / 2;
                    const subtree_start: [3]usize = .{ pos[0] + x * half_size, pos[1] + y * half_size, pos[2] + z * half_size };
                    if(do_construct_svo(allocator, subtree_start, size / 2, data_size, data)) |child| {
                        child_mask |= child_bit;
                        child_ptrs[i] = child;
                    }
                } else {
                    const data_pos = (pos[0] + x) + (pos[1] + y) * data_size + (pos[2] + z) * data_size * data_size;
                    if (data[data_pos] > 0) {
                        child_mask |= child_bit;
                    }
                }
            }
        }
    }
    if (child_mask > 0) { 
        const child_ptr = allocator.create(Node) catch unreachable;
        child_ptr.* = Node { .child_mask = child_mask, .children = child_ptrs }; 
        return child_ptr; 
    } else { 
        return null; 
    }
}

test "cleans up after itself" {
    var dense_grid = [1]u8{ 0 } ** (8*8*8);
    dense_grid[0] = 1;
    dense_grid[4 + 4 * 8 + 4 * 8 * 8] = 1;

    const svo = construct_svo(std.testing.allocator, 8, &dense_grid).?; 
    svo.deinit(std.testing.allocator);
}

test "can construct from 8x8x8 dense grid" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    const allocator = gpa.allocator();

    var dense_grid = [1]u8{ 0 } ** (8*8*8);
    dense_grid[0] = 1;
    dense_grid[4 + 4 * 8 + 4 * 8 * 8] = 1;

    const svo: *Node = construct_svo(allocator, 8, &dense_grid).?; 
    defer svo.deinit(allocator);
    
    try std.testing.expectEqual(0b10000001, svo.child_mask);
    for (1..7) |i| { try std.testing.expectEqual(null, svo.children[i]); }

    try std.testing.expectEqual(0b10000000, svo.children[0].?.child_mask);
    try std.testing.expectEqual(0b10000000, svo.children[0].?.children[0].?.child_mask);
    for (0..8) |i| { try std.testing.expectEqual(null, svo.children[0].?.children[0].?.children[i]); }

    try std.testing.expectEqual(0b10000000, svo.children[7].?.child_mask);
    try std.testing.expectEqual(0b10000000, svo.children[7].?.children[0].?.child_mask);
    for (0..8) |i| { try std.testing.expectEqual(null, svo.children[7].?.children[0].?.children[i]); }
}
