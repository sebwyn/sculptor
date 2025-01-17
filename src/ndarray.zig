const std = @import("std");

// TODO: Currently several of these functions panic on out of bounds, change this at the expense of all usages having
// to explicitly handle the error
// alternatively provide a way to check if in index is in bounds
// or like in rust, checked and unchecked variants

pub fn NdarrayView(T: type) type {
    return struct {
        const Self = @This();

        buffer: []T,
        shape: []const usize,
        strides: []const usize,

        fn toOffset(self: *const Self, index: []const usize) usize {
            var offset: usize = 0;
            for (self.strides[0..index.len], index) |stride, i| { offset += stride * i; }
            return offset;
        }

        pub fn fill(self: *Self, fill_value: T) !void {
            @memset(self.buffer, fill_value);
        }

        pub fn at(self: *const Self, index: []const usize) *T {
            if (index.len != self.shape.len) { @panic("Must specify an index along all axes when getting a value from an ndarray!"); }
            return &self.buffer[self.toOffset(index)];
        }
        
        pub fn slice(self: *const Self, index: []const usize) Self {
            const slice_start = self.toOffset(index);
            const slice_end = slice_start + self.strides[index.len - 1];

            return Self {
                .shape = self.shape[index.len..],
                .strides = self.strides[index.len..],
                .buffer = self.buffer[slice_start..slice_end],
            };
        }

        pub fn write(self: *const Self, data: []const T) void {
            @memcpy(self.buffer, data);
        }
    };
}

pub fn Ndarray(T: type) type {
    return struct {
        const Self = @This();

        ndarray_view: NdarrayView(T),
        allocator: std.mem.Allocator,

        pub fn init(shape_: []const usize, allocator: std.mem.Allocator) Self {
            var size: usize = 1; 
            for (shape_) |axis_len| { size *= axis_len; }

            var current_stride = size;
            var strides = allocator.alloc(usize, shape_.len) catch unreachable;
            for (0.., shape_) |i, axis_len| {
                current_stride /= axis_len;
                strides[i] = current_stride;
            }
            
            const owned_shape = allocator.alloc(usize, shape_.len) catch unreachable;
            @memcpy(owned_shape, shape_);

            return Self {
                .ndarray_view = NdarrayView(T) {
                    .buffer = allocator.alloc(T, size) catch unreachable,
                    .shape = owned_shape,
                    .strides = strides,
                },
                .allocator = allocator,
            };
        }

        pub fn cloneFromView(view: NdarrayView(T), allocator: std.mem.Allocator) Self {
            const buffer_ = allocator.allocate(T, view.buffer.len);
            const shape_ = allocator.allocate(usize, view.buffer.stride);
            const stride = allocator.allocate(usize, view.buffer.stride);

            return Self {
                .ndarray_view = NdarrayView(T) {
                    .buffer = buffer_,
                    .shape = shape_,
                    .stride = stride,
                },
                .allocator = allocator,
            };
        }

        pub inline fn shape(self: *const Self ) []const usize { return self.ndarray_view.shape; }
        pub inline fn buffer(self: *const Self) []const T { return self.ndarray_view.buffer; }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.ndarray_view.shape);
            self.allocator.free(self.ndarray_view.strides);
            self.allocator.free(self.ndarray_view.buffer);
        }

        pub fn fill(self: *Self, fill_value: []usize) void { self.ndarray_view.fill(fill_value); }
        pub fn at(self: *const Self, index: []const usize) *T { return self.ndarray_view.at(index); }
        pub fn slice(self: *const Self, index: []const usize) NdarrayView(T) { return self.ndarray_view.slice(index); }
        pub fn write(self: *const Self, data: []const T) void { return self.ndarray_view.write(data); }

    };
}

fn create_ndarray() Ndarray(u8) {
    return Ndarray(u8).init(&[_]usize{ 3, 6, 9, 10}, std.testing.allocator);
}

test "Ndarray owns all data and doesn't leak" {
    const array: Ndarray(u8) = create_ndarray();
    defer array.deinit();
    try std.testing.expectEqualDeep(&[_]usize{ 3, 6, 9, 10 }, array.shape());
}

test "Ndarray write modifies buffer" {
    var array: Ndarray(u8) = Ndarray(u8).init(&[_]usize{ 2, 3 }, std.testing.allocator);
    try std.testing.expectEqual(2 * 3, array.buffer().len);

    defer array.deinit();
    array.write(&[_]u8{ 0, 1, 2, 4, 5, 6 });
    try std.testing.expectEqualDeep(&[_]u8{ 0, 1, 2, 4, 5, 6 }, array.buffer());
}

test "Ndarray slicing" {
    var array: Ndarray(u8) = Ndarray(u8).init(&[_]usize{ 3, 2 }, std.testing.allocator);
    defer array.deinit();
    array.write(&[_]u8{ 0, 1, 4, 5, 11, 12 });

    const slice = array.slice(&.{1});
    try std.testing.expectEqualDeep(&[_]u8{4, 5}, slice.buffer);
}

test "Ndarray writing to a view modifies the ndarray" {
    var array: Ndarray(u8) = Ndarray(u8).init(&[_]usize{ 3, 2 }, std.testing.allocator);
    defer array.deinit();
    array.write(&[_]u8{ 0, 1, 4, 5, 11, 12 });

    var slice1 = array.slice(&.{1});
    slice1.write(&.{2, 3});

    var slice2 = array.slice(&.{2});
    slice2.write(&.{4, 5});

    try std.testing.expectEqualDeep(&[_]u8{ 0, 1, 2, 3, 4, 5 }, array.buffer());
}

test "Ndarray indexing" {
    var array: Ndarray(u8) = Ndarray(u8).init(&[_]usize{ 2, 2, 4 }, std.testing.allocator);
    defer array.deinit();
    
    const trash_multidimensional_array = [2][2][4]u8{
        .{ 
            .{ 'a', 'b', 'c', 'd' },
            .{ 'e', 'f', 'g', 'h' },
        },
        .{
            .{ 'i', 'j', 'k', 'l'},
            .{ 'm', 'n', 'o', 'p'},
        }
    };

    for (0..2) |x| { for (0..2) |y| { for (0..4) |z| {
        array.at(&[_]usize{ x, y, z }).* = trash_multidimensional_array[x][y][z];
    }}}

    try std.testing.expectEqual('g', array.slice(&.{0}).at(&.{1, 2}).*);

    try std.testing.expectEqual('g', array.at(&.{ 0, 1, 2 }).*);
    try std.testing.expectEqualDeep(&[_]u8{ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'}, array.buffer()[0..9]);

}
