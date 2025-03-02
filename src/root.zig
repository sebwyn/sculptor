const std = @import("std");
const testing = std.testing;

test {
    _ = @import("ndarray.zig");
    _ = @import("svo.zig");
    _ = @import("shader.zig");
    testing.refAllDecls(@This());
}
