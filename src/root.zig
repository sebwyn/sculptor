const std = @import("std");
const testing = std.testing;

test {
    _ = @import("ndarray.zig");
    _ = @import("svo.zig");
    testing.refAllDecls(@This());
}
