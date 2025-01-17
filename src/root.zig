const std = @import("std");
const testing = std.testing;

test {
    _ = @import("ndarray.zig");
    testing.refAllDecls(@This());
}
