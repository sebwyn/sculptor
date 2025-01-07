const std = @import("std");
const glfw = @import("mach-glfw");

pub const ScrollEventListener = struct { instance: *anyopaque, fn_ptr: *const fn(*anyopaque, glfw.Window, f64, f64) void, };
pub const MouseButtonListener = struct { instance: *anyopaque, fn_ptr: *const fn(*anyopaque, glfw.Window, glfw.MouseButton, glfw.Action, glfw.Mods) void, };
pub const CursorMoveListener = struct { instance: *anyopaque, fn_ptr: *const fn(*anyopaque, glfw.Window, f64, f64) void, };
pub const WindowRefreshListener = struct { instance: *anyopaque, fn_ptr: *const fn(*anyopaque, glfw.Window) void, };

pub const EventHandler = struct {
    scroll_event_listeners: std.ArrayList(ScrollEventListener),
    mouse_button_listeners: std.ArrayList(MouseButtonListener),
    cursor_move_listeners: std.ArrayList(CursorMoveListener),
    allocator: std.mem.Allocator,

    pub const Listeners = struct {
        scroll_listener: ?ScrollEventListener = null,
        mouse_button_listener: ?MouseButtonListener = null,
        cursor_move_listener: ?CursorMoveListener = null,
        window_refresh_listener: ?WindowRefreshListener = null,
    };
    
    pub fn init(allocator: std.mem.Allocator) EventHandler {
        return EventHandler {
            .scroll_event_listeners = std.ArrayList(ScrollEventListener).init(allocator),
            .mouse_button_listeners = std.ArrayList(MouseButtonListener).init(allocator),
            .cursor_move_listeners = std.ArrayList(CursorMoveListener).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn subscribe(self: *EventHandler, handlers: Listeners) void {
        if (handlers.scroll_listener) |scroll_handler| { self.scroll_event_listeners.append(scroll_handler) catch unreachable; }
        if (handlers.mouse_button_listener) |mouse_button_handler| { self.mouse_button_listeners.append(mouse_button_handler) catch unreachable; }
        if (handlers.cursor_move_listener) |cursor_move_handler| { self.cursor_move_listeners.append(cursor_move_handler) catch unreachable; }
    }

    pub fn handleMouseScroll(self: *const EventHandler, window: glfw.Window, x_offset: f64, y_offset: f64) void {
        for (self.scroll_event_listeners.items) |listener| { listener.fn_ptr(listener.instance, window, x_offset, y_offset); }
    }

    pub fn handleMouseButton(self: *const EventHandler, window: glfw.Window, button: glfw.MouseButton, action: glfw.Action, mods: glfw.Mods) void {
        for (self.mouse_button_listeners.items) |listener| { listener.fn_ptr(listener.instance, window, button, action, mods); }
    }

    pub fn handleCursorMove(self: *const EventHandler, window: glfw.Window, xpos: f64, ypos: f64) void {
        for (self.cursor_move_listeners.items) |listener| { listener.fn_ptr(listener.instance, window, xpos, ypos); }
    }

    pub fn deinit(self: EventHandler) void {
        self.scroll_event_listeners.deinit();
        self.mouse_button_listeners.deinit();
        self.cursor_move_listeners.deinit();
    }
};

pub const Window = struct {
    event_handler: []EventHandler,
    glfw_window: glfw.Window,
    allocator: std.mem.Allocator,
    window_resize_listeners: std.ArrayList(WindowRefreshListener),

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, app_name: []const u8) !Window {
        const events: []EventHandler = try allocator.alloc(EventHandler, 1);
        events[0] = EventHandler.init(allocator);

        const glfw_window = glfw.Window.create(width, height, @ptrCast(app_name), null, null, .{ .client_api = .no_api, }) orelse {
            std.log.err("failed to create GLFW window: {?s}", .{glfw.getErrorString()});
            std.process.exit(1);
        };
        glfw.Window.setUserPointer(glfw_window, @ptrCast(events));
        glfw.Window.setScrollCallback(glfw_window, handleMouseScroll);
        glfw.Window.setMouseButtonCallback(glfw_window, handleMouseButton);
        glfw.Window.setCursorPosCallback(glfw_window, handleCursorMove);

        return Window {
            .event_handler = events,
            .glfw_window = glfw_window,
            .allocator = allocator,
            .window_resize_listeners = std.ArrayList(WindowRefreshListener).init(allocator),
        };
    }

    pub fn subscribeToEvents(self: *Window, listeners: EventHandler.Listeners) void {
        self.event_handler[0].subscribe(listeners); 
        
        if (listeners.window_refresh_listener) |window_refresh_handler| { 
            self.window_resize_listeners.append(window_refresh_handler) catch unreachable; 
        }
    }

    pub fn handleMouseScroll(window: glfw.Window, x_offset: f64, y_offset: f64) void {
        const internal: *EventHandler = glfw.Window.getUserPointer(window, EventHandler) orelse @panic("U fucked up!");
        internal.handleMouseScroll(window, x_offset, y_offset);
    }
    
    pub fn handleMouseButton(window: glfw.Window, button: glfw.MouseButton, action: glfw.Action, mods: glfw.Mods) void {
        const internal: *EventHandler = glfw.Window.getUserPointer(window, EventHandler) orelse @panic("U fucked up!");
        internal.handleMouseButton(window, button, action, mods);
    }

    pub fn handleCursorMove(window: glfw.Window, xpos: f64, ypos: f64) void {
        const internal: *EventHandler = glfw.Window.getUserPointer(window, EventHandler) orelse @panic("U fucked up!");
        internal.handleCursorMove(window, xpos, ypos);
    }

    pub fn emitResizeEvent(self: *const Window) void {
        for (self.window_resize_listeners.items) |listener| {
            listener.fn_ptr(listener.instance, self.glfw_window);
        }
    }

    pub fn deinit(self: Window) void {
        self.glfw_window.destroy();
        self.event_handler[0].deinit();
        self.allocator.free(@as([]EventHandler, @ptrCast(self.event_handler)));
        self.window_resize_listeners.deinit();
    }
};
