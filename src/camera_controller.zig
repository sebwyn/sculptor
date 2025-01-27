const std = @import("std");
const zlm_helpers = @import("zlm");
const zlm = @import("zlm").SpecializeOn(f32);
const events = @import("window.zig");
const Window = @import("window.zig").Window;
const glfw = @import("mach-glfw");

pub const Camera = struct {
    view_matrix: zlm.Mat4,
    proj_matrix: zlm.Mat4,
    screen_size: [4]f32,
};

pub const CameraController = struct {
    camera: Camera,

    target: zlm.Vec3,
    x_angle: f32,
    y_angle: f32,
    distance_to_target: f32,
    zooming_speed: f32 = 0.5,

    pan: ?struct {
        start_x: f64,
        start_y: f64,
        start_x_angle: f32,
        start_y_angle: f32,
    } = null,

    window_width: f32,
    window_height: f32,

    pub fn init(window: *const Window) CameraController {
        const target = zlm.Vec3.new(0.0, 0.0, 0.0);
        const distance_to_target = 20;
        const x_angle = 0.0;
        const y_angle = 0.0;

        const window_size = window.glfw_window.getFramebufferSize();
        const window_width: f32 = @floatFromInt(window_size.width);
        const window_height: f32 = @floatFromInt(window_size.height);

        const proj = zlm.Mat4.createPerspective(zlm_helpers.toRadians(90.0), window_width / window_height, 0.01, 10000);

        const camera = Camera {
            .view_matrix = zlm.Mat4.identity,
            .proj_matrix = proj,
            .screen_size = .{ window_width, window_height, 0, 0 }
        };

        var controller = CameraController { 
            .camera = camera,
            .target = target,
            .x_angle = x_angle,
            .y_angle = y_angle,
            .distance_to_target = distance_to_target,
            .window_width = window_width,
            .window_height = window_height,
        }; 
        controller.recomputeViewMatrix();

        return controller;
    }

    pub fn registerWithEvents(self: *CameraController, window: *Window) void {
        window.subscribeToEvents(events.EventHandler.Listeners {
            .scroll_listener = .{ .instance = self, .fn_ptr = @ptrCast(&CameraController.handleMouseScroll) },
            .mouse_button_listener = events.MouseButtonListener { .instance = self, .fn_ptr = @ptrCast(&CameraController.handleMouseButton) },
            .cursor_move_listener = events.CursorMoveListener { .instance = self, .fn_ptr = @ptrCast(&CameraController.handleMouseMove) },
            .window_refresh_listener = events.WindowRefreshListener { .instance = self, .fn_ptr = @ptrCast(&CameraController.handleWindowRefresh) },
        });
    }

    fn recomputeViewMatrix(self: *CameraController) void {
        const x = std.math.cos(self.y_angle);
        const z = std.math.sin(self.y_angle);
        const y = std.math.sin(self.x_angle);
        const backwards = zlm.Vec3.new(x, y, z).normalize();
        const camera_pos = self.target.add(backwards.scale(self.distance_to_target));
        const view = zlm.Mat4.createLook(camera_pos, backwards.scale(-1.0), zlm.Vec3.new(0.0, 1.0, 0.0));
        self.camera.view_matrix = view;
    }

    pub fn handleMouseScroll(self: *CameraController, _: glfw.Window, _: f64, y_offset: f64) void {
        self.distance_to_target += @as(f32, @floatCast(self.zooming_speed * y_offset));
        self.recomputeViewMatrix();
    }

    fn handleMouseButton(self: *CameraController, window: glfw.Window, button: glfw.MouseButton, action: glfw.Action, _: glfw.Mods) void {
        if (button == glfw.MouseButton.left) {
            if (action == glfw.Action.press) {
                const cursor_pos = glfw.Window.getCursorPos(window);
                self.pan = .{ 
                    .start_x = cursor_pos.xpos,
                    .start_y = cursor_pos.ypos,
                    .start_x_angle = self.x_angle,
                    .start_y_angle = self.y_angle,
                };
            } else if (action == glfw.Action.release) self.pan = null;
        }
    }

    pub fn handleMouseMove(self: *CameraController, _: glfw.Window, xpos: f64, ypos: f64) void {
        if (self.pan) |pan| {
            const y_angle_delta: f32 = @floatCast((xpos - pan.start_x) * (10*std.math.pi) / self.window_width);  
            const x_angle_delta: f32 = @floatCast((ypos - pan.start_y) * (10*std.math.pi) / self.window_height);
            self.y_angle = @mod(pan.start_y_angle + y_angle_delta, 2.0*std.math.pi);
            self.x_angle = @min(@max(pan.start_x_angle + x_angle_delta, -std.math.pi / 2.0), std.math.pi / 2.0);
            self.recomputeViewMatrix();
        }
    }

    // TODO: This function gets called even when the window size hasn't changed, only recompute if the size has actually changed
    pub fn handleWindowRefresh(self: *CameraController, window: glfw.Window) void {
        const new_window_size = window.getFramebufferSize();
        self.window_width = @floatFromInt(new_window_size.width);
        self.window_height = @floatFromInt(new_window_size.height);
        self.camera.proj_matrix = zlm.Mat4.createPerspective(zlm_helpers.toRadians(90.0), self.window_width / self.window_height, 1, 1000);
        self.camera.screen_size[0] = self.window_width;
        self.camera.screen_size[1] = self.window_height;
    }

};
