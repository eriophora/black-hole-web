use std::collections::HashSet;

use glam::Vec2;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraCommand {
    /// Local translational input: x = right/left (+D/-A), y = up/down (+Q/-E), z = fwd/back (+W/-S).
    pub move_axis: [f32; 3],
    /// Continuous keyboard rotation rates in [-1, 1]:
    /// pitch (+up: Shift+W / ArrowUp, -down: Shift+S / ArrowDown),
    /// yaw (+right: Shift+D / ArrowRight, -left: Shift+A / ArrowLeft),
    /// roll_cw (+CW: Shift+E, -CCW: Shift+Q).
    pub pitch_axis: f32,
    pub yaw_axis: f32,
    pub roll_axis: f32,
    /// Mouse look delta in pixels (already gated by left_mouse_down || cursor_locked).
    pub mouse_delta: Vec2,
    /// Speed multiplier (1.0 normal, 0.25 with Ctrl fine control, 3.0 with Space boost).
    pub speed_scale: f32,
    /// Edge-triggered one-shot actions.
    pub reset_requested: bool,
    pub toggle_hud_requested: bool,
    pub toggle_cursor_lock_requested: bool,
    pub snapshot_requested: bool,
}

#[derive(Debug, Clone)]
pub struct InputState {
    pressed_keys: HashSet<KeyCode>,
    pub left_mouse_down: bool,
    pub cursor_locked: bool,
    pub window_focused: bool,
    pending_mouse_delta: Vec2,
    pending_reset: bool,
    pending_toggle_hud: bool,
    pending_toggle_cursor_lock: bool,
    pending_snapshot: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            left_mouse_down: false,
            cursor_locked: false,
            window_focused: true,
            pending_mouse_delta: Vec2::ZERO,
            pending_reset: false,
            pending_toggle_hud: false,
            pending_toggle_cursor_lock: false,
            pending_snapshot: false,
        }
    }
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_down(&self, code: KeyCode) -> bool {
        self.pressed_keys.contains(&code)
    }

    pub fn on_key(
        &mut self,
        code: KeyCode,
        state: ElementState,
        repeat: bool,
        ui_wants_keyboard: bool,
    ) {
        match state {
            ElementState::Released => {
                self.pressed_keys.remove(&code);
            }
            ElementState::Pressed => {
                if ui_wants_keyboard {
                    return;
                }
                self.pressed_keys.insert(code);
                if !repeat {
                    match code {
                        KeyCode::KeyR => self.pending_reset = true,
                        KeyCode::KeyH => self.pending_toggle_hud = true,
                        KeyCode::KeyL => self.pending_toggle_cursor_lock = true,
                        KeyCode::KeyP | KeyCode::F12 => self.pending_snapshot = true,
                        KeyCode::Escape => {
                            if self.cursor_locked {
                                self.pending_toggle_cursor_lock = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    pub fn on_mouse_button(
        &mut self,
        button: MouseButton,
        state: ElementState,
        ui_wants_pointer: bool,
    ) {
        if button == MouseButton::Left {
            match state {
                ElementState::Pressed => {
                    if !ui_wants_pointer {
                        self.left_mouse_down = true;
                    }
                }
                ElementState::Released => {
                    self.left_mouse_down = false;
                }
            }
        }
    }

    pub fn on_mouse_motion(&mut self, dx: f32, dy: f32) {
        if self.window_focused && (self.left_mouse_down || self.cursor_locked) {
            self.pending_mouse_delta += Vec2::new(dx, dy);
        }
    }

    pub fn on_focus_lost(&mut self) {
        self.pressed_keys.clear();
        self.left_mouse_down = false;
        self.cursor_locked = false;
        self.pending_mouse_delta = Vec2::ZERO;
        self.pending_reset = false;
        self.pending_toggle_hud = false;
        self.pending_toggle_cursor_lock = false;
        self.pending_snapshot = false;
        self.window_focused = false;
    }

    pub fn on_focus_gained(&mut self) {
        self.window_focused = true;
        self.pending_mouse_delta = Vec2::ZERO;
    }

    pub fn on_resize(&mut self) {
        self.pending_mouse_delta = Vec2::ZERO;
    }

    pub fn take_command(&mut self) -> CameraCommand {
        let shift =
            self.is_down(KeyCode::ShiftLeft) || self.is_down(KeyCode::ShiftRight);
        let ctrl =
            self.is_down(KeyCode::ControlLeft) || self.is_down(KeyCode::ControlRight);
        let space = self.is_down(KeyCode::Space);

        let mut move_x = 0.0_f32;
        let mut move_y = 0.0_f32;
        let mut move_z = 0.0_f32;
        if !shift {
            if self.is_down(KeyCode::KeyD) {
                move_x += 1.0;
            }
            if self.is_down(KeyCode::KeyA) {
                move_x -= 1.0;
            }
            if self.is_down(KeyCode::KeyQ) {
                move_y += 1.0;
            }
            if self.is_down(KeyCode::KeyE) {
                move_y -= 1.0;
            }
            if self.is_down(KeyCode::KeyW) {
                move_z += 1.0;
            }
            if self.is_down(KeyCode::KeyS) {
                move_z -= 1.0;
            }
        }

        let mut pitch_axis = 0.0_f32;
        let mut yaw_axis = 0.0_f32;
        let mut roll_axis = 0.0_f32;

        if (shift && self.is_down(KeyCode::KeyW)) || self.is_down(KeyCode::ArrowUp) {
            pitch_axis += 1.0;
        }
        if (shift && self.is_down(KeyCode::KeyS)) || self.is_down(KeyCode::ArrowDown) {
            pitch_axis -= 1.0;
        }
        if (shift && self.is_down(KeyCode::KeyD)) || self.is_down(KeyCode::ArrowRight) {
            yaw_axis += 1.0;
        }
        if (shift && self.is_down(KeyCode::KeyA)) || self.is_down(KeyCode::ArrowLeft) {
            yaw_axis -= 1.0;
        }
        if shift && self.is_down(KeyCode::KeyE) {
            roll_axis += 1.0;
        }
        if shift && self.is_down(KeyCode::KeyQ) {
            roll_axis -= 1.0;
        }

        let speed_scale = if ctrl {
            0.25
        } else if space {
            3.0
        } else {
            1.0
        };

        let cmd = CameraCommand {
            move_axis: [move_x, move_y, move_z],
            pitch_axis: pitch_axis.clamp(-1.0, 1.0),
            yaw_axis: yaw_axis.clamp(-1.0, 1.0),
            roll_axis: roll_axis.clamp(-1.0, 1.0),
            mouse_delta: self.pending_mouse_delta,
            speed_scale,
            reset_requested: self.pending_reset,
            toggle_hud_requested: self.pending_toggle_hud,
            toggle_cursor_lock_requested: self.pending_toggle_cursor_lock,
            snapshot_requested: self.pending_snapshot,
        };

        self.pending_mouse_delta = Vec2::ZERO;
        self.pending_reset = false;
        self.pending_toggle_hud = false;
        self.pending_toggle_cursor_lock = false;
        self.pending_snapshot = false;

        cmd
    }
}
