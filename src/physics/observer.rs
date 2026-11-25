use glam::{Quat, Vec3};

use crate::input::CameraCommand;

#[derive(Debug, Clone)]
pub struct ObserverState {
    pub pos: Vec3,
    pub fwd: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub move_speed: f32,
    pub key_rot_speed: f32,
    pub mouse_sensitivity: f32,
    pub fov_y_deg: f32,
    pub hud_visible: bool,
}

impl Default for ObserverState {
    fn default() -> Self {
        Self::new_hero()
    }
}

impl ObserverState {
    pub fn new_hero() -> Self {
        let mut obs = Self {
            pos: Vec3::ZERO,
            fwd: Vec3::Y,
            right: Vec3::X,
            up: Vec3::Z,
            move_speed: 3.8,
            key_rot_speed: 1.4,
            mouse_sensitivity: 0.0025,
            fov_y_deg: 50.0,
            hud_visible: true,
        };
        obs.reset_to_hero();
        obs
    }

    pub fn reset_to_hero(&mut self) {
        self.pos = Vec3::new(0.0, -21.5, 3.2);
        let target = Vec3::new(-7.6, 0.0, -0.5);
        let fwd0 = (target - self.pos).normalize();
        let right0 = fwd0.cross(Vec3::Z).normalize();
        let up0 = right0.cross(fwd0).normalize();

        let roll_q = Quat::from_axis_angle(fwd0, 13.5_f32.to_radians());
        self.fwd = fwd0;
        self.right = (roll_q * right0).normalize();
        self.up = (roll_q * up0).normalize();
        self.fov_y_deg = 50.0;
        self.reorthonormalize();
    }

    /// Two-pass Modified Gram-Schmidt preserving right-handedness (fwd x up == right).
    pub fn reorthonormalize(&mut self) {
        self.fwd = self.fwd.normalize_or(Vec3::Y);
        // Pass 1
        self.up = (self.up - self.fwd * self.fwd.dot(self.up)).normalize_or(Vec3::Z);
        self.right = self.fwd.cross(self.up).normalize();
        self.up = self.right.cross(self.fwd).normalize();
        // Pass 2
        self.up = (self.up - self.fwd * self.fwd.dot(self.up)).normalize();
        self.right = self.fwd.cross(self.up).normalize();
        self.up = self.right.cross(self.fwd).normalize();
    }

    /// Smooth distance-adaptive speed scale (M/s multiplier relative to hero vantage r ≈ 21.5M):
    /// slows progressively as you approach the horizon & singularity for fine control,
    /// and scales up smoothly when exploring far out in deep space.
    pub fn distance_speed_scale(&self) -> f32 {
        let r = self.pos.length();
        let ratio = (r / 21.5).max(0.005);
        ratio.powf(1.15).clamp(0.015, 8.0)
    }

    pub fn effective_move_speed(&self) -> f32 {
        self.move_speed * self.distance_speed_scale()
    }

    pub fn apply_command(&mut self, cmd: &CameraCommand, dt: f32) {
        if cmd.toggle_hud_requested {
            self.hud_visible = !self.hud_visible;
        }
        if cmd.reset_requested {
            self.reset_to_hero();
            return;
        }

        let dt = dt.clamp(0.0, 0.1);

        let yaw =
            cmd.yaw_axis * self.key_rot_speed * dt + cmd.mouse_delta.x * self.mouse_sensitivity;
        let pitch =
            cmd.pitch_axis * self.key_rot_speed * dt - cmd.mouse_delta.y * self.mouse_sensitivity;
        let roll_cw = cmd.roll_axis * self.key_rot_speed * dt;

        if yaw.abs() > 1e-7 {
            let q = Quat::from_axis_angle(self.up, -yaw);
            self.fwd = q * self.fwd;
            self.right = q * self.right;
        }
        if pitch.abs() > 1e-7 {
            let q = Quat::from_axis_angle(self.right, pitch);
            self.fwd = q * self.fwd;
            self.up = q * self.up;
        }
        if roll_cw.abs() > 1e-7 {
            let q = Quat::from_axis_angle(self.fwd, roll_cw);
            self.right = q * self.right;
            self.up = q * self.up;
        }
        self.reorthonormalize();

        let move_vec =
            self.right * cmd.move_axis[0] + self.up * cmd.move_axis[1] + self.fwd * cmd.move_axis[2];
        if move_vec.length_squared() > 1e-8 && dt > 0.0 {
            let dir = if move_vec.length_squared() > 1.0 {
                move_vec.normalize()
            } else {
                move_vec
            };
            let speed = self.effective_move_speed() * cmd.speed_scale;
            self.pos += dir * (speed * dt);
        }
    }
}
