use glam::{DMat4, DQuat, DVec3, DVec4};

/// Horizon-penetrating Ingoing Cartesian Kerr-Schild (CKS) spacetime metric
/// with spin aligned along +z and coordinates x^mu = (t, x, y, z), signature (-, +, +, +).
#[derive(Debug, Clone, Copy)]
pub struct KerrSchild {
    pub mass: f64,
    pub spin: f64,
}

impl KerrSchild {
    pub fn new(mass: f64, a_star: f64) -> Self {
        let clamped_a_star = a_star.clamp(-0.999, 0.999);
        Self {
            mass,
            spin: clamped_a_star * mass,
        }
    }

    #[inline]
    pub fn a_star(&self) -> f64 {
        if self.mass > 0.0 {
            self.spin / self.mass
        } else {
            0.0
        }
    }

    #[inline]
    pub fn event_horizon_radius(&self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin * self.spin).max(0.0).sqrt()
    }

    #[inline]
    pub fn inner_horizon_radius(&self) -> f64 {
        self.mass - (self.mass * self.mass - self.spin * self.spin).max(0.0).sqrt()
    }

    #[inline]
    pub fn ergosphere_radius(&self, x: DVec4) -> f64 {
        let r = self.kerr_radius(DVec3::new(x.y, x.z, x.w));
        let cos_theta = if r > 1e-12 { x.w / r } else { 0.0 };
        self.mass
            + (self.mass * self.mass - self.spin * self.spin * cos_theta * cos_theta)
                .max(0.0)
                .sqrt()
    }

    /// Cancellation-free Kerr Boyer-Lindquist radial coordinate r(x, y, z).
    #[inline]
    pub fn kerr_radius(&self, pos: DVec3) -> f64 {
        let a = self.spin;
        let a2 = a * a;
        let r_euclid2 = pos.length_squared();
        let b = r_euclid2 - a2;
        let disc = (b * b + 4.0 * a2 * pos.z * pos.z).sqrt();
        let r2 = if b >= 0.0 {
            0.5 * (b + disc)
        } else {
            (2.0 * a2 * pos.z * pos.z) / (disc - b).max(1e-24)
        };
        r2.max(1e-16).sqrt()
    }

    /// Prograde or retrograde equatorial ISCO radius in geometric units.
    /// Positive spin -> prograde orbit in +phi direction; negative spin -> retrograde.
    pub fn isco_radius(&self) -> f64 {
        let a_s = self.a_star().clamp(-0.999, 0.999);
        let z1 = 1.0
            + (1.0 - a_s * a_s).cbrt() * ((1.0 + a_s).cbrt() + (1.0 - a_s).cbrt());
        let z2 = (3.0 * a_s * a_s + z1 * z1).sqrt();
        let sign = if a_s >= 0.0 { -1.0 } else { 1.0 };
        self.mass * (3.0 + z2 + sign * ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt())
    }

    /// Conserved specific energy E_ISCO and angular momentum L_ISCO for an equatorial
    /// circular orbit at r = r_ISCO.
    pub fn isco_energy_and_ang_mom(&self) -> (f64, f64) {
        let m = self.mass;
        let a = self.spin;
        let r = self.isco_radius();
        let rt_m = m.sqrt();
        let rt_r = r.sqrt();
        let denom = r.powf(0.75)
            * (r * rt_r - 3.0 * m * rt_r + 2.0 * a * rt_m)
                .max(1e-12)
                .sqrt();
        let e = (r * rt_r - 2.0 * m * rt_r + a * rt_m) / denom;
        let l = (rt_m * (r * r - 2.0 * a * rt_m * rt_r + a * a)) / denom;
        (e, l)
    }

    /// Equatorial prograde or retrograde circular photon orbit radius r_ph.
    pub fn photon_orbit_radius(&self, prograde: bool) -> f64 {
        let a_abs = self.a_star().abs().clamp(0.0, 0.999);
        let arg = if prograde { -a_abs } else { a_abs };
        2.0 * self.mass * (1.0 + ((2.0 / 3.0) * arg.acos()).cos())
    }

    /// Bardeen (1973) critical equatorial impact parameter xi_c / M for dimensionless spin `a_star`.
    /// Returns +3*sqrt(3) (prograde) or -3*sqrt(3) (retrograde) in the Schwarzschild limit.
    pub fn bardeen_critical_xi(a_star: f64, prograde: bool) -> f64 {
        let a = a_star.abs().clamp(0.0, 0.999);
        if a < 1e-6 {
            return if prograde {
                3.0 * 3.0_f64.sqrt()
            } else {
                -3.0 * 3.0_f64.sqrt()
            };
        }
        let sign = if prograde { -1.0 } else { 1.0 };
        let r_ph = 2.0 * (1.0 + ((2.0 / 3.0) * (sign * a).acos()).cos());
        -(r_ph * r_ph * r_ph - 3.0 * r_ph * r_ph + a * a * r_ph + a * a) / (a * (r_ph - 1.0))
    }

    /// Returns `(b_prograde_mag, b_retrograde_mag)` in units of `mass`.
    pub fn bardeen_shadow_edges(mass: f64, a_star: f64) -> (f64, f64) {
        let pro = mass * Self::bardeen_critical_xi(a_star, true).abs();
        let ret = mass * Self::bardeen_critical_xi(a_star, false).abs();
        (pro, ret)
    }

    /// Evaluates `(r, f, l_mu)` where `l_mu = (1, l_x, l_y, l_z)` is the principal ingoing null covector.
    #[inline]
    pub fn f_and_l_cov(&self, x: DVec4) -> (f64, f64, DVec4) {
        let pos = DVec3::new(x.y, x.z, x.w);
        let r = self.kerr_radius(pos);
        let r2 = r * r;
        let a = self.spin;
        let a2 = a * a;
        let sigma_ks = (r2 * r2 + a2 * pos.z * pos.z).max(1e-24);
        let denom_a = r2 + a2;
        let f = (2.0 * self.mass * r2 * r) / sigma_ks;
        let lx = (r * pos.x + a * pos.y) / denom_a;
        let ly = (r * pos.y - a * pos.x) / denom_a;
        let lz = pos.z / r;
        (r, f, DVec4::new(1.0, lx, ly, lz))
    }

    #[inline]
    fn spatial_derivatives(&self, x: DVec4) -> (f64, f64, DVec4, DVec3, [DVec3; 3]) {
        let pos = DVec3::new(x.y, x.z, x.w);
        let r = self.kerr_radius(pos);
        let r2 = r * r;
        let r3 = r2 * r;
        let r4 = r2 * r2;
        let a = self.spin;
        let a2 = a * a;
        let sigma_ks = (r4 + a2 * pos.z * pos.z).max(1e-24);
        let denom_a = r2 + a2;
        let f = (2.0 * self.mass * r3) / sigma_ks;

        let lx = (r * pos.x + a * pos.y) / denom_a;
        let ly = (r * pos.y - a * pos.x) / denom_a;
        let lz = pos.z / r;
        let l_cov = DVec4::new(1.0, lx, ly, lz);

        let dr = DVec3::new(
            (r3 * pos.x) / sigma_ks,
            (r3 * pos.y) / sigma_ks,
            (r * pos.z * denom_a) / sigma_ks,
        );

        let df_coeff = (2.0 * self.mass * r2) / (sigma_ks * sigma_ks);
        let poly = 3.0 * a2 * pos.z * pos.z - r4;
        let df = DVec3::new(
            df_coeff * poly * dr.x,
            df_coeff * poly * dr.y,
            df_coeff * (poly * dr.z - 2.0 * a2 * r * pos.z),
        );

        let dl_dx = DVec3::new(
            (r + (pos.x - 2.0 * r * lx) * dr.x) / denom_a,
            (-a + (pos.y - 2.0 * r * ly) * dr.x) / denom_a,
            (-lz * dr.x) / r,
        );
        let dl_dy = DVec3::new(
            (a + (pos.x - 2.0 * r * lx) * dr.y) / denom_a,
            (r + (pos.y - 2.0 * r * ly) * dr.y) / denom_a,
            (-lz * dr.y) / r,
        );
        let dl_dz = DVec3::new(
            ((pos.x - 2.0 * r * lx) * dr.z) / denom_a,
            ((pos.y - 2.0 * r * ly) * dr.z) / denom_a,
            (1.0 - lz * dr.z) / r,
        );

        (r, f, l_cov, df, [dl_dx, dl_dy, dl_dz])
    }

    pub fn metric_cov(&self, x: DVec4) -> DMat4 {
        let (_, f, l) = self.f_and_l_cov(x);
        let l_arr = [l.x, l.y, l.z, l.w];
        let eta = [-1.0, 1.0, 1.0, 1.0];
        let mut cols = [[0.0; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                let diag = if row == col { eta[row] } else { 0.0 };
                cols[col][row] = diag + f * l_arr[row] * l_arr[col];
            }
        }
        DMat4::from_cols_array_2d(&cols)
    }

    pub fn metric_inv(&self, x: DVec4) -> DMat4 {
        let (_, f, l_cov) = self.f_and_l_cov(x);
        let l_con = [-l_cov.x, l_cov.y, l_cov.z, l_cov.w];
        let eta = [-1.0, 1.0, 1.0, 1.0];
        let mut cols = [[0.0; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                let diag = if row == col { eta[row] } else { 0.0 };
                cols[col][row] = diag - f * l_con[row] * l_con[col];
            }
        }
        DMat4::from_cols_array_2d(&cols)
    }

    pub fn hamiltonian(&self, x: DVec4, p: DVec4) -> f64 {
        let g_inv = self.metric_inv(x);
        0.5 * contract_inv(&g_inv, p, p)
    }

    pub fn d_hamiltonian_dx(&self, x: DVec4, p: DVec4) -> DVec4 {
        let (_, f, l_cov, df, dl) = self.spatial_derivatives(x);
        let l_con = DVec4::new(-l_cov.x, l_cov.y, l_cov.z, l_cov.w);
        let s = l_con.dot(p);
        let p_sp = DVec3::new(p.y, p.z, p.w);

        let dh_dx = -(0.5 * s * s * df.x + f * s * p_sp.dot(dl[0]));
        let dh_dy = -(0.5 * s * s * df.y + f * s * p_sp.dot(dl[1]));
        let dh_dz = -(0.5 * s * s * df.z + f * s * p_sp.dot(dl[2]));
        DVec4::new(0.0, dh_dx, dh_dy, dh_dz)
    }

    pub fn eulerian_frame(&self, x: DVec4, cam_rot: DQuat) -> [DVec4; 4] {
        let (_, f, l_cov) = self.f_and_l_cov(x);
        let w = (1.0 + f).sqrt();
        let inv_w = 1.0 / w;
        let l_sp = DVec3::new(l_cov.y, l_cov.z, l_cov.w);

        let e0 = DVec4::new(
            w,
            -f * inv_w * l_sp.x,
            -f * inv_w * l_sp.y,
            -f * inv_w * l_sp.z,
        );

        let q1 = cam_rot * DVec3::X;
        let q2 = cam_rot * DVec3::Y;
        let q3 = cam_rot * DVec3::Z;

        let map_spatial = |q: DVec3| -> DVec4 {
            let v = q + (inv_w - 1.0) * l_sp.dot(q) * l_sp;
            DVec4::new(0.0, v.x, v.y, v.z)
        };

        [e0, map_spatial(q1), map_spatial(q2), map_spatial(q3)]
    }

    fn zamo_omega(&self, x: DVec4, r: f64, f: f64) -> f64 {
        let a = self.spin;
        let a2 = a * a;
        let r2_plus_a2 = r * r + a2;
        let rho2 = x.y * x.y + x.z * x.z;
        let denom = r2_plus_a2 + (f * a2 * rho2) / r2_plus_a2.max(1e-24);
        (f * a) / denom.max(1e-24)
    }

    pub fn zamo_four_velocity(&self, x: DVec4) -> Option<DVec4> {
        let r_plus = self.event_horizon_radius();
        let (r, f, _) = self.f_and_l_cov(x);
        if r <= 1.0005 * r_plus {
            return None;
        }
        let g = self.metric_cov(x);
        let omega_zamo = self.zamo_omega(x, r, f);
        let u_tilde = DVec4::new(1.0, -omega_zamo * x.z, omega_zamo * x.y, 0.0);
        let norm2 = -contract_cov(&g, u_tilde, u_tilde);
        if norm2 > 1e-12 {
            Some(u_tilde / norm2.sqrt())
        } else {
            None
        }
    }

    pub fn hover_four_velocity(&self, x: DVec4) -> Option<DVec4> {
        let r_plus = self.event_horizon_radius();
        let (r, f, _) = self.f_and_l_cov(x);
        if r <= 1.002 * r_plus {
            return None;
        }
        let g = self.metric_cov(x);
        let xi_t = DVec4::new(1.0, 0.0, 0.0, 0.0);
        let xi_phi = DVec4::new(0.0, -x.z, x.y, 0.0);
        let omega_zamo = self.zamo_omega(x, r, f);
        let blend = smoothstep(0.15, 0.0, 1.0 - f);
        let omega_hover = blend * omega_zamo;

        let u_tilde = xi_t + omega_hover * xi_phi;
        let norm2 = -contract_cov(&g, u_tilde, u_tilde);
        if norm2 > 1e-10 {
            Some(u_tilde / norm2.sqrt())
        } else {
            self.zamo_four_velocity(x)
        }
    }
}

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn contract_cov(g: &DMat4, a: DVec4, b: DVec4) -> f64 {
    a.dot(*g * b)
}

#[inline]
pub fn contract_inv(g_inv: &DMat4, p: DVec4, q: DVec4) -> f64 {
    p.dot(*g_inv * q)
}

pub fn gram_schmidt_lorentz(g: &DMat4, tetrad: &mut [DVec4; 4]) {
    let norm0 = (-contract_cov(g, tetrad[0], tetrad[0])).max(1e-18).sqrt();
    tetrad[0] /= norm0;

    for i in 1..4 {
        let mut v = tetrad[i];
        let proj0 = contract_cov(g, tetrad[0], v);
        v += proj0 * tetrad[0];
        for j in 1..i {
            let proj_j = contract_cov(g, tetrad[j], v);
            v -= proj_j * tetrad[j];
        }
        let proj0_2 = contract_cov(g, tetrad[0], v);
        v += proj0_2 * tetrad[0];
        for j in 1..i {
            let proj_j_2 = contract_cov(g, tetrad[j], v);
            v -= proj_j_2 * tetrad[j];
        }
        let norm_i = contract_cov(g, v, v).max(1e-18).sqrt();
        tetrad[i] = v / norm_i;
    }
}
