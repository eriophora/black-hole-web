// GPU verification compute shader testing curvature-adaptive RK4 Kerr-Schild geodesic accuracy:
// null Hamiltonian conservation |H(x, p)| / p0^2, Bardeen shadow edges, and horizon penetration.

#include "common/constants.wgsl"
#include "metrics/kerr_schild.wgsl"
#include "integrator/rk4_core.wgsl"

struct VerifyResult {
    max_h_err: f32,
    r_final: f32,
    escaped: u32,
    steps: u32,
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(0) @binding(1) var<storage, read> init_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> results: array<VerifyResult>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= u.resolution.x {
        return;
    }

    var x = init_rays[idx * 2u];
    var p = init_rays[idx * 2u + 1u];

    let mass = u.bh_mass;
    let spin_clamped = clamp(u.bh_spin, -0.999, 0.999);
    let a = spin_clamped * mass;
    let r_plus = ks_event_horizon(mass, spin_clamped);
    let r_minus = ks_inner_horizon(mass, spin_clamped);
    let r_inner_sink = max(1.03 * r_minus, 0.16 * mass);
    let r_escape = R_ESCAPE_TAIL * mass;

    var k1 = ks_rhs(x, p, mass, a);
    var r_curr = k1.r;
    let p0_norm2 = max(0.5 * dot(p, p), 1.0);
    var max_err = abs(ks_hamiltonian(x, p, mass, a)) / p0_norm2;
    if !(max_err >= 0.0 && max_err < 1e20) || !(r_curr >= 0.0 && r_curr < 1e20) {
        results[idx] = VerifyResult(1e30, r_curr, 0u, 0u);
        return;
    }

    var escaped: u32 = 0u;
    var steps: u32 = 0u;

    for (var step = 0u; step < 1200u; step = step + 1u) {
        let h = ks_adaptive_step_size(x, k1, u, r_plus, step, 0.0);
        let res = rk4_step(x, p, k1, h, mass, a);
        steps += 1u;

        x = res.x_next;
        p = res.p_next;
        k1 = GeodesicRhs(res.dx_next, res.dp_next, res.r_next);
        r_curr = res.r_next;

        let p_norm2 = max(0.5 * dot(p, p), 1.0);
        let h_val = abs(ks_hamiltonian(x, p, mass, a)) / p_norm2;
        if !(h_val >= 0.0 && h_val < 1e20) || !(r_curr >= 0.0 && r_curr < 1e20) {
            max_err = 1e30;
            break;
        }
        max_err = max(max_err, h_val);

        if r_curr <= r_inner_sink {
            break;
        }
        if abs(r_curr - r_plus) < 0.10 * mass && abs(k1.dx.x) > 80.0 * max(abs(p.x), 0.02) {
            break;
        }
        if r_curr >= r_escape && dot(x.yzw, k1.dx.yzw) > 0.0 {
            escaped = 1u;
            break;
        }
    }

    results[idx] = VerifyResult(max_err, r_curr, escaped, steps);
}
