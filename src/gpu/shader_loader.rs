use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct LoadedShader {
    pub source: String,
}

pub struct ShaderLoader;

fn embedded_shader(rel_path: &str) -> Option<&'static str> {
    match rel_path {
        "blit.wgsl" => Some(include_str!("../../shaders/blit.wgsl")),
        "bloom_down.comp.wgsl" => Some(include_str!("../../shaders/bloom_down.comp.wgsl")),
        "bloom_up.comp.wgsl" => Some(include_str!("../../shaders/bloom_up.comp.wgsl")),
        "photon_ssaa.comp.wgsl" => Some(include_str!("../../shaders/photon_ssaa.comp.wgsl")),
        "postprocess.comp.wgsl" => Some(include_str!("../../shaders/postprocess.comp.wgsl")),
        "raytrace.comp.wgsl" => Some(include_str!("../../shaders/raytrace.comp.wgsl")),
        "common/constants.wgsl" => Some(include_str!("../../shaders/common/constants.wgsl")),
        "metrics/kerr_schild.wgsl" => Some(include_str!("../../shaders/metrics/kerr_schild.wgsl")),
        "integrator/rk4_core.wgsl" => Some(include_str!("../../shaders/integrator/rk4_core.wgsl")),
        "integrator/rk4_adaptive.wgsl" => {
            Some(include_str!("../../shaders/integrator/rk4_adaptive.wgsl"))
        }
        "astrophysics/spectral_grrt.wgsl" => {
            Some(include_str!("../../shaders/astrophysics/spectral_grrt.wgsl"))
        }
        "astrophysics/disk_gargantua.wgsl" => {
            Some(include_str!("../../shaders/astrophysics/disk_gargantua.wgsl"))
        }
        "astrophysics/planet.wgsl" => Some(include_str!("../../shaders/astrophysics/planet.wgsl")),
        "astrophysics/sky_procedural.wgsl" => {
            Some(include_str!("../../shaders/astrophysics/sky_procedural.wgsl"))
        }
        _ => None,
    }
}

impl ShaderLoader {
    /// Expands compile-time embedded WGSL shaders, resolving `#include "subpath.wgsl"`
    /// directives with cycle detection and diamond deduplication.
    pub fn load_embedded(entry_rel: &str) -> Result<LoadedShader, String> {
        let mut visiting = Vec::new();
        let mut visited = HashSet::new();
        let mut output = String::with_capacity(24 * 1024);

        process_embedded(entry_rel, None, &mut visiting, &mut visited, &mut output)?;

        Ok(LoadedShader { source: output })
    }
}

fn process_embedded(
    rel_path: &str,
    included_from: Option<(&str, usize)>,
    visiting: &mut Vec<String>,
    visited: &mut HashSet<String>,
    output: &mut String,
) -> Result<(), String> {
    let norm = rel_path.replace('\\', "/");
    let contents = embedded_shader(&norm).ok_or_else(|| match included_from {
        Some((parent, line_num)) => format!(
            "Missing embedded shader include '{}' referenced at {}:{}",
            norm, parent, line_num
        ),
        None => format!("Missing embedded shader '{}'", norm),
    })?;

    if visiting.contains(&norm) {
        let mut chain = visiting.clone();
        chain.push(norm);
        return Err(format!("Shader #include cycle detected: {}", chain.join(" -> ")));
    }
    if visited.contains(&norm) {
        return Ok(());
    }

    visiting.push(norm.clone());
    for (line_idx, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#include") {
            let arg = rest.trim();
            if !(arg.starts_with('"') && arg.ends_with('"') && arg.len() >= 2) {
                return Err(format!(
                    "Malformed #include directive at {}:{}: '{}'",
                    norm,
                    line_idx + 1,
                    trimmed
                ));
            }
            let inc_rel = &arg[1..arg.len() - 1];
            process_embedded(inc_rel, Some((&norm, line_idx + 1)), visiting, visited, output)?;
            output.push('\n');
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }
    visiting.pop();
    visited.insert(norm);
    Ok(())
}
