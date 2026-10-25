struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,

  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,
  light_direction: vec3<f32>,
  light_cos_cutoff: f32,
  light_cos_inner: f32,
  slide_matrix: mat4x4<f32>,

  light2_diffuse: vec3<f32>,
  light2_specular: vec3<f32>,
  light2_position: vec4<f32>,
  light2_attenuation: vec3<f32>,

  fog_color: vec3<f32>,
  fog_density: f32,

  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,

  enable_fog: i32,
  enable_clip: i32,
  enable_light2: i32,
  enable_bump: i32,
  enable_slide: i32,
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,

  specular_color: vec3<f32>,
  shininess: f32,

  emission: vec3<f32>,
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;
@group(3) @binding(2) var normal_texture: texture_2d<f32>;
@group(3) @binding(3) var slide_texture: texture_2d<f32>;
@group(3) @binding(4) var slide_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) light: vec3<f32>,
  @location(3) light2: vec3<f32>,
  @location(4) view: vec3<f32>,
  @location(5) texcoord: vec2<f32>,
  @location(6) slide: vec4<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let veye = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  out.light = global.light_position.xyz - veye;
  out.light2 = global.light2_position.xyz - veye;
  out.view = -veye;
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.tangent = (m.vertex * vec4<f32>(tangent, 0.0)).xyz;
  out.texcoord = texcoord;
  out.slide = global.slide_matrix * vec4<f32>(veye, 1.0);
  out.clip_position = global.projection * vec4<f32>(veye, 1.0);
  return out;
}

fn apply_fog (color: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
  if (global.enable_fog == 0) {
    return color;
  }
  let dist = length(view);
  let f = clamp(exp(-pow(global.fog_density * dist, 2.0)), 0.0, 1.0);
  return mix(global.fog_color, color, f);
}

@fragment
fn fs_main (in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
  let texel = textureSample(decal_texture, decal_sampler, in.texcoord);
  let nmap = textureSample(normal_texture, decal_sampler, in.texcoord);
  let sw = max(in.slide.w, 1e-5);
  let sndc = in.slide.xy / sw;
  let suv = vec2<f32>(sndc.x * 0.5 + 0.5, 0.5 - sndc.y * 0.5);
  let slide_texel = textureSample(slide_texture, slide_sampler, suv);
  let inside = in.slide.w > 0.0 && all(suv >= vec2<f32>(0.0)) && all(suv <= vec2<f32>(1.0));

  if (global.enable_clip != 0) {
    let p = vec4<f32>(-in.view, 1.0);
    if (dot(global.clip_plane, p) < 0.0) {
      discard;
    }
    if (!front) {
      return vec4<f32>(apply_fog(global.clip_plane_color.rgb, in.view), 1.0);
    }
  }

  let base = material.base_color * texel.rgb;

  var normal = normalize(in.normal);
  if (global.enable_bump != 0 && length(in.tangent) > 1e-4) {
    let t = normalize(in.tangent - normal * dot(normal, in.tangent));
    let b = cross(normal, t);
    let n_ts = nmap.xyz * 2.0 - 1.0;
    normal = normalize(t * n_ts.x + b * n_ts.y + normal * n_ts.z);
  }
  let light = normalize(in.light);
  let view = normalize(in.view);

  var color = material.emission + base * global.light_ambient;

  var l1_color = global.light_diffuse;
  var l1_amount = 0.0;
  var l1_spec = global.light_specular;
  if (global.enable_slide != 0) {
    l1_amount = select(0.0, 1.0, inside);
    l1_color = global.light_diffuse * slide_texel.rgb;
    l1_spec = global.light_specular * 0.25;
  } else {
    let cos_angle = dot(-light, normalize(global.light_direction));
    l1_amount = smoothstep(global.light_cos_cutoff, global.light_cos_inner, cos_angle);
  }
  let ndotl = dot(normal, light);
  if (l1_amount > 0.0 && ndotl > 0.0) {
    color += l1_amount * base * l1_color * ndotl;
    let refl = reflect(-light, normal);
    color += l1_amount * material.specular_color * l1_spec * pow(max(0.0, dot(refl, view)), material.shininess);
  }

  if (global.enable_light2 != 0) {
    let d = length(in.light2);
    let l2 = in.light2 / d;
    let att = 1.0 / (global.light2_attenuation.x + global.light2_attenuation.y * d + global.light2_attenuation.z * d * d);
    let ndotl2 = dot(normal, l2);
    if (ndotl2 > 0.0) {
      color += att * base * global.light2_diffuse * ndotl2;
      let r2 = reflect(-l2, normal);
      color += att * material.specular_color * global.light2_specular * pow(max(0.0, dot(r2, view)), material.shininess);
    }
  }

  color = apply_fog(color, in.view);
  return vec4<f32>(color, material.opacity * texel.a);
}
