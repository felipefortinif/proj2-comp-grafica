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

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) light: vec3<f32>,
  @location(2) view: vec3<f32>,
  @location(3) texcoord: vec2<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let veye = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  out.light = global.light_position.xyz - veye;
  out.view = -veye;
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.texcoord = texcoord;
  out.clip_position = global.projection * vec4<f32>(veye, 1.0);
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  let texel = textureSample(decal_texture, decal_sampler, in.texcoord);
  let base = material.base_color * texel.rgb;

  let normal = normalize(in.normal);
  let light = normalize(in.light);
  let view = normalize(in.view);

  var color = material.emission + base * global.light_ambient;
  let cos_angle = dot(-light, normalize(global.light_direction));
  let spot = smoothstep(global.light_cos_cutoff, global.light_cos_inner, cos_angle);
  let ndotl = dot(normal, light);
  if (spot > 0.0 && ndotl > 0.0) {
    color += spot * base * global.light_diffuse * ndotl;
    let refl = reflect(-light, normal);
    color += spot * material.specular_color * global.light_specular * pow(max(0.0, dot(refl, view)), material.shininess);
  }
  return vec4<f32>(color, material.opacity * texel.a);
}
