// Iluminacao por fragmento (Phong shading) com uma fonte de luz spot e
// textura decal.
// O shader trabalha no espaco da camera: "vertex" leva o vertice do espaco
// do objeto para o espaco da camera, a posicao e a direcao da spot chegam ja
// mapeadas para esse espaco e o observador fica na origem.
// vs_main so transforma posicao/normal e repassa a coordenada de textura;
// a formula (ambiente + difusa + especular), o cone da spot e a modulacao
// pela textura sao avaliados em fs_main, por pixel.

struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,   // espaco da camera -> NDC
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>, // posicao da spot no espaco da camera
  light_direction: vec3<f32>,// eixo do cone (normalizado), espaco da camera
  light_cos_cutoff: f32,     // cos do semi-angulo externo: fora dele, so ambiente
  light_cos_inner: f32,      // cos do semi-angulo interno: dentro dele, intensidade total
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,

  specular_color: vec3<f32>,
  shininess: f32,

  emission: vec3<f32>,       // brilho proprio (lampada); zero se nao definido
}
@group(2) @binding(0) var<uniform> material: Material;

// decal: a cor da textura modula a cor base do material (objetos sem
// padronagem usam uma textura branca 1x1)
@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) light: vec3<f32>,
  @location(2) view: vec3<f32>,
  @location(3) texcoord: vec2<f32>,
}

// Cube/Sphere ligam coord, normal, tangente e texcoord nos slots 0..3. A
// tangente nao e usada aqui, mas precisa ser declarada: um slot sem
// atributos no layout e descartado pelo driver e o texcoord passa a ser
// lido do buffer errado.
@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let veye = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  // nao normaliza aqui: o vetor ate a luz e interpolado sem distorcao e so
  // normalizado no fragmento
  out.light = global.light_position.xyz - veye;
  out.view = -veye;
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.texcoord = texcoord;
  out.clip_position = global.projection * vec4<f32>(veye, 1.0);
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  // textureSample precisa de controle de fluxo uniforme: amostra antes dos ifs
  let texel = textureSample(decal_texture, decal_sampler, in.texcoord);
  let base = material.base_color * texel.rgb;

  let normal = normalize(in.normal);
  let light = normalize(in.light);
  let view = normalize(in.view);

  var color = material.emission + base * global.light_ambient;
  // cone da spot: compara o angulo entre o eixo e o raio luz->fragmento com
  // os dois semi-angulos; smoothstep suaviza a borda do circulo de luz
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
