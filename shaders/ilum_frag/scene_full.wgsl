// Iluminacao por fragmento (Phong shading) com duas fontes de luz (uma spot
// que tambem pode virar projetor de slide e uma pontual), textura decal,
// normal mapping, plano de corte e fog. Cada efeito tem uma chave em
// "global" (enable_*), alternada pelo teclado.
// O shader trabalha no espaco da camera: "vertex" leva o vertice do espaco
// do objeto para o espaco da camera, as posicoes e direcoes das luzes
// chegam ja mapeadas para esse espaco e o observador fica na origem.
// vs_main so transforma posicao/normal/tangente e repassa os dados
// interpolados; todas as contas de luz sao feitas em fs_main, por pixel.

struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,   // espaco da camera -> NDC

  // luz 1: spot da luminaria (no modo slide vira projetor)
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>, // posicao da spot no espaco da camera
  light_direction: vec3<f32>,// eixo do cone (normalizado), espaco da camera
  light_cos_cutoff: f32,     // cos do semi-angulo externo: fora dele, so ambiente
  light_cos_inner: f32,      // cos do semi-angulo interno: dentro dele, intensidade total
  slide_matrix: mat4x4<f32>, // espaco da camera -> clip do projetor

  // luz 2: pontual de teto
  light2_diffuse: vec3<f32>,
  light2_specular: vec3<f32>,
  light2_position: vec4<f32>,
  light2_attenuation: vec3<f32>, // (constante, linear, quadratica)

  // fog
  fog_color: vec3<f32>,
  fog_density: f32,

  // plano de corte, ja no espaco da camera
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,

  // chaves dos efeitos (0/1)
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

  emission: vec3<f32>,       // brilho proprio (lampada); zero se nao definido
}
@group(2) @binding(0) var<uniform> material: Material;

// decal: a cor da textura modula a cor base do material (objetos sem
// padronagem usam uma textura branca 1x1)
@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;
@group(3) @binding(2) var normal_texture: texture_2d<f32>;   // normal map (linear)
@group(3) @binding(3) var slide_texture: texture_2d<f32>;    // slide do projetor
@group(3) @binding(4) var slide_sampler: sampler;            // clamp-to-edge

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) light: vec3<f32>,    // ate a luz 1 (nao normalizado)
  @location(3) light2: vec3<f32>,   // ate a luz 2 (nao normalizado)
  @location(4) view: vec3<f32>,     // -veye
  @location(5) texcoord: vec2<f32>,
  @location(6) slide: vec4<f32>,    // posicao no clip do projetor
}

// Cube/Sphere ligam coord, normal, tangente e texcoord nos slots 0..3; os
// quatro precisam ser declarados: um slot sem atributos no layout e
// descartado pelo driver e o texcoord passa a ser lido do buffer errado.
@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let veye = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  // nao normaliza aqui: o vetor ate a luz e interpolado sem distorcao e so
  // normalizado no fragmento
  out.light = global.light_position.xyz - veye;
  out.light2 = global.light2_position.xyz - veye;
  out.view = -veye;
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  // tangente e vetor de superficie: transforma pela model-view, nao pela normal matrix
  out.tangent = (m.vertex * vec4<f32>(tangent, 0.0)).xyz;
  out.texcoord = texcoord;
  // interpolar em clip e dividir no fragmento garante a projecao perspectiva correta
  out.slide = global.slide_matrix * vec4<f32>(veye, 1.0);
  out.clip_position = global.projection * vec4<f32>(veye, 1.0);
  return out;
}

// fog exponencial quadratico pela distancia ao olho (camera na origem)
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
  // textureSample precisa de controle de fluxo uniforme: amostra tudo antes
  // de qualquer discard ou if que dependa do fragmento
  let texel = textureSample(decal_texture, decal_sampler, in.texcoord);
  let nmap = textureSample(normal_texture, decal_sampler, in.texcoord);
  let sw = max(in.slide.w, 1e-5);
  let sndc = in.slide.xy / sw;
  // NDC y para cima, t da textura para baixo: inverte o y
  let suv = vec2<f32>(sndc.x * 0.5 + 0.5, 0.5 - sndc.y * 0.5);
  let slide_texel = textureSample(slide_texture, slide_sampler, suv);
  // fora do frustum (ou atras do projetor) o slide nao ilumina
  let inside = in.slide.w > 0.0 && all(suv >= vec2<f32>(0.0)) && all(suv <= vec2<f32>(1.0));

  // plano de corte: descarta o lado negativo; vendo o objeto por dentro
  // (face de tras), pinta a secao de corte com cor solida
  if (global.enable_clip != 0) {
    let p = vec4<f32>(-in.view, 1.0);   // posicao do fragmento no espaco da camera
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
    // Gram-Schmidt: a tangente interpolada pode deixar de ser ortogonal a normal
    let t = normalize(in.tangent - normal * dot(normal, in.tangent));
    let b = cross(normal, t);
    let n_ts = nmap.xyz * 2.0 - 1.0;
    normal = normalize(t * n_ts.x + b * n_ts.y + normal * n_ts.z);
  }
  let light = normalize(in.light);
  let view = normalize(in.view);

  var color = material.emission + base * global.light_ambient;

  // luz 1: cone da spot (smoothstep entre os dois semi-angulos suaviza a
  // borda) ou, no modo slide, mascara do frustum do projetor com a cor do slide
  var l1_color = global.light_diffuse;
  var l1_amount = 0.0;
  var l1_spec = global.light_specular;
  if (global.enable_slide != 0) {
    l1_amount = select(0.0, 1.0, inside);
    l1_color = global.light_diffuse * slide_texel.rgb;
    // o brilho especular nao deve lavar a imagem projetada
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

  // luz 2: pontual com atenuacao
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
