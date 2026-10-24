from __future__ import annotations

# Cena 3D: a mesa da Tarefa 2.1 (cubos) com objetos sobre o tampo - uma
# caixa de presente com uma esfera verde em cima, um globo terrestre num
# pedestal e uma luminaria de mesa. A luminaria carrega uma luz spot, presa
# ao no da cupula (luz em espaco local): mexer na luminaria mexe na luz.
# Iluminacao por fragmento; tampo/pernas, caixa e globo com textura decal;
# camera com arcball.
#
# Controles:
#   arrastar com o botao do mouse -> gira a camera (arcball)
#   roda do mouse                 -> zoom
#   setas esquerda/direita        -> gira o braco da luminaria
#   setas cima/baixo              -> inclina a cupula (direcao da spot)
#   = / -                         -> abre/fecha o cone da spot
#   r                             -> volta a vista inicial
#   q                             -> sai

import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ha um pacote "node" de terceiros pre-carregado no startup que sombreia o
# node.py do framework
sys.modules.pop("node", None)

import graphics_math as gm
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera3d import *
from spotlight import *
from shader import *
from pipeline import *
from phongmaterial import *
from texture import *
from sampler import *
from textureset import *
from transform import *
from node import *
from scene import *
from renderer import *
from cube import *
from sphere import *

SHADER_PATH = os.path.join(HERE, "..", "shaders", "ilum_frag", "spot_textured.wgsl")
IMAGES = os.path.join(HERE, "..", "images")

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera3D
scene: Scene
spot: SpotLight

# dimensoes da mesa (unidades de mundo)
TABLE_W = 3.0       # largura (x)
TABLE_D = 2.0       # profundidade (z)
TABLE_H = 1.0       # altura das pernas
TOP_T = 0.2         # espessura do tampo
LEG_T = 0.2         # espessura das pernas
TOP_Y = TABLE_H + TOP_T   # altura da superficie do tampo

# luminaria: haste vertical, braco horizontal e cupula na ponta do braco
LAMP_X, LAMP_Z = 1.2, -0.75   # posicao da base sobre o tampo
POST_H = 1.25                 # altura da haste
ARM_L = 1.0                   # comprimento do braco
INNER_RATIO = 0.7             # semi-angulo interno / semi-angulo de corte

# estado da luminaria, alterado pelo teclado
lamp_yaw = -120.0     # giro do braco em torno da haste (graus)
lamp_tilt = 10.0      # inclinacao da cupula em relacao a vertical (graus)
arm_trf: Transform
head_trf: Transform

def make_material (r: float, g: float, b: float, spec: float, shininess: float) -> PhongMaterial:
  mat = PhongMaterial(r, g, b)
  mat.set_specular(spec, spec, spec)
  mat.set_shininess(shininess)
  return mat

def box (x: float, y: float, z: float, sx: float, sy: float, sz: float) -> Transform:
  """Transformacao de um cubo unitario (x,z em [-0.5,0.5], y em [0,1]) para
  uma caixa de dimensoes (sx,sy,sz) com o centro da base em (x,y,z)."""
  trf = Transform()
  trf.translate(x, y, z)
  trf.scale(sx, sy, sz)
  return trf

def sphere_at (x: float, y: float, z: float, r: float) -> Transform:
  """Transformacao da esfera unitaria para uma esfera de raio r com centro em (x,y,z)."""
  trf = Transform()
  trf.translate(x, y, z)
  trf.scale(r, r, r)
  return trf

def update_lamp () -> None:
  """Refaz as transformacoes articuladas da luminaria a partir de
  lamp_yaw/lamp_tilt; a spot acompanha porque vive no espaco da cupula."""
  arm_trf.load_identity()
  arm_trf.translate(0.0, POST_H, 0.0)
  arm_trf.rotate(lamp_yaw, 0.0, 1.0, 0.0)
  head_trf.load_identity()
  head_trf.translate(ARM_L, 0.0, 0.0)
  head_trf.rotate(lamp_tilt, 0.0, 0.0, 1.0)

def build_scene (device: wgpu.GPUDevice, target_format: str) -> tuple[Camera3D, Scene, SpotLight]:
  """Cria camera, luz, materiais, texturas e o grafo de cena."""
  global arm_trf, head_trf

  cam = Camera3D(2.4, 3.4, 4.6)
  cam.set_center(0.0, TOP_Y, 0.0)

  # spot em espaco local: posicao/direcao dadas no sistema da cupula (luz
  # logo abaixo dela, apontando para baixo); o framework compoe a matriz de
  # modelo da cupula com a de vista a cada quadro
  light = SpotLight(0.0, -0.07, 0.0, 0.0, -1.0, 0.0,
                    cutoff=40.0, inner=40.0*INNER_RATIO, space="local")
  light.set_ambient(0.25, 0.25, 0.25)
  light.set_diffuse(1.0, 0.95, 0.85)
  light.set_specular(1.0, 1.0, 1.0)

  shader = Shader(device, SHADER_PATH, light=light, space="camera")
  # Cube/Sphere ligam coord, normal, tangente e texcoord nos slots 0..3; a
  # tangente nao e usada pelo shader, mas o slot 2 precisa de um atributo
  # (ver comentario em spot_textured.wgsl)
  shader.set_vertex_buffers([
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "tangent"}]},
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
  ])
  pipeline = Pipeline(shader, target_format, primitive={"cull_mode": "back"})

  # objetos texturizados usam cor base branca: a cor vem do decal
  wood = make_material(1.0, 1.0, 1.0, 0.25, 16.0)
  gift = make_material(1.0, 1.0, 1.0, 0.3, 32.0)
  globe = make_material(1.0, 1.0, 1.0, 0.6, 48.0)
  green = make_material(0.35, 0.75, 0.30, 0.8, 64.0)
  pedestal = make_material(0.12, 0.12, 0.14, 0.9, 96.0)
  lamp_body = make_material(0.80, 0.15, 0.12, 0.9, 96.0)
  bulb = make_material(0.0, 0.0, 0.0, 0.0, 1.0)
  bulb.set("emission", gm.vec3(1.0, 0.95, 0.8))
  for mat in (wood, gift, globe, green, pedestal, lamp_body, bulb):
    shader.add_material(mat)

  sampler = Sampler(device, "decal_sampler")
  def decal (filename: str) -> TextureSet:
    ts = TextureSet([Texture(device, "decal_texture", os.path.join(IMAGES, filename), mipmaps=True), sampler])
    shader.add_texture_set(ts)
    return ts
  # sem padronagem: textura branca 1x1, que deixa a cor do material intacta
  plain = TextureSet([Texture(device, "decal_texture", None, texel=gm.vec3(1.0, 1.0, 1.0)), sampler])
  shader.add_texture_set(plain)
  wood_tex = decal("wood.png")
  gift_tex = decal("giftwrap.png")
  earth_tex = decal("earth.jpg")

  cube = Cube(device)
  sphere = Sphere(device)

  # mesa: tampo + 4 pernas, todos cubos escalados, com o decal de madeira
  lx = TABLE_W/2 - LEG_T/2
  lz = TABLE_D/2 - LEG_T/2
  legs = [
    Node(None, box(sx*lx, 0.0, sz*lz, LEG_T, TABLE_H, LEG_T), None, [cube])
    for sx in (-1, 1) for sz in (-1, 1)
  ]
  table = Node(None, None, [wood, wood_tex],
               nodes = [Node(None, box(0.0, TABLE_H, 0.0, TABLE_W, TOP_T, TABLE_D), None, [cube])] + legs)

  # caixa de presente sobre o tampo, a frente a direita, com a esfera verde
  # apoiada nela; o grupo so translada, entao a esfera nao herda a escala
  # (nem o decal) da caixa
  BOX_W, BOX_H, BOX_D = 0.9, 0.35, 0.7
  GREEN_R = 0.22
  trf_group = Transform()
  trf_group.translate(0.6, TOP_Y, 0.3)
  box_group = Node(None, trf_group,
                   nodes = [
                     Node(None, box(0.0, 0.0, 0.0, BOX_W, BOX_H, BOX_D), [gift, gift_tex], [cube]),
                     Node(None, sphere_at(0.15, BOX_H + GREEN_R, 0.0, GREEN_R), [green], [sphere]),
                   ])

  # globo ao fundo a esquerda: pedestal (cubo) e esfera com o mapa da Terra,
  # com o eixo inclinado como num globo de verdade
  GLOBE_R = 0.38
  PED_H = 0.08
  trf_globe = Transform()
  trf_globe.translate(0.0, PED_H + GLOBE_R, 0.0)
  trf_globe.rotate(23.5, 0.0, 0.0, 1.0)
  trf_globe.rotate(-95.0, 0.0, 1.0, 0.0)
  trf_globe.scale(GLOBE_R, GLOBE_R, GLOBE_R)
  trf_globe_group = Transform()
  trf_globe_group.translate(-0.7, TOP_Y, -0.35)
  globe_group = Node(None, trf_globe_group,
                     nodes = [
                       Node(None, box(0.0, 0.0, 0.0, 0.35, PED_H, 0.35), [pedestal], [cube]),
                       Node(None, trf_globe, [globe, earth_tex], [sphere]),
                     ])

  # luminaria: base -> haste -> braco (gira em torno da haste) -> cupula
  # (inclina na ponta do braco); a spot tem a cupula como referencia
  arm_trf = Transform()
  head_trf = Transform()
  update_lamp()
  # a cupula e uma esfera achatada: a escala fica so no no dela, nao no no
  # de referencia da luz
  trf_shade = Transform()
  trf_shade.translate(0.0, 0.03, 0.0)
  trf_shade.scale(0.17, 0.09, 0.17)
  head = Node(None, head_trf,
              nodes = [
                Node(None, trf_shade, None, [sphere]),
                Node(None, sphere_at(0.0, -0.07, 0.0, 0.06), [bulb], [sphere]),
              ])
  arm = Node(None, arm_trf,
             nodes = [
               Node(None, box(ARM_L/2, -0.025, 0.0, ARM_L, 0.05, 0.05), None, [cube]),
               head,
             ])
  trf_lamp = Transform()
  trf_lamp.translate(LAMP_X, TOP_Y, LAMP_Z)
  lamp = Node(None, trf_lamp, [lamp_body],
              nodes = [
                Node(None, box(0.0, 0.0, 0.0, 0.36, 0.05, 0.36), None, [cube]),
                Node(None, box(0.0, 0.05, 0.0, 0.05, POST_H - 0.05, 0.05), None, [cube]),
                arm,
              ])
  light.set_reference(head)

  root = Node(pipeline, None, [plain], nodes = [table, box_group, globe_group, lamp])
  return cam, Scene(root), light

def draw () -> None:
  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)

def on_key (event: Any) -> None:
  global lamp_yaw, lamp_tilt
  key = event["key"]
  if key == "q":
    canvas.close()
  elif key == "r":
    camera.get_arcball().mat = gm.mat4(1)
  elif key in ("ArrowLeft", "ArrowRight"):
    lamp_yaw += 10.0 if key == "ArrowLeft" else -10.0
    update_lamp()
  elif key in ("ArrowUp", "ArrowDown"):
    lamp_tilt = max(-60.0, min(60.0, lamp_tilt + (5.0 if key == "ArrowUp" else -5.0)))
    update_lamp()
  elif key in ("=", "-"):
    cutoff = max(10.0, min(70.0, spot.get_cutoff() + (5.0 if key == "=" else -5.0)))
    spot.set_cutoff(cutoff, cutoff*INNER_RATIO)

def on_wheel (event: Any) -> None:
  # aproxima/afasta ao longo do eixo de visao (dz proporcional a distancia)
  camera.get_arcball().translate(0.0, 0.0, -event["dy"] * 0.001)

def main () -> None:
  global canvas, device, context, renderer, camera, scene, spot

  canvas = RenderCanvas(size=(800, 600), title="comp grafica - mesa com spot", update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, depth_test=True, clear_value=(1.0, 1.0, 1.0, 1.0))

  camera, scene, spot = build_scene(device, target_format)
  arcball = camera.create_arcball()
  arcball.attach(canvas)

  canvas.add_event_handler(on_key, "key_down")
  canvas.add_event_handler(on_wheel, "wheel")
  canvas.request_draw(draw)
  loop.run()

if __name__ == "__main__":
  main()
