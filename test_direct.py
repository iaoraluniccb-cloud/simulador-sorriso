"""
Teste direto: processa a imagem sem servidor HTTP e salva debugs intermediários.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import io, numpy as np
from PIL import Image, ImageFilter, ImageDraw

# Importa funções do server
from server import detect_teeth, build_mask_L, call_openai_edit_retry, post_process_teeth, STYLE_PROMPTS, DIR

STYLE = "faceta_bl1"

print("=== Teste Direto (sem servidor HTTP) ===")
with open("paciente_teste.jpg", "rb") as f:
    img_bytes = f.read()

orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
orig_w, orig_h = orig.size
print(f"Original: {orig_w}x{orig_h}")

# Upscale
min_dim = min(orig_w, orig_h)
scale = max(min(1536 / orig_w, 1536 / orig_h, 1.0), 1024 / min_dim if min_dim < 1024 else 1.0)
work_w = int(orig_w * scale)
work_h = int(orig_h * scale)
img_work = orig.resize((work_w, work_h), Image.LANCZOS)
print(f"Work: {work_w}x{work_h} (scale={scale:.2f})")

img_arr = np.array(img_work)
info = detect_teeth(img_arr)

if info is None:
    print("ERRO: Rosto não detectado!")
    sys.exit(1)

teeth_poly, lips_poly = info
print(f"Landmarks detectados OK. Dentes: {len(teeth_poly)} pontos")

api_size = 1024
api_scale = min(api_size / work_w, api_size / work_h)
api_w = int(work_w * api_scale)
api_h = int(work_h * api_scale)
img_resized = img_work.resize((api_w, api_h), Image.LANCZOS)

canvas = Image.new("RGB", (api_size, api_size), (0, 0, 0))
ox = (api_size - api_w) // 2
oy = (api_size - api_h) // 2
canvas.paste(img_resized, (ox, oy))

def to_canvas(px, py):
    return (int(px * api_scale) + ox, int(py * api_scale) + oy)

teeth_cv = [to_canvas(p[0], p[1]) for p in teeth_poly]
lips_cv  = [to_canvas(p[0], p[1]) for p in lips_poly]
mask_L   = build_mask_L(teeth_cv, api_size, api_size, lips_cv)

print(f"Canvas: {api_size}x{api_size} | offset: ({ox},{oy}) | img: {api_w}x{api_h}")
print(f"Dentes no canvas: x={min(p[0] for p in teeth_cv)}-{max(p[0] for p in teeth_cv)} y={min(p[1] for p in teeth_cv)}-{max(p[1] for p in teeth_cv)}")

mask_arr = np.array(mask_L)
print(f"Máscara: pixels brancos = {(mask_arr > 128).sum()}")

# Salva imagem enviada pra IA
canvas_rgba = canvas.convert("RGBA")
canvas_rgba.putalpha(Image.fromarray(255 - mask_arr))
canvas_rgba.save("debug_direct_sent.png")
print("Salvo: debug_direct_sent.png")

buf_img = io.BytesIO()
canvas_rgba.save(buf_img, format="PNG")

prompt = STYLE_PROMPTS[STYLE]
print(f"\nPrompt ({len(prompt)} chars):\n{prompt[:200]}...\n")
print("Chamando OpenAI gpt-image-2...")

import time
t0 = time.time()
result_bytes, err = call_openai_edit_retry(buf_img.getvalue(), prompt, api_size, api_size)
elapsed = time.time() - t0

if not result_bytes:
    print(f"ERRO OpenAI: {err}")
    sys.exit(1)

print(f"OpenAI OK em {elapsed:.1f}s — {len(result_bytes)//1024}KB")

# Salva raw da IA
Image.open(io.BytesIO(result_bytes)).save("debug_direct_ai_raw.jpg")
print("Salvo: debug_direct_ai_raw.jpg")

# Pós-processamento e blend
result_canvas = Image.open(io.BytesIO(result_bytes)).convert("RGB")
result_canvas = post_process_teeth(result_canvas, mask_arr, ox, oy, api_w, api_h)

blend_mask = mask_L.filter(ImageFilter.GaussianBlur(radius=2))
orig_arr   = np.array(canvas, dtype=np.float32)
result_arr = np.array(result_canvas, dtype=np.float32)
bm_arr = np.array(blend_mask, dtype=np.float32)[:,:,np.newaxis] / 255.0
blended = orig_arr * (1 - bm_arr) + result_arr * bm_arr
blended = np.clip(blended, 0, 255).astype(np.uint8)
blended_canvas = Image.fromarray(blended)

edited_region = blended_canvas.crop((ox, oy, ox + api_w, oy + api_h))
result_work = edited_region.resize((work_w, work_h), Image.LANCZOS)
result_final = result_work.resize((orig_w, orig_h), Image.LANCZOS)

result_final.save("debug_direct_result.jpg", quality=95)
print("Salvo: debug_direct_result.jpg")
print("=== DONE ===")
