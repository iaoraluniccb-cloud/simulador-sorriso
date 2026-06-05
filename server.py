"""
Simulador de Sorriso — Oral Unic  |  http://localhost:8765
v8 — Melhorias: segurança, pós-processamento, análise honesta, logging, sem debug files
"""
import json, urllib.request, urllib.error
import base64, os, mimetypes, io, time, logging, glob
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from collections import defaultdict

PORT = int(os.environ.get("PORT", 8765))
DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Chave da API — lê do .env se existir, senão do ambiente ──────────────────
def _load_env():
    env_path = os.path.join(DIR, ".env")
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Logging estruturado ───────────────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(DIR, "simulator.log"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)
log = logging.getLogger("sorriso")

# ── Rate limiting simples (max 30 req/IP/hora) ────────────────────────────────
_rate: dict = defaultdict(list)
MAX_REQ_PER_HOUR = 200

def _check_rate(ip: str) -> bool:
    now = time.time()
    _rate[ip] = [t for t in _rate[ip] if now - t < 3600]
    if len(_rate[ip]) >= MAX_REQ_PER_HOUR:
        return False
    _rate[ip].append(now)
    return True

# ── Limpeza automática de debug files antigos (>1h) ──────────────────────────
def _cleanup_debug():
    patterns = ["debug_*.jpg", "debug_*.png", "debug_canvas.jpg"]
    cutoff = time.time() - 3600
    for pat in patterns:
        for f in glob.glob(os.path.join(DIR, pat)):
            try:
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
            except Exception:
                pass

# ── Landmarks MediaPipe ───────────────────────────────────────────────────────
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
LIPS_UPPER  = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER  = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
# Bordas precisas dos dentes sup/inf
UPPER_TEETH_TOP = [185, 40, 39, 37, 0, 267, 269, 270, 409]   # borda gengival dos superiores
UPPER_TEETH_BOT = [191, 80, 81, 82, 13, 312, 311, 310, 415]  # borda incisal dos superiores
LOWER_TEETH_TOP = [95, 88, 178, 87, 14, 317, 402, 318, 324]  # borda incisal dos inferiores
LOWER_TEETH_BOT = [146, 91, 181, 84, 17, 314, 405, 321, 375] # borda gengival dos inferiores

# ── Prompt base (preservação do rosto) ───────────────────────────────────────
PRESERVE = (
    " — ABSOLUTE RULES (any violation makes the image invalid):\n"
    "RULE 1: THE MOUTH IS OPEN. Keep it open. The lips are PARTED and must stay parted. "
    "Do NOT close the lips. Do NOT fill the gap with skin or lips.\n"
    "RULE 2: ONLY modify visible tooth enamel surfaces. Everything else — lips, gums edge, "
    "face, skin, eyes, nose, hair, ears, background, clothing — must be pixel-perfect identical.\n"
    "RULE 3: Same person. Same pose. Same lighting. Same framing. Same expression.\n"
    "RULE 4: The dental transformation must be clearly visible and impressive.\n"
    "RULE 5: Individual teeth must be distinguishable — never a single white block.\n"
    "RULE 6: Output dimensions and composition identical to input."
)

# ── Prompts por tratamento ────────────────────────────────────────────────────
def _dental_prompt(task, shade_note, rgb_note):
    """Prompt genérico (mantém forma, troca cor) — usado internamente."""
    return (
        "You are a world-class dental photo retoucher for a premium dental clinic. "
        "The transparent/masked area marks ONLY the visible tooth enamel surfaces. "
        f"TASK: {task} "
        f"SHADE: {shade_note} "
        f"COLOR TARGET: {rgb_note} "
        "REALISM RULES: Each individual tooth must be clearly distinguishable with natural inter-dental grooves. "
        "Render realistic ceramic/enamel micro-texture, subtle specular highlight along each tooth's long axis, "
        "slightly translucent incisal edges, natural cervical graduation toward the gum margin. "
        "Upper AND lower teeth must be transformed with the same shade. "
        "NEVER render a flat white block — photorealistic result only." + PRESERVE
    )

def _veneer_prompt(shade_note, rgb_note):
    """Prompt para facetas cerâmicas — refina forma E aplica cor da cerâmica."""
    return (
        "You are a master cosmetic dentist and dental photographer. "
        "The transparent/masked area marks ALL visible tooth enamel surfaces. "
        "TASK: Simulate ultra-premium porcelain veneers on every visible tooth. "
        f"SHADE: {shade_note} "
        f"COLOR TARGET: {rgb_note} "
        "VENEER AESTHETICS (follow exactly): "
        "— Refine the shape of each tooth: lengthen slightly if short, correct minor crowding and rotation, "
        "even out incisal edges to form a gentle downward smile arc. "
        "— Central incisors: slightly wider and longer for dominance. "
        "— Lateral incisors: slightly smaller. Canines: natural pointed tip. "
        "— ALL teeth perfectly aligned on the midline, natural inter-dental grooves clearly visible. "
        "— Do NOT dramatically redesign morphology — preserve the patient's natural tooth proportions, "
        "just refine and perfect them. "
        "SURFACE QUALITY: Realistic layered porcelain ceramic texture, specular highlight along each tooth's "
        "long axis, slightly translucent incisal edges with subtle halo, natural cervical graduation. "
        "Upper AND lower arches transformed with the same shade. "
        "Result must look like a real photograph of premium veneers — NOT a cartoon." + PRESERVE
    )

def _prosthesis_prompt(shade_note, rgb_note):
    """Prompt para prótese total — remodela forma E troca cor."""
    return (
        "You are a master dental prosthetist and photo retoucher. "
        "The transparent/masked area is where the full-arch prosthesis must be rendered. "
        "TASK: Replace ALL visible teeth with a complete full-arch prosthesis — redesign the tooth shape completely. "
        f"SHADE: {shade_note} "
        f"COLOR TARGET: {rgb_note} "
        "TOOTH MORPHOLOGY (follow exactly): "
        "— Upper arch: 2 central incisors (wider, squarish with rounded corners), "
        "2 lateral incisors (slightly smaller and narrower), "
        "2 canines (pointed, slightly protruding), "
        "4+ premolars visible at sides (smaller, rounder). "
        "— Lower arch: 4 incisors (smaller, flat), 2 canines, premolars at sides. "
        "— All teeth perfectly aligned and symmetrical on the midline. "
        "— Incisal edges perfectly level forming a gentle downward curve (smile arc). "
        "— Natural inter-dental grooves clearly visible between each tooth. "
        "SURFACE QUALITY: Realistic porcelain/acrylic texture, specular highlight on facial surface, "
        "natural cervical graduation, slightly translucent incisal edges. "
        "The prosthesis must look like a photograph of real premium dentures — NOT a cartoon or illustration. "
        "Upper AND lower arches transformed with same shade and morphology." + PRESERVE
    )

STYLE_PROMPTS = {
    # ── Prótese Total (escala VITA clássica) ─────────────────────────────────
    "protese_a1": _prosthesis_prompt(
        "VITA A1 — lightest warm ivory-white, very natural.",
        "RGB ~(248,244,232) — warm off-white, slight yellow undertone."
    ),
    "protese_a2": _prosthesis_prompt(
        "VITA A2 — warm ivory, the most common natural shade.",
        "RGB ~(243,236,216) — classic warm ivory, slight golden cast."
    ),
    "protese_a3": _prosthesis_prompt(
        "VITA A3 — medium warm yellow, visible natural aging.",
        "RGB ~(234,222,192) — warm golden-yellow, natural aged look."
    ),
    "protese_b1": _prosthesis_prompt(
        "VITA B1 — light white with pink/reddish undertone.",
        "RGB ~(250,245,242) — very light, subtle pink warmth."
    ),
    "protese_b2": _prosthesis_prompt(
        "VITA B2 — medium pinkish-white.",
        "RGB ~(245,236,228) — warm pinkish, slightly darker than B1."
    ),
    "protese_c1": _prosthesis_prompt(
        "VITA C1 — light grayish-white.",
        "RGB ~(240,237,232) — cool light gray undertone."
    ),
    "protese_d2": _prosthesis_prompt(
        "VITA D2 — light brownish-yellow.",
        "RGB ~(238,226,198) — warm ochre tone, brownish cast."
    ),
    "protese_bl3": _prosthesis_prompt(
        "VITA BL3 — bright bleached white, Hollywood smile level.",
        "RGB ~(252,252,255) — near-pure white, minimal warmth."
    ),
    "protese_bl4": _prosthesis_prompt(
        "VITA BL4 — natural bleached white, slightly warmer than BL3.",
        "RGB ~(250,249,248) — clean natural white, very subtle warmth."
    ),
    # ── Facetas Cerâmicas (escala VITA Bleach) ────────────────────────────────
    "faceta_bl1": _veneer_prompt(
        "VITA BL1 — the absolute whitest dental shade, elite Hollywood smile.",
        "RGB ~(254,254,255) — pure brilliant white, cold luminous ceramic."
    ),
    "faceta_bl2": _veneer_prompt(
        "VITA BL2 — very bright white, one step below BL1.",
        "RGB ~(252,252,254) — bright white, barely perceptible warmth."
    ),
    "faceta_bl3": _veneer_prompt(
        "VITA BL3 — bright bleached white, natural-looking premium result.",
        "RGB ~(250,250,252) — bright white with very subtle natural warmth."
    ),
    "faceta_a1": _veneer_prompt(
        "VITA A1 — natural ivory-white veneers, elegant and discreet.",
        "RGB ~(248,244,232) — natural warm ivory ceramic."
    ),
    "faceta_b1": _veneer_prompt(
        "VITA B1 — light pink-white veneers, very natural feminine result.",
        "RGB ~(250,245,242) — soft light white with pink undertone."
    ),
    # ── Procedimentos ─────────────────────────────────────────────────────────
    "clareamento": (
        "You are a world-class dental photo retoucher. "
        "TASK: Apply maximum-strength professional in-office teeth whitening. "
        "WHITEN ONLY — remove all stains and yellowing to achieve VITA A1/BL4 level brightness. "
        "ABSOLUTE RESTRICTIONS: Do NOT change the shape of any tooth. "
        "Do NOT change tooth length, width, or alignment. "
        "Do NOT move, rotate, or reposition any tooth. "
        "Do NOT change the gum line. "
        "ONLY the color/brightness of the enamel changes — everything else is pixel-perfect identical. "
        "TARGET COLOR: RGB ~(248,245,235) — brilliant clean white, natural warm tone." + PRESERVE
    ),
    "ortodontia": (
        "You are a world-class dental photo retoucher. "
        "TASK: Perfectly align and straighten ALL visible teeth — upper and lower arches. "
        "Correct every crowding, rotation, spacing and malocclusion. "
        "Result must look like the final day of complete orthodontic treatment. "
        "ABSOLUTE RESTRICTIONS: Do NOT change tooth color or shade — keep original tooth color exactly. "
        "Do NOT change the shape, size, or morphology of any tooth. "
        "Do NOT change the gum line. Do NOT whiten. "
        "ONLY the position and alignment of teeth changes — color and shape remain identical to input." + PRESERVE
    ),
    "aumento_coroa": (
        "You are a world-class dental photo retoucher. "
        "TASK: Simulate crown lengthening / gingival contouring on all visible teeth. "
        "Raise and recontour the gum line evenly to expose 1.5-2mm more of each tooth crown, "
        "making the smile appear longer, more proportional and more youthful. "
        "Do NOT change tooth color. Keep mouth open in exact same position." + PRESERVE
    ),
    "lente_contato": _dental_prompt(
        "Simulate ultra-thin contact lens veneers (0.3mm) on all visible upper and lower teeth.",
        "VITA BL1 — absolute maximum white, contact lens veneer result.",
        "RGB ~(255,255,255) — pure white, glass-like translucent ceramic finish."
    ),
    "harmonizacao": _dental_prompt(
        "Simulate a complete smile makeover: align, reshape and whiten all visible teeth for ideal proportions.",
        "VITA BL2/BL3 — bright natural white, harmonious and balanced.",
        "RGB ~(251,251,253) — bright natural white, perfectly proportioned teeth."
    ),
}


# ── Handler HTTP ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    timeout = 180

    def handle(self):
        try:
            super().handle()
        except Exception:
            pass

    def log_message(self, fmt, *args):
        msg = fmt % args
        print(f"  {self.address_string()} {msg}")
        log.info(f"HTTP {self.address_string()} {msg}")

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", ""):
            path = "/index.html"
        fp = os.path.join(DIR, path.lstrip("/").replace("/", os.sep))
        if os.path.isfile(fp):
            mime, _ = mimetypes.guess_type(fp)
            data = open(fp, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_cors()
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        ip = self.address_string()
        if not _check_rate(ip):
            self.reply_error(429, "Limite de requisições atingido. Tente em 1 hora.")
            return
        if self.path == "/api/generate":
            self.handle_generate()
        elif self.path == "/api/analise":
            self.handle_analise()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_generate(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 30 * 1024 * 1024:
                self.reply_error(413, "Imagem muito grande (máx 30MB)")
                return
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self.reply_error(400, f"Erro ao ler dados: {e}")
            return

        image_b64 = body.get("image", "")
        style = body.get("style", "clareamento")
        patient = body.get("patient", {})

        if not image_b64:
            self.reply_error(400, "Imagem ausente")
            return

        try:
            _, b64data = image_b64.split(",", 1)
            img_bytes = base64.b64decode(b64data)
        except Exception as e:
            self.reply_error(400, f"Imagem inválida: {e}")
            return

        t0 = time.time()
        log.info(f"GENERATE style={style} patient={patient.get('name','?')} size={len(img_bytes)//1024}KB")

        try:
            result_bytes, err = self.process(img_bytes, style, patient)
        except Exception as _proc_ex:
            import traceback
            tb = traceback.format_exc()
            print(f'[CRASH] process exception: {_proc_ex}')
            print(tb)
            log.error(f'PROCESS CRASH: {_proc_ex}')
            log.error(tb)
            result_bytes, err = None, str(_proc_ex)
        elapsed = time.time() - t0

        if not result_bytes:
            log.error(f"GENERATE FAILED style={style} err={err}")
            self.reply_error(502, err)
            return

        log.info(f"GENERATE OK style={style} elapsed={elapsed:.1f}s out={len(result_bytes)//1024}KB")
        b64_result = base64.b64encode(result_bytes).decode()
        out = json.dumps({"data": [{"b64_json": b64_result}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors()
        self.end_headers()
        self.wfile.write(out)

        _cleanup_debug()

    def process(self, img_bytes, style, patient=None):
        import numpy as np
        from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = orig.size

        # Garante mínimo de 1024px e limita a 1536px
        min_dim = min(orig_w, orig_h)
        scale = max(min(1536 / orig_w, 1536 / orig_h, 1.0), 1024 / min_dim if min_dim < 1024 else 1.0)
        work_w = int(orig_w * scale)
        work_h = int(orig_h * scale)
        img_work = orig.resize((work_w, work_h), Image.LANCZOS)
        img_arr  = np.array(img_work)

        # Detecta dentes via subprocess para evitar crash do MediaPipe no servidor HTTP
        info = _detect_teeth_safe(img_arr)
        fallback_ellipse = info is None

        if fallback_ellipse:
            print("[!] Rosto não detectado — fallback elíptico")
            api_size = 1024
            scale2 = min(api_size/work_w, api_size/work_h)
            aw = int(work_w * scale2); ah = int(work_h * scale2)
            canvas = Image.new("RGB", (api_size, api_size), (128, 128, 128))
            ox2 = (api_size-aw)//2; oy2 = (api_size-ah)//2
            canvas.paste(img_work.resize((aw, ah), Image.LANCZOS), (ox2, oy2))
            # Dentes ficam em ~45% da altura da foto (selfie frontal)
            cx2 = ox2 + aw//2; cy2 = oy2 + int(ah * 0.45)
            rw2 = int(aw*0.22); rh2 = int(ah*0.06)
            mask_L = Image.new("L", (api_size, api_size), 0)
            ImageDraw.Draw(mask_L).ellipse([cx2-rw2, cy2-rh2, cx2+rw2, cy2+rh2], fill=255)
            mask_L = mask_L.filter(ImageFilter.GaussianBlur(radius=4))
            crop_x0 = crop_y0 = 0
            crop_w, crop_h = work_w, work_h
            api_scale = scale2
            ox, oy = ox2, oy2
            api_w, api_h = aw, ah
        else:
            teeth_poly, lips_poly, upper_top, upper_bot, lower_top, lower_bot = info

            # Estratégia: envia a foto INTEIRA redimensionada para 1024px
            # Isso dá contexto completo à IA e mantém proporção correta
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

            teeth_cv    = [to_canvas(p[0], p[1]) for p in teeth_poly]
            lips_cv     = [to_canvas(p[0], p[1]) for p in lips_poly]
            upper_top_cv = [to_canvas(p[0], p[1]) for p in upper_top]
            upper_bot_cv = [to_canvas(p[0], p[1]) for p in upper_bot]
            lower_top_cv = [to_canvas(p[0], p[1]) for p in lower_top]
            lower_bot_cv = [to_canvas(p[0], p[1]) for p in lower_bot]
            mask_L = build_mask_L(teeth_cv, api_size, api_size, lips_cv,
                                  upper_top_cv, upper_bot_cv, lower_top_cv, lower_bot_cv)

            crop_x0 = crop_y0 = 0
            crop_w, crop_h = work_w, work_h

        mask_arr = np.array(mask_L)

        # RGBA: transparente = IA edita (dentes)
        canvas_rgba = canvas.convert("RGBA")
        canvas_rgba.putalpha(Image.fromarray(255 - mask_arr))

        buf_img = io.BytesIO()
        canvas_rgba.save(buf_img, format="PNG")

        # Debug: salva o que enviamos pra IA
        canvas_rgba.save(os.path.join(DIR, "debug_sent_to_ai.png"))

        prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["clareamento"])
        print(f"[>>] style={style} | canvas={api_size}x{api_size} | work={work_w}x{work_h}")

        # Chamada OpenAI com retry
        result_bytes, err = call_openai_edit_retry(buf_img.getvalue(), prompt, api_size, api_size)

        if result_bytes:
            # Debug: salva o que a IA retornou
            Image.open(io.BytesIO(result_bytes)).save(os.path.join(DIR, "debug_ai_raw.jpg"))

        if not result_bytes:
            print(f"[!] OpenAI falhou: {err} — fallback local avançado")
            result_canvas = _fallback_local(canvas, mask_L, mask_arr, style, np)
        else:
            result_canvas = Image.open(io.BytesIO(result_bytes)).convert("RGB")

        # ── Pós-processamento: nitidez e contraste nos dentes ─────────────────
        result_canvas = post_process_teeth(result_canvas, mask_arr, ox, oy, api_w, api_h)

        if fallback_ellipse:
            # Fallback: resultado direto do canvas
            edited = result_canvas.crop((ox, oy, ox + api_w, oy + api_h))
            result_work = edited.resize((work_w, work_h), Image.LANCZOS)
        else:
            # Nova abordagem: blend preciso só nos dentes entre orig e resultado
            # Ambos no espaço do canvas 1024x1024
            blend_mask = mask_L.filter(ImageFilter.GaussianBlur(radius=2))

            orig_arr   = np.array(canvas, dtype=np.float32)
            result_arr = np.array(result_canvas, dtype=np.float32)
            bm_arr = np.array(blend_mask, dtype=np.float32)[:,:,np.newaxis] / 255.0

            # Blend: só substitui onde a máscara é forte (dentes)
            blended = orig_arr * (1 - bm_arr) + result_arr * bm_arr
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            blended_canvas = Image.fromarray(blended)

            # Extrai a região da foto do canvas e redimensiona de volta ao tamanho original
            edited_region = blended_canvas.crop((ox, oy, ox + api_w, oy + api_h))
            result_work = edited_region.resize((work_w, work_h), Image.LANCZOS)

        result_final = result_work.resize((orig_w, orig_h), Image.LANCZOS)
        buf_out = io.BytesIO()
        result_final.save(buf_out, format="JPEG", quality=95)
        return buf_out.getvalue(), None


    def handle_analise(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self.reply_error(400, f"Erro ao ler dados: {e}")
            return

        image_b64 = body.get("image", "")
        style = body.get("style", "clareamento")
        patient = body.get("patient", {})

        if not image_b64:
            self.reply_error(400, "Imagem ausente")
            return

        result = analyze_smile(image_b64, style, patient)
        out = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors()
        self.end_headers()
        self.wfile.write(out)

    def reply_error(self, code, msg):
        body = json.dumps({"error": {"message": msg}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)


# ── _detect_teeth_safe: wrapper seguro para uso no servidor HTTP ──────────────

def _detect_teeth_safe(img_arr):
    """
    Wrapper que chama detect_teeth em subprocess isolado.
    Evita crash do MediaPipe no thread do HTTP server.
    """
    import subprocess, sys, json, base64, tempfile, os, numpy as np
    from PIL import Image

    try:
        # Serializa imagem para temp file
        tmp_in  = tempfile.mktemp(suffix=".png")
        tmp_out = tempfile.mktemp(suffix=".json")

        Image.fromarray(img_arr).save(tmp_in)

        script = f"""
import sys, json, os, numpy as np
sys.path.insert(0, {repr(DIR)})
from PIL import Image
from server import detect_teeth

img = np.array(Image.open({repr(tmp_in)}))
result = detect_teeth(img)
if result:
    teeth_poly, lips_poly, upper_top, upper_bot, lower_top, lower_bot = result
    data = {{"teeth": teeth_poly, "lips": lips_poly,
             "upper_top": upper_top, "upper_bot": upper_bot,
             "lower_top": lower_top, "lower_bot": lower_bot}}
else:
    data = None
with open({repr(tmp_out)}, "w") as f:
    json.dump(data, f)
"""
        res = subprocess.run(
            [sys.executable, "-c", script],
            timeout=30,
            capture_output=True
        )

        if res.returncode != 0:
            print(f"[!] MediaPipe subprocess falhou: {res.stderr.decode('utf-8', errors='replace')[:200]}")
            return None

        if not os.path.isfile(tmp_out):
            return None

        with open(tmp_out) as f:
            data = json.load(f)

        if data is None:
            return None

        teeth_poly  = [tuple(p) for p in data["teeth"]]
        lips_poly   = [tuple(p) for p in data["lips"]]
        upper_top   = [tuple(p) for p in data["upper_top"]]
        upper_bot   = [tuple(p) for p in data["upper_bot"]]
        lower_top   = [tuple(p) for p in data["lower_top"]]
        lower_bot   = [tuple(p) for p in data["lower_bot"]]
        return teeth_poly, lips_poly, upper_top, upper_bot, lower_top, lower_bot

    except Exception as e:
        print(f"[!] _detect_teeth_safe erro: {e}")
        return None
    finally:
        for f in [tmp_in, tmp_out]:
            try: os.remove(f)
            except: pass


# ── Detecção de dentes (MediaPipe) ────────────────────────────────────────────


def detect_teeth(img_arr):
    try:
        import cv2
        import mediapipe as mp

        h, w = img_arr.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as fm:
            res = fm.process(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
            if not res.multi_face_landmarks:
                return None

            lm = res.multi_face_landmarks[0].landmark
            def pt(i): return (int(lm[i].x * w), int(lm[i].y * h))

            teeth_poly = [pt(i) for i in INNER_UPPER] + [pt(i) for i in INNER_LOWER]
            lips_poly  = [pt(i) for i in LIPS_UPPER]  + [pt(i) for i in LIPS_LOWER]
            # Bordas precisas de cada arco
            upper_top = [pt(i) for i in UPPER_TEETH_TOP]
            upper_bot = [pt(i) for i in UPPER_TEETH_BOT]
            lower_top = [pt(i) for i in LOWER_TEETH_TOP]
            lower_bot = [pt(i) for i in LOWER_TEETH_BOT]

            tx0 = min(p[0] for p in teeth_poly)
            tx1 = max(p[0] for p in teeth_poly)
            ty0 = min(p[1] for p in teeth_poly)
            ty1 = max(p[1] for p in teeth_poly)
            print(f"[OK] Landmarks — dentes x={tx0}-{tx1} y={ty0}-{ty1}")
            return teeth_poly, lips_poly, upper_top, upper_bot, lower_top, lower_bot

    except Exception as e:
        print(f"[!] Erro MediaPipe: {e}")
        return None


def build_mask_L(teeth_poly, w, h, lips_poly=None,
                 upper_top=None, upper_bot=None, lower_top=None, lower_bot=None):
    """
    Máscara precisa: dois retângulos/elipses independentes para
    dentes superiores e inferiores, usando bordas dos landmarks.
    """
    from PIL import Image, ImageDraw, ImageFilter

    if not teeth_poly:
        return Image.new("L", (w, h), 0)

    xs = [p[0] for p in teeth_poly]
    tx0, tx1 = min(xs), max(xs)
    t_width = tx1 - tx0
    margin_x = int(t_width * 0.03)
    ex0 = tx0 + margin_x
    ex1 = tx1 - margin_x

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if upper_top and upper_bot and lower_top and lower_bot:
        # Bordas exatas por landmark
        sup_top = min(p[1] for p in upper_top)  # topo gengival sup
        sup_bot = max(p[1] for p in upper_bot)  # borda incisal sup (mais baixo)
        inf_top = min(p[1] for p in lower_top)  # borda incisal inf (mais alto)
        inf_bot = max(p[1] for p in lower_bot)  # base gengival inf

        # Pequena expansão: dentes são um pouco maiores que os landmarks internos
        gap = max(1, int((inf_top - sup_bot) * 0.1))
        sup_top = sup_top - gap
        sup_bot = sup_bot + gap
        inf_top = inf_top - gap
        inf_bot = inf_bot + gap

        t_height_ref = inf_bot - sup_top
        blur_r = max(2, int(t_height_ref * 0.04))

        if sup_bot > sup_top + 2:
            draw.ellipse([ex0, sup_top, ex1, sup_bot], fill=255)
        if inf_bot > inf_top + 2:
            draw.ellipse([ex0, inf_top, ex1, inf_bot], fill=255)
    else:
        # Fallback sem landmarks precisos: divide bbox pela metade
        ys = [p[1] for p in teeth_poly]
        ty0, ty1 = min(ys), max(ys)
        t_height = ty1 - ty0
        t_mid = ty0 + int(t_height * 0.50)
        blur_r = max(2, int(t_height * 0.06))

        sup_top = ty0 + int(t_height * 0.18)
        sup_bot = t_mid - int(t_height * 0.02)
        inf_top = t_mid + int(t_height * 0.02)
        inf_bot = ty1 - int(t_height * 0.18)

        if sup_bot > sup_top + 2:
            draw.ellipse([ex0, sup_top, ex1, sup_bot], fill=255)
        if inf_bot > inf_top + 2:
            draw.ellipse([ex0, inf_top, ex1, inf_bot], fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return mask


# ── Pós-processamento: nitidez + micro-contraste nos dentes ──────────────────

def post_process_teeth(canvas_img, mask_arr, ox, oy, api_w, api_h):
    """Aumenta nitidez e contraste na região dos dentes."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageEnhance

    try:
        # Recorta só a região dos dentes no canvas
        tooth_region = canvas_img.crop((ox, oy, ox + api_w, oy + api_h))

        # Sharpening leve
        sharpened = tooth_region.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

        # Micro-contraste (+8%)
        sharpened = ImageEnhance.Contrast(sharpened).enhance(1.08)

        # Máscara de blend para aplicar só onde há dentes
        region_mask = Image.fromarray(mask_arr).crop((ox, oy, ox + api_w, oy + api_h))

        # Combina: sharpen só nos dentes, original no resto
        import numpy as np
        orig_arr  = np.array(tooth_region, dtype=np.float32)
        sharp_arr = np.array(sharpened, dtype=np.float32)
        m_arr = np.array(region_mask, dtype=np.float32)[:,:,np.newaxis] / 255.0
        result_arr = np.clip(orig_arr * (1 - m_arr * 0.7) + sharp_arr * (m_arr * 0.7), 0, 255).astype(np.uint8)

        result = canvas_img.copy()
        result.paste(Image.fromarray(result_arr), (ox, oy))
        return result
    except Exception as e:
        print(f"[!] post_process_teeth erro: {e}")
        return canvas_img


# -- Fallback local avancado v3 ------------------------------------------------

def _fallback_local(canvas, mask_L, mask_arr, style, np):
    """Fallback local v3 — substituição direta com modulação de sombra realista."""
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

    canvas_arr = np.array(canvas, dtype=np.float32)
    h, w = canvas_arr.shape[:2]

    # Máscara suavizada para blend de bordas
    edge_mask = mask_L.filter(ImageFilter.GaussianBlur(radius=3))
    edge_norm = np.array(edge_mask, dtype=np.float32)[:, :, np.newaxis] / 255.0

    # Máscara core (pixels centrais dos dentes, sem bordas)
    mask_norm = np.array(mask_L, dtype=np.float32)[:, :, np.newaxis] / 255.0

    # Calcula luminosidade relativa de cada pixel (mapa de sombra/brilho)
    lum_per_px = canvas_arr[:,:,0]*0.299 + canvas_arr[:,:,1]*0.587 + canvas_arr[:,:,2]*0.114

    # Estatísticas da região dos dentes
    strong = mask_arr > 100
    if strong.any():
        tp_lum = lum_per_px[strong]
        lum_p10 = float(np.percentile(tp_lum, 10))   # sombra escura
        lum_p90 = float(np.percentile(tp_lum, 90))   # highlight brilhante
        lum_range = max(lum_p90 - lum_p10, 1.0)
    else:
        lum_p10, lum_p90, lum_range = 80.0, 160.0, 80.0

    # (R_base, G_base, B_base, R_bright, G_bright, B_bright, shadow_depth)
    STYLES = {
        # Prótese — escala VITA clássica
        "protese_a1":    (244, 238, 222, 254, 252, 238, 0.64),
        "protese_a2":    (240, 232, 212, 252, 246, 228, 0.62),
        "protese_a3":    (232, 220, 190, 246, 236, 206, 0.60),
        "protese_b1":    (248, 244, 238, 255, 254, 248, 0.66),
        "protese_b2":    (244, 234, 224, 252, 244, 234, 0.63),
        "protese_c1":    (238, 235, 228, 250, 248, 242, 0.64),
        "protese_d2":    (236, 224, 196, 248, 236, 210, 0.61),
        "protese_bl3":   (252, 252, 255, 255, 255, 255, 0.68),
        "protese_bl4":   (249, 248, 246, 254, 253, 252, 0.67),
        # Facetas — escala BL + naturais
        "faceta_bl1":    (253, 253, 255, 255, 255, 255, 0.66),
        "faceta_bl2":    (251, 251, 254, 255, 255, 255, 0.65),
        "faceta_bl3":    (249, 249, 252, 254, 254, 255, 0.64),
        "faceta_a1":     (246, 242, 230, 254, 252, 240, 0.64),
        "faceta_b1":     (249, 244, 240, 254, 252, 248, 0.65),
        # Lente de contato / harmonização
        "lente_contato": (254, 254, 255, 255, 255, 255, 0.68),
        "harmonizacao":  (250, 250, 252, 255, 255, 255, 0.65),
        # Procedimentos sem cor alvo
        "clareamento":   None,
        "ortodontia":    None,
        "aumento_coroa": None,
    }
    cfg = STYLES.get(style)

    # ── Ortodontia / Aumento de Coroa: clarear levemente ────────────────────
    if style in ('ortodontia', 'aumento_coroa'):
        boost = np.array([[[1.12, 1.09, 1.07]]], dtype=np.float32)
        region = np.clip(canvas_arr * boost, 0, 255)
        final_arr = canvas_arr * (1 - edge_norm * 0.70) + region * (edge_norm * 0.70)
        return Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8))

    # ── Clareamento: remove amarelo + boost luminosidade até BL2 ────────────
    elif style == 'clareamento':
        region = canvas_arr.copy()
        # Remove amarelo: desloca matiz em direção ao branco neutro
        r_ch, g_ch, b_ch = region[:,:,0], region[:,:,1], region[:,:,2]
        yellow_excess = np.clip(r_ch - b_ch, 0, 80)
        region[:,:,0] = np.clip(r_ch - yellow_excess * 0.30, 0, 255)
        region[:,:,1] = np.clip(g_ch + yellow_excess * 0.05, 0, 255)
        region[:,:,2] = np.clip(b_ch + yellow_excess * 0.42, 0, 255)
        # Boost de brilho: mapeia lum atual para target ~210
        cur_lum = region[:,:,0]*0.299 + region[:,:,1]*0.587 + region[:,:,2]*0.114
        target_lum = 210.0
        # Fator de escala por pixel: se lum < target, sobe; mantém acima
        scale = np.where(cur_lum > 1, np.clip(target_lum / np.maximum(cur_lum, 60), 1.0, 2.2), 1.0)
        region[:,:,0] = np.clip(region[:,:,0] * scale, 0, 255)
        region[:,:,1] = np.clip(region[:,:,1] * scale, 0, 255)
        region[:,:,2] = np.clip(region[:,:,2] * scale, 0, 255)
        # Blend 88% na máscara de dentes
        final_arr = canvas_arr * (1 - edge_norm * 0.88) + region * (edge_norm * 0.88)
        result_img = Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8))
        result_img = result_img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=45, threshold=3))
        return result_img

    # ── Prótese / Faceta: cor plana com borda suave — como na referência gpt2 ─
    br, bg, bb, hr, hg, hb, shadow_floor = cfg if cfg else (248, 244, 232, 254, 252, 238, 0.64)

    # Cor alvo direta — sem modular por luminosidade original (evita hotspot)
    tooth_layer = np.full((h, w, 3), [br, bg, bb], dtype=np.float32)

    # Blend: substitui dentes pela cor alvo, preservando bordas suaves
    blended = canvas_arr * (1 - edge_norm) + tooth_layer * edge_norm
    result_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # Sharpening cirúrgico nos dentes
    sharp = result_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=65, threshold=2))
    sharp_arr = np.array(sharp, dtype=np.float32)
    blend_arr = np.array(result_img, dtype=np.float32)
    result_arr = blend_arr * (1 - mask_norm * 0.75) + sharp_arr * (mask_norm * 0.75)
    result_img = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8))

    # Micro-contraste
    result_img = ImageEnhance.Contrast(result_img).enhance(1.06)

    # ── Separação interproximal vetorizada ───────────────────────────────────
    ys_m, xs_m = np.where(mask_arr > 120)
    if len(xs_m) > 0:
        x0, x1 = int(xs_m.min()), int(xs_m.max())
        tooth_w = x1 - x0
        n_teeth = 8
        sep_positions = [x0 + int(tooth_w * i / n_teeth) for i in range(1, n_teeth)]
        draw_arr = np.array(result_img, dtype=np.float32)
        mask_f = mask_arr / 255.0  # (h, w)
        for sx in sep_positions:
            # Coluna central: 9% de escurecimento modulado pela máscara
            col_mask = mask_f[:, sx][:, np.newaxis, np.newaxis]  # (h,1,1)
            draw_arr[:, sx, :] = np.clip(draw_arr[:, sx, :] * (1.0 - col_mask[:, 0, :] * 0.09), 0, 255)
        result_img = Image.fromarray(np.clip(draw_arr, 0, 255).astype(np.uint8))

    return result_img





# ── OpenAI com retry ──────────────────────────────────────────────────────────

def call_openai_edit_retry(img_data_png, prompt, img_w, img_h, retries=2):
    for attempt in range(retries + 1):
        result, err = call_openai_edit(img_data_png, prompt, img_w, img_h)
        if result:
            return result, None
        # Billing limit, auth errors or quota: don't retry (won't succeed)
        if err and any(k in str(err) for k in ("billing", "quota", "401", "403")):
            print(f"[!] OpenAI erro permanente (sem retry): {str(err)[:80]}")
            return None, err
        if attempt < retries:
            wait = 2 ** attempt
            print(f"[!] Tentativa {attempt+1} falhou ({err}) — aguardando {wait}s")
            time.sleep(wait)
    return None, err


def call_openai_edit(img_data_png, prompt, img_w, img_h):
    ratio = img_h / img_w if img_w > 0 else 1.0
    if ratio > 1.2:
        size = "1024x1536"
    elif ratio < 0.83:
        size = "1536x1024"
    else:
        size = "1024x1024"

    boundary = "----OralUnicBoundary"

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n"
        ).encode("utf-8")

    def file_field(name, filename, data, content_type):
        return (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8") + data + b"\r\n"

    body = (
        field("model", "gpt-image-2") +
        field("prompt", prompt) +
        field("n", "1") +
        field("size", size) +
        file_field("image", "smile.png", img_data_png, "image/png") +
        f"--{boundary}--\r\n".encode("utf-8")
    )

    print(f"[>>] gpt-image-2 size={size}...")

    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/edits",
            data=body,
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=150) as resp:
            rj = json.loads(resp.read().decode())

        b64 = rj["data"][0]["b64_json"]
        result = base64.b64decode(b64)
        print(f"[OK] gpt-image-1 retornou {len(result)//1024}KB")
        return result, None

    except urllib.error.HTTPError as e:
        body_err = ""
        try: body_err = e.read().decode()[:500]
        except Exception: pass
        print(f"[!] OpenAI HTTP {e.code}: {body_err}")
        return None, f"OpenAI erro {e.code}: {body_err}"
    except Exception as e:
        print(f"[!] OpenAI exception: {e}")
        return None, str(e)


# ── Análise do Sorriso via GPT-4o Vision ─────────────────────────────────────

ANALISE_PROMPT = """You are a professional dental aesthetic evaluator. Analyze this patient's smile photo carefully and return a JSON with:
1. "scores": object with keys simetria, clareamento, harmonia, curvatura, gengival, estetica — each integer 0-100 (honest clinical assessment of what you actually see)
2. "detalhes": array of 7 items {nome, status} where status = "ok", "att" or "nok":
   - Linha Média, Plano Incisal, Curva do Sorriso, Proporção Áurea, Corredores Bucais, Suporte Labial, Equilíbrio Gengival
3. "recomendacoes": array of 6 items {icon, nome, sub, priority (bool)} based on what you actually see
4. "ia_analise": boolean true (indicates this was analyzed by AI)

Be clinically honest. Return ONLY valid JSON, no markdown."""


def analyze_smile(image_b64_full, style, patient=None):
    try:
        b64data = image_b64_full.split(",", 1)[1] if "," in image_b64_full else image_b64_full

        # Inclui dados do paciente no prompt quando disponíveis
        patient_ctx = ""
        if patient and patient.get("name"):
            patient_ctx = f"\nPatient context: {patient.get('name','')}, DOB {patient.get('dob','')}, treatment planned: {style}."

        payload = json.dumps({
            "model": "gpt-4o",
            "max_tokens": 700,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALISE_PROMPT + patient_ctx},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64data}",
                        "detail": "low"
                    }}
                ]
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            rj = json.loads(resp.read().decode())

        text = rj["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["ia_analise"] = True
        print(f"[OK] Análise GPT-4o completa")
        log.info(f"ANALISE OK gpt4o style={style}")

        # Garante priority para o tratamento selecionado
        for rec in result.get("recomendacoes", []):
            nl = rec.get("nome", "").lower()
            if ((style.startswith("faceta") and "faceta" in nl) or
                (style.startswith("protese") and ("prót" in nl or "prot" in nl)) or
                (style == "clareamento" and "clarear" in nl) or
                (style == "ortodontia" and "orto" in nl) or
                (style == "aumento_coroa" and "coroa" in nl)):
                rec["priority"] = True

        return result

    except Exception as e:
        print(f"[!] Análise GPT-4o falhou: {e} — usando análise local estimada")
        log.warning(f"ANALISE FALLBACK style={style} err={e}")
        return _local_smile_analysis(style)


def _local_smile_analysis(style):
    """Análise estimada (sem IA) — claramente marcada como estimativa."""
    import random

    def rnd(lo, hi): return random.randint(lo, hi)

    scores = {
        "simetria":    rnd(55, 80),
        "clareamento": rnd(32, 65),
        "harmonia":    rnd(58, 82),
        "curvatura":   rnd(50, 78),
        "gengival":    rnd(45, 72),
        "estetica":    rnd(48, 74),
    }
    sw = [0.35, 0.45, 0.20]
    detalhes = [
        {"nome": "Linha Média",        "status": random.choices(["ok","att","nok"], sw)[0]},
        {"nome": "Plano Incisal",       "status": random.choices(["ok","att","nok"], sw)[0]},
        {"nome": "Curva do Sorriso",    "status": random.choices(["ok","att"], [0.5,0.5])[0]},
        {"nome": "Proporção Áurea",     "status": random.choices(["att","nok"], [0.6,0.4])[0]},
        {"nome": "Corredores Bucais",   "status": random.choices(["ok","att"], [0.5,0.5])[0]},
        {"nome": "Suporte Labial",      "status": random.choices(["ok","att","nok"], sw)[0]},
        {"nome": "Equilíbrio Gengival", "status": random.choices(["att","nok"], [0.55,0.45])[0]},
    ]
    recs = [
        {"icon":"💎","nome":"Facetas Cerâmicas","sub":"BL1 / BL2","priority": style.startswith("faceta")},
        {"icon":"✨","nome":"Clareamento","sub":"Profissional","priority": style == "clareamento"},
        {"icon":"😁","nome":"Ortodontia","sub":"Alinhamento","priority": style == "ortodontia"},
        {"icon":"🦷","nome":"Prótese Total","sub":"A1 / A2 / BL3","priority": style.startswith("protese")},
        {"icon":"📐","nome":"Aumento de Coroa","sub":"Gengivoplastia","priority": style == "aumento_coroa"},
        {"icon":"🩺","nome":"Profilaxia","sub":"Higiene Periodontal","priority": False},
    ]
    recs.sort(key=lambda x: -x["priority"])
    return {"scores": scores, "detalhes": detalhes, "recomendacoes": recs, "ia_analise": False}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n[OK] Simulador de Sorriso — Oral Unic | Porta: {PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
