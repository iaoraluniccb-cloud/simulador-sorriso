"""
Teste direto: envia paciente_teste.jpg para o servidor e salva o resultado.
"""
import urllib.request, json, base64, os, time

DIR = os.path.dirname(os.path.abspath(__file__))

# Lê a foto
with open(os.path.join(DIR, "paciente_teste.jpg"), "rb") as f:
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

payload = json.dumps({
    "image": img_b64,
    "style": "faceta_bl1",
    "patient": {"name": "Paciente Teste"}
}).encode("utf-8")

print(f"[>>] Enviando para /api/generate ({len(payload)//1024}KB)...")
t0 = time.time()

req = urllib.request.Request(
    "http://localhost:8765/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        rj = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    b64_result = rj["data"][0]["b64_json"]
    result_bytes = base64.b64decode(b64_result)
    out_path = os.path.join(DIR, "test_result_final.jpg")
    with open(out_path, "wb") as f:
        f.write(result_bytes)
    print(f"[OK] Resultado salvo: {out_path} ({len(result_bytes)//1024}KB) em {elapsed:.1f}s")
except Exception as e:
    print(f"[ERRO] {e}")
