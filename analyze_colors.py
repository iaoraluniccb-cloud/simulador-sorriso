import numpy as np, cv2
from PIL import Image

attachments = "C:/Users/DWOS/.g4os-public/workspaces/my-workspace/sessions/260515-eager-ridge/attachments"
files = [
    attachments + "/786415cc-93f6-4fb0-ae15-0c78172cb408_WhatsApp Image 2026-05-27 at 12.28.13 (4).jpeg",
    attachments + "/e8aa696a-b0ba-4102-b5f4-15a28d2e2c79_WhatsApp Image 2026-05-27 at 12.28.13 (3).jpeg",
    attachments + "/7ab826df-8fc9-4fcc-b935-64a4080cc1ee_WhatsApp Image 2026-05-27 at 12.28.13 (1).jpeg",
    attachments + "/be53e3ab-5411-40a1-bd50-54636675d9b0_WhatsApp Image 2026-05-27 at 12.28.13 (2).jpeg",
    attachments + "/a9ba37c7-2e77-464d-900c-2806ca9629ae_WhatsApp Image 2026-05-27 at 12.28.13.jpeg",
    attachments + "/70b8c892-15f0-4d0e-9da7-da3678a89010_WhatsApp Image 2026-05-27 at 12.28.12 (1).jpeg",
    attachments + "/7274676d-765f-4bc9-ae2b-cd16ac1359ed_WhatsApp Image 2026-05-27 at 12.28.12.jpeg",
]

results = []
for p in files:
    try:
        img = np.array(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        # Metade inferior = foto "depois"
        after = img[h//2:, :]
        ah, aw = after.shape[:2]
        # Regiao central dos dentes
        cx, cy = aw//2, ah//4
        roi = after[max(0,cy-50):cy+70, max(0,cx-100):cx+100]
        lab = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB).astype(np.float32)
        # Filtra pixels claros (dentes, nao labios nem gengiva)
        mask = lab[:,:,0] > 128
        if mask.any():
            L = float(lab[:,:,0][mask].mean())
            A = float(lab[:,:,1][mask].mean())
            B = float(lab[:,:,2][mask].mean())
            results.append((L,A,B))
            print("L=%.1f A=%.1f B=%.1f  amarelo_relativo=+%.1f" % (L, A, B, B-128))
    except Exception as e:
        print("ERRO:", e)

if results:
    Lm = np.mean([r[0] for r in results])
    Am = np.mean([r[1] for r in results])
    Bm = np.mean([r[2] for r in results])
    print()
    print("=== MEDIA ORAL UNIC FACETAS ===")
    print("L=%.1f A=%.1f B=%.1f" % (Lm, Am, Bm))
    print("(LAB OpenCV: L 0-255, A/B neutro=128)")
