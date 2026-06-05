"""Testa alinhamento dos dentes da IA com os dentes originais usando deteccao de regiao branca."""
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np, cv2
from PIL import Image, ImageFilter

import importlib.util
spec = importlib.util.spec_from_file_location('srv', os.path.join(BASE, 'server.py'))
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

img = Image.open(os.path.join(BASE, 'paciente_teste.jpg')).convert('RGB')
img_arr = np.array(img, dtype=np.float32)
info = srv.detect_teeth(np.array(img))
bbox, teeth_poly, lm, lips_poly = info
x0, y0, x1, y1 = bbox

mask_teeth = srv.make_poly_mask(teeth_poly, img_arr.shape[1], img_arr.shape[0], blur=0)
mask_lips  = srv.make_poly_mask(lips_poly,  img_arr.shape[1], img_arr.shape[0], blur=4)

# Bbox dos dentes na imagem original
teeth_bool = mask_teeth > 0.5
rows = np.any(teeth_bool, axis=1); cols = np.any(teeth_bool, axis=0)
rmin,rmax = np.where(rows)[0][[0,-1]]; cmin,cmax = np.where(cols)[0][[0,-1]]
print(f'Dentes originais bbox: ({cmin},{rmin})-({cmax},{rmax})  size={cmax-cmin}x{rmax-rmin}')

# Carrega IA raw e detecta regiao branca dos dentes
ai_sq = Image.open(os.path.join(BASE, 'debug_ai_raw.jpg')).convert('RGB')
ai_arr_full = np.array(ai_sq, dtype=np.float32)

# Detecta dentes na IA: pixels muito claros (L>200 no LAB)
ai_bgr = cv2.cvtColor(ai_arr_full.astype(np.uint8), cv2.COLOR_RGB2BGR)
ai_lab = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2LAB)
ai_teeth_mask = (ai_lab[:,:,0] > 180).astype(np.uint8)
# Kernel para preencher buracos
kernel = np.ones((5,5), np.uint8)
ai_teeth_mask = cv2.morphologyEx(ai_teeth_mask, cv2.MORPH_CLOSE, kernel)
rows_ai = np.any(ai_teeth_mask, axis=1); cols_ai = np.any(ai_teeth_mask, axis=0)
if rows_ai.any() and cols_ai.any():
    rmin_ai,rmax_ai = np.where(rows_ai)[0][[0,-1]]; cmin_ai,cmax_ai = np.where(cols_ai)[0][[0,-1]]
    print(f'Dentes IA bbox: ({cmin_ai},{rmin_ai})-({cmax_ai},{rmax_ai})  size={cmax_ai-cmin_ai}x{rmax_ai-rmin_ai}')

    # Recorta dentes da IA
    ai_teeth_crop = ai_arr_full[rmin_ai:rmax_ai, cmin_ai:cmax_ai]
    ai_teeth_pil  = Image.fromarray(ai_teeth_crop.astype(np.uint8))

    # Redimensiona para o tamanho dos dentes originais
    orig_w = cmax - cmin
    orig_h = rmax - rmin
    ai_teeth_resized = np.array(ai_teeth_pil.resize((orig_w, orig_h), Image.LANCZOS), dtype=np.float32)

    # Ajusta cor para alvo Oral Unic
    ai_bgr2 = cv2.cvtColor(ai_teeth_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ai_lab2  = cv2.cvtColor(ai_bgr2, cv2.COLOR_BGR2LAB).astype(np.float32)
    ai_L = float(ai_lab2[:,:,0].mean()); ai_A = float(ai_lab2[:,:,1].mean()); ai_B = float(ai_lab2[:,:,2].mean())
    tL,tA,tB = 168, 143, 144
    ai_lab2[:,:,0] = np.clip(ai_lab2[:,:,0] + (tL-ai_L), 0, 255)
    ai_lab2[:,:,1] = np.clip(ai_lab2[:,:,1] + (tA-ai_A)*0.85, 0, 255)
    ai_lab2[:,:,2] = np.clip(ai_lab2[:,:,2] + (tB-ai_B)*0.85, 0, 255)
    ai_rgb_adj = cv2.cvtColor(cv2.cvtColor(ai_lab2.clip(0,255).astype(np.uint8), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB).astype(np.float32)

    # Composicao: cola os dentes da IA na posicao exata dos dentes originais
    mask_region = mask_teeth[rmin:rmax, cmin:cmax]
    m_pil = Image.fromarray((mask_region*255).astype(np.uint8),'L').filter(ImageFilter.GaussianBlur(3))
    m_b   = np.array(m_pil, dtype=np.float32)[:,:,np.newaxis] / 255.0

    orig_region = img_arr[rmin:rmax, cmin:cmax].astype(np.float32)
    blended_region = orig_region*(1-m_b) + ai_rgb_adj*m_b

    result = img_arr.copy()
    result[rmin:rmax, cmin:cmax] = blended_region

    # Suaviza borda na regiao dos labios tambem
    lips_region = mask_lips[rmin:rmax, cmin:cmax]
    m_lips = np.array(Image.fromarray((lips_region*255).astype(np.uint8),'L').filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)[:,:,np.newaxis]/255.0

    Image.fromarray(result.clip(0,255).astype('uint8')).save(os.path.join(BASE,'test_alinhado.jpg'), quality=95)
    print('Salvo test_alinhado.jpg')
else:
    print('Nao detectou dentes na IA')
