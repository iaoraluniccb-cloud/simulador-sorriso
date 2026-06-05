"""Alinha dentes da IA usando a propria mascara dos landmarks — sem depender de threshold de cor."""
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
dw, dh = x1-x0, y1-y0

mask_teeth = srv.make_poly_mask(teeth_poly, img_arr.shape[1], img_arr.shape[0], blur=3)
mask_lips  = srv.make_poly_mask(lips_poly,  img_arr.shape[1], img_arr.shape[0], blur=4)

# Recorte enviado para a IA
pad_x = int(dw*1.0); pad_y = int(dh*1.2)
cx0=max(0,x0-pad_x); cy0=max(0,y0-pad_y)
cx1=min(img_arr.shape[1],x1+pad_x); cy1=min(img_arr.shape[0],y1+pad_y)
cw, ch = cx1-cx0, cy1-cy0
print(f'Recorte original: {cw}x{ch} px  bbox dentes: ({x0},{y0})-({x1},{y1})')

# Calcula onde os dentes ficam DENTRO do quadrado 1024x1024 enviado para a IA
size = 1024
scale = size / max(cw, ch)
new_w = int(cw*scale); new_h = int(ch*scale)
ox = (size-new_w)//2; oy = (size-new_h)//2

# Posicao dos dentes no espaco do recorte (relativo ao recorte)
dx0_rel = x0 - cx0; dy0_rel = y0 - cy0
dx1_rel = x1 - cx0; dy1_rel = y1 - cy0
# Posicao no quadrado 1024 (apos scale + offset)
dx0_sq = int(dx0_rel * scale) + ox
dy0_sq = int(dy0_rel * scale) + oy
dx1_sq = int(dx1_rel * scale) + ox
dy1_sq = int(dy1_rel * scale) + oy
print(f'Dentes no quadrado 1024: ({dx0_sq},{dy0_sq})-({dx1_sq},{dy1_sq})')

# Mascara dos dentes no espaco do quadrado
teeth_crop_local = mask_teeth[cy0:cy1, cx0:cx1]
lips_crop_local  = mask_lips[cy0:cy1, cx0:cx1]

teeth_scaled = np.array(Image.fromarray((teeth_crop_local*255).astype(np.uint8),'L').resize((new_w,new_h),Image.LANCZOS), dtype=np.float32)/255.0
lips_scaled  = np.array(Image.fromarray((lips_crop_local*255).astype(np.uint8),'L').resize((new_w,new_h),Image.LANCZOS), dtype=np.float32)/255.0

# Carrega IA e extrai regiao dos dentes usando a mascara
ai_sq = Image.open(os.path.join(BASE, 'debug_ai_raw.jpg')).convert('RGB')
ai_region = np.array(ai_sq, dtype=np.float32)[oy:oy+new_h, ox:ox+new_w]

# Ajusta cor para Oral Unic
ai_bgr = cv2.cvtColor(ai_region.astype(np.uint8), cv2.COLOR_RGB2BGR)
ai_lab  = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
m_t = teeth_scaled > 0.3
if m_t.any():
    ai_L = float(ai_lab[:,:,0][m_t].mean())
    ai_A = float(ai_lab[:,:,1][m_t].mean())
    ai_B = float(ai_lab[:,:,2][m_t].mean())
    print(f'IA cor dentes: L={ai_L:.1f} A={ai_A:.1f} B={ai_B:.1f}')
    tL,tA,tB = 168, 143, 144
    ai_lab[:,:,0] = np.clip(ai_lab[:,:,0] + (tL-ai_L), 0, 255)
    ai_lab[:,:,1] = np.clip(ai_lab[:,:,1] + (tA-ai_A)*0.85, 0, 255)
    ai_lab[:,:,2] = np.clip(ai_lab[:,:,2] + (tB-ai_B)*0.85, 0, 255)
ai_adj = cv2.cvtColor(cv2.cvtColor(ai_lab.clip(0,255).astype(np.uint8),cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB).astype(np.float32)

# Volta para o tamanho do recorte original
ai_back = np.array(Image.fromarray(ai_adj.astype(np.uint8)).resize((cw,ch),Image.LANCZOS), dtype=np.float32)

# Blend na regiao do recorte — dentes internos 100% IA, bordas lips 25%
m_in  = teeth_crop_local[:,:,np.newaxis]
m_out = np.maximum(lips_crop_local - teeth_crop_local, 0)[:,:,np.newaxis] * 0.30
m_b   = np.clip(m_in + m_out, 0, 1)
m_b_smooth = np.array(Image.fromarray((m_b[:,:,0]*255).astype(np.uint8),'L').filter(ImageFilter.GaussianBlur(4)),dtype=np.float32)[:,:,np.newaxis]/255.0

orig_c = img_arr[cy0:cy1,cx0:cx1].astype(np.float32)
blended = orig_c*(1-m_b_smooth) + ai_back*m_b_smooth
result = img_arr.copy()
result[cy0:cy1,cx0:cx1] = blended
Image.fromarray(result.clip(0,255).astype('uint8')).save(os.path.join(BASE,'test_alinhado2.jpg'),quality=95)
print('Salvo test_alinhado2.jpg')
