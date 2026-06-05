"""Testa a composicao usando o debug_ai_raw.jpg ja gerado — sem chamar a IA novamente."""
import sys, os, importlib.util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np, cv2
from PIL import Image, ImageFilter

spec = importlib.util.spec_from_file_location('srv', os.path.join(BASE, 'server.py'))
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

img = Image.open(os.path.join(BASE, 'paciente_teste.jpg')).convert('RGB')
img_arr = np.array(img, dtype=np.float32)
info = srv.detect_teeth(np.array(img))
bbox, teeth_poly, lm, lips_poly = info
x0, y0, x1, y1 = bbox
dw, dh = x1-x0, y1-y0
pad_x = int(dw*1.0); pad_y = int(dh*1.2)
cx0=max(0,x0-pad_x); cy0=max(0,y0-pad_y)
cx1=min(img_arr.shape[1],x1+pad_x); cy1=min(img_arr.shape[0],y1+pad_y)
cw, ch = cx1-cx0, cy1-cy0

mask_teeth = srv.make_poly_mask(teeth_poly, img_arr.shape[1], img_arr.shape[0], blur=3)
mask_lips  = srv.make_poly_mask(lips_poly,  img_arr.shape[1], img_arr.shape[0], blur=4)
teeth_crop = mask_teeth[cy0:cy1, cx0:cx1]
lips_crop  = mask_lips[cy0:cy1, cx0:cx1]
edit_mask  = np.maximum(teeth_crop, lips_crop)

# Carrega resultado da IA ja gerado
size = 1024
scale = size / max(cw, ch)
new_w = int(cw*scale); new_h = int(ch*scale)
ox = (size-new_w)//2; oy = (size-new_h)//2

ai_sq   = Image.open(os.path.join(BASE, 'debug_ai_raw.jpg')).convert('RGB')
ai_crop = ai_sq.crop((ox, oy, ox+new_w, oy+new_h))
ai_orig = ai_crop.resize((cw, ch), Image.LANCZOS)
ai_arr  = np.array(ai_orig, dtype=np.float32)

orig_crop_u8 = img_arr[cy0:cy1,cx0:cx1].astype(np.uint8)
ai_crop_u8   = ai_arr.clip(0,255).astype(np.uint8)

orig_lab = cv2.cvtColor(cv2.cvtColor(orig_crop_u8, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB).astype(np.float32)
ai_lab   = cv2.cvtColor(cv2.cvtColor(ai_crop_u8,   cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB).astype(np.float32)

m_bool = edit_mask > 0.3
ai_L_mean  = float(ai_lab[:,:,0][m_bool].mean())
ai_A_mean  = float(ai_lab[:,:,1][m_bool].mean())
ai_B_mean  = float(ai_lab[:,:,2][m_bool].mean())
print(f'IA cor media: L={ai_L_mean:.1f} A={ai_A_mean:.1f} B={ai_B_mean:.1f}')

# Alvo Oral Unic BL1
tL, tA, tB = 168, 143, 144
dL = tL - ai_L_mean; dA = tA - ai_A_mean; dB = tB - ai_B_mean
ai_lab_adj = ai_lab.copy()
ai_lab_adj[:,:,0] = np.clip(ai_lab[:,:,0] + dL, 0, 255)
ai_lab_adj[:,:,1] = np.clip(ai_lab[:,:,1] + dA*0.90, 0, 255)
ai_lab_adj[:,:,2] = np.clip(ai_lab[:,:,2] + dB*0.90, 0, 255)
ai_bgr_adj = cv2.cvtColor(ai_lab_adj.clip(0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
ai_rgb_adj = cv2.cvtColor(ai_bgr_adj, cv2.COLOR_BGR2RGB).astype(np.float32)

# Blend: forma da IA nos dentes, transicao suave nas bordas
m_in  = teeth_crop[:,:,np.newaxis]
m_out = np.maximum(lips_crop - teeth_crop, 0)[:,:,np.newaxis] * 0.25
m_b   = np.clip(m_in + m_out, 0, 1)
m_b_pil = Image.fromarray((m_b[:,:,0]*255).astype(np.uint8),'L').filter(ImageFilter.GaussianBlur(4))
m_b = np.array(m_b_pil, dtype=np.float32)[:,:,np.newaxis] / 255.0

orig_full_f = img_arr[cy0:cy1,cx0:cx1].astype(np.float32)
blended = orig_full_f*(1-m_b) + ai_rgb_adj*m_b
result = img_arr.copy()
result[cy0:cy1,cx0:cx1] = blended
Image.fromarray(result.clip(0,255).astype('uint8')).save(os.path.join(BASE,'test_forma_BL1.jpg'), quality=95)
print('Salvo test_forma_BL1.jpg')
