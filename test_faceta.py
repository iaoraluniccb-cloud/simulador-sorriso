import sys, os, importlib.util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('srv', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server.py'))
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
import numpy as np
from PIL import Image

img = Image.open('paciente_teste.jpg').convert('RGB')
img_arr = np.array(img, dtype=np.float32)
info = srv.detect_teeth(np.array(img))
bbox, teeth_poly, lm, lips_poly = info
mask_teeth = srv.make_poly_mask(teeth_poly, img_arr.shape[1], img_arr.shape[0], blur=3)
mask_lips  = srv.make_poly_mask(lips_poly,  img_arr.shape[1], img_arr.shape[0], blur=4)

for vita in ['BL1', 'BL2', 'BL3']:
    result = srv.apply_faceta_ai(img_arr, mask_lips, mask_teeth, bbox, {'vita': vita})
    Image.fromarray(result.clip(0,255).astype('uint8')).save('test_' + vita + '.jpg', quality=95)
    print('Salvo test_' + vita + '.jpg')
