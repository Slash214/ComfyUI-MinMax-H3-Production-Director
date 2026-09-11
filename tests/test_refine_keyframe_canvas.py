import ast
from pathlib import Path
import unittest
import torch

class RefineKeyframeCanvasTests(unittest.TestCase):
    def setUp(self):
        source=Path(__file__).resolve().parents[1]/'director/refine_sampling.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_resize_refine_keyframes')
        self.calls=[]
        def decode(vae, data):
            self.calls.append('decode')
            t=data['samples'];return torch.zeros((t.shape[2],t.shape[-2]*32,t.shape[-1]*32,3))
        def scale(images,w,h):return torch.zeros((images.shape[0],h,w,3))
        def encode(vae,images):return {'samples':torch.zeros((1,96,images.shape[0],images.shape[1]//32,images.shape[2]//32))}
        env={'torch':torch,'_decode_video':decode,'_scale_images':scale,'_encode_video':encode}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
        self.resize=env['_resize_refine_keyframes']

    def test_fl2v_first_and_last_anchors_match_new_canvas_without_mutating_source(self):
        latent=torch.zeros((1,96,1,21,38))
        original=[[torch.zeros(1),{'minimax_keyframes':[{'resolved_frame_index':0,'latent':latent},{'resolved_frame_index':169,'latent':latent}], 'other':'preserve'}]]
        out=self.resize(original,None,1344,768)
        self.assertEqual([k['resolved_frame_index'] for k in out[0][1]['minimax_keyframes']],[0,169])
        for k in out[0][1]['minimax_keyframes']:self.assertEqual(tuple(k['latent'].shape),(1,96,1,24,42))
        self.assertIs(original[0][1]['minimax_keyframes'][0]['latent'],latent)
        self.assertEqual(tuple(latent.shape[-2:]),(21,38))
        self.assertEqual(out[0][1]['other'],'preserve')

    def test_matching_canvas_and_reference_only_conditioning_do_not_decode(self):
        latent=torch.zeros((1,96,1,24,42))
        original=[[None,{'minimax_keyframes':[{'latent':latent}]}],[None,{'minimax_refs':['identity']}]]
        out=self.resize(original,None,1344,768)
        self.assertEqual(self.calls,[])
        self.assertIs(out[0][1]['minimax_keyframes'][0]['latent'],latent)
        self.assertEqual(out[1][1]['minimax_refs'],['identity'])

if __name__=='__main__':unittest.main()
