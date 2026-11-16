"""Export the trained encoder (output/model/palmprint_encoder.pth) to ONNX for the Rust port.
Output: the 512-d pooled ResNet-18 features that app.py actually uses. Run from the repo root:
    python rust/tools/export_onnx.py rust/models/palmprint_encoder_512.onnx
Note: utils/palmprint_encoder.onnx is NOT this model: utils/model_diagram.py exports an untrained
PalmprintEncoder (ImageNet backbone + random 256-d head) at 224×224 for the diagram."""
import sys, numpy as np, torch, torchvision, onnxruntime as ort
sd = torch.load('output/model/palmprint_encoder.pth', map_location='cpu', weights_only=True)
r = torchvision.models.resnet18(weights=None)
enc = torch.nn.Sequential(*list(r.children())[:-1]); enc.load_state_dict(sd); enc.eval()
class Flat(torch.nn.Module):
    def __init__(s, e): super().__init__(); s.e = e
    def forward(s, x): return s.e(x).flatten(1)
m = Flat(enc).eval(); x = torch.rand(1, 3, 138, 138)
torch.onnx.export(m, x, sys.argv[1], input_names=['input'], output_names=['embedding'], opset_version=17,
                  dynamic_axes={'input': {0: 'b', 2: 'h', 3: 'w'}, 'embedding': {0: 'b'}}, dynamo=False)
with torch.no_grad(): ref = m(x).numpy()
got = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).run(None, {'input': x.numpy()})[0]
print('exported', sys.argv[1], 'max|Δ| vs PyTorch =', float(np.abs(got - ref).max()))
