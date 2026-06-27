import onnxruntime as ort

print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())

from insightface.app import FaceAnalysis

app = FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

app.prepare(ctx_id=0)

print("GPU SUCCESS")