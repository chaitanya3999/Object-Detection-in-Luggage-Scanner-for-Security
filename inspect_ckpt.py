import torch
ckpt = torch.load('checkpoints/stage2_ultimate.pth', map_location='cpu')
sd = ckpt['model_state_dict']
print("First layer shape:", sd['yolo.model.model.0.conv.weight'].shape)
print("Detect head shape:", sd['yolo.model.model.22.cv3.0.2.weight'].shape)
