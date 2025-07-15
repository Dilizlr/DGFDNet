import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models.DADNet import build_net
from PIL import Image
from torchvision import transforms

def register_hooks(model, layer_name):
    def hook_fn(module, input, output):
        global activations
        activations = output[1].detach()
    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(hook_fn)

def generate_heatmap(activation):
    weights = torch.mean(activation, dim=[2, 3], keepdim=True)
    heatmap = torch.sum(weights * activation, dim=1, keepdim=True)
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap

def compute_dark_channel(image, window_size=9):
    min_channel = np.min(image, axis=2)  # 在 RGB 通道中取最小值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)  # 取局部最小值
    return dark_channel

def save_heatmap_and_dark_channel(image_path, model, heatmap_path, dark_channel_path):
    global activations
    trans = transforms.ToTensor()
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = trans(input_image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    heatmap = generate_heatmap(activations)
    heatmap = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, color_heatmap)
    
    # 计算并保存暗通道
    image_np = np.array(input_image)
    dark_channel = compute_dark_channel(image_np)
    cv2.imwrite(dark_channel_path, dark_channel)

model = build_net()
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
if device != 'cpu':
    model.to(device)
state_dict = torch.load('/data/home/zlr/Code/DGFDNet/NHhaze/Dehazing/Dense-Haze/results/DCFNet/Dense/20250318_165522-nhhaze-20.49/Best.pkl')
model.load_state_dict(state_dict['model'])
register_hooks(model, 'layer5.blocks.1.hfounit') ## 5 1

save_heatmap_and_dark_channel(
    '/data/home/zlr/Code/Hazydataset/NH-HAZE/test/hazy/53_hazy.png',
    model,
    'heatmap_output_color53_1.jpg',
    'dark_channel_output53.jpg'
)
