# 修改了 DSDCN
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
from einops import rearrange, reduce
from torch.nn import init as init
from torchvision.ops.deform_conv import DeformConv2d


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################

## Fre-MLP
class frequency_selection(nn.Module):
    def __init__(self, dim, dw=2, norm='backward', act_method=nn.GELU, bias=False):
        super().__init__()
        self.act_fft = act_method()
        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        return y


class MGAM(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()

        hidden_features = int(dim * 1)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)

        self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=2, dilation=2, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=4, dilation=2, groups=hidden_features , bias=bias)

        self.sig3 = nn.Sigmoid()
        self.sig5 = nn.Sigmoid()
        
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_features*2, dim, 1, bias=bias),
            nn.GELU()
        )

        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=bias),
            nn.GELU()
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.maskconv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, xd):
        id = x
        x = self.project_in(x)

        x1_3, x2_3 = self.dwconv3x3(x).chunk(2, dim=1)
        x1_5, x2_5 = self.dwconv5x5(x).chunk(2, dim=1)


        x1 = self.dwconv3x3_1(x1_3) * self.sig3(x2_3)
        x2 = self.dwconv5x5_1(x1_5) * self.sig5(x2_5)

        x = self.fuse(torch.cat([x1, x2], dim=1))

        x = self.ca_conv(torch.cat([id,x], dim=1))
        attn = self.ca(x)
        x = self.project_out(x * attn)
        mask = self.maskconv(xd * attn)

        return x, mask


class CDEMoudle(nn.Module):
    def __init__(self, fourier_dim, in_ch=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, fourier_dim, kernel_size=5, padding=2, dilation=1, padding_mode='reflect'),
            nn.Conv2d(fourier_dim, fourier_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, dark):
        x = self.layer1(dark)
        return x


class HAFM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim),
            # nn.GELU()
        )
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
        self.freq = frequency_selection(dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.GELU()
        )

 
    def forward(self, inp, dark):
        x = self.conv1(inp)
        xd = self.pa(dark)
        x = self.dw2(x) * self.sigmoid(xd) + x
        xf = self.freq(x)
        x = self.fuse(torch.cat([x, xf], dim=1))
        return x, xd


class DGFDBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # self.norm1 = LayerNorm(dim)
        # self.norm2 = LayerNorm(dim)


        self.attn = HAFM(dim)
        self.ffn = MGAM(dim)
    

    def forward(self, x):

        inp, dark = x


        x = inp
        
        # first stage
        identity = x
        x = self.norm1(x)
        x, xd = self.attn(x, dark)
        x = identity + x


        # second stage
        identity = x
        x = self.norm2(x)
        x, mask = self.ffn(x, xd)
        x = identity + x

        return x, mask

 
class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [DGFDBlock(dim=dim) for i in range(depth)])
        self.fuse = SKFusion(dim=dim)

    def forward(self, x):
        inp, dark, mask = x
        x = inp
        ori_dark = dark
        for blk in self.blocks:
            # print('000')
            if mask is not None:
                dark = self.fuse([ori_dark, mask])
            x, mask = blk([x, dark])
        return x, mask

#######################################################################
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class DGFDNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 embed_dims=[32, 64, 128, 64, 32],
                 depths=[2, 2, 4, 2, 2]):
        super().__init__()

        fourier_dim = 32
        self.cdemoudle = CDEMoudle(fourier_dim, 1)
 
        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.dark_down1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)


        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.dark_down2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
       

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        self.dark_up1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        
        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        
        self.dark_up2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        
        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        

    def forward_features(self, x, dark):

        # dark_img = self.get_dark(cond)
        density0 = self.cdemoudle(dark)
        density1 = self.dark_down1(density0)
        density2 = self.dark_down2(density1)

        # encder
        x = self.patch_embed(x)
        x, mask0 = self.layer1([x, density0, None])
        skip1 = x

        x = self.patch_merge1(x)
        mask1 = self.dark_down1(mask0)
        x, mask1 = self.layer2([x, density1, mask1])
        skip2 = x

        x = self.patch_merge2(x)
        mask2 = self.dark_down2(mask1)
        x, mask2 = self.layer3([x, density2, mask2])

        # decoder
        x = self.patch_split1(x)
        mask1 = self.dark_up1(mask2)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x, mask1 = self.layer4([x, density1, mask1])

        x = self.patch_split2(x)
        mask0 = self.dark_up2(mask1)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x, mask0 = self.layer5([x, density0, mask0])

        x = self.patch_unembed(x)
        return x
    
    def get_dark(self, input):

        pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        values, _ = torch.min(input, dim=1, keepdim=True)
        
        input = pool(1 - values)

        return 1 - input

    def forward(self, inp):
        # H, W = inp.shape[2:]
        dark = self.get_dark(inp)

        feat = self.forward_features(inp, dark)
        out = inp + feat

        return out


if __name__ == '__main__':
    net = DGFDNet()
    # net = DWConv2d(3, 48)


    input = torch.randn((4, 3, 256, 256))
    cond = torch.randn((4, 3, 256, 256))

    output = net(input, cond, 100)
    print('output', output.shape)

def build_net():
    return DGFDNet()
