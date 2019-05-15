import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from style_decorator import style_decorator, adain
from VGGdecoder import Decoder
from normalisedVGG import NormalisedVGG


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = NormalisedVGG(pretrained_path='vgg_normalised_conv5_1.pth').net
        self.block1 = vgg[: 4]
        self.block2 = vgg[4: 11]
        self.block3 = vgg[11: 18]
        self.block4 = vgg[18: 31]

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=True):
        h1 = self.block1(images)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class VGGDecoder(nn.Module):
    def __init__(self, level=4, pretrained_path=None):
        super().__init__()
        decoder_relu4_1 = Decoder(level, pretrained_path).net
        self.block4 = decoder_relu4_1[:13]
        self.block3 = decoder_relu4_1[13: 20]
        self.block2 = decoder_relu4_1[20: 26]
        self.block1 = decoder_relu4_1[27:]

    def forward(self, reconstructed_features, style_intermediate_features):
        d4 = self.block4(reconstructed_features)
        d4 = adain(d4, style_intermediate_features[2])

        d3 = self.block3(d4)
        d3 = adain(d3, style_intermediate_features[1])

        d2 = self.block2(d3)
        d2 = adain(d2, style_intermediate_features[0])

        d1 = self.block1(d2)

        return d1


class AutoEncoder(nn.Module):
    def __init__(self, decoder_level=4, decoder_pretrained_path='decoder_relu4_1.pth'):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder(decoder_level, decoder_pretrained_path)

    @staticmethod
    def calc_tv_loss(img, weight):
        """
        Compute total variation loss.
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the average total variation
          loss for img weighted by tv_weight.
        """
        # w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        # h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        w_variance = F.mse_loss(img[:, :, :, :-1], img[:, :, :, 1:])
        h_variance = F.mse_loss(img[:, :, :-1, :], img[:, :, 1:, :])
        loss = weight * (h_variance + w_variance)
        return loss

    @staticmethod
    def calc_content_loss(out, target):
        return F.mse_loss(out, target)

    @staticmethod
    def calc_perceptual_loss(out_middle_features, target_middle_features, weight):
        loss = 0
        for c, t in zip(out_middle_features, target_middle_features,):
            loss += F.mse_loss(c, t)
        loss = weight * loss
        return loss

    def forward(self, content_images, style_images, patch_size, alpha, lam1=0.1, lam2=0.01):
        reconstructed_images = self.generate(content_images, style_images, patch_size, alpha)

        # image reconstruction loss
        content_loss = self.calc_content_loss(content_images, reconstructed_images)

        # perceptual loss
        content_images_latent = self.encoder(content_images, output_last_feature=False)
        reconstructed_images_latent = self.encoder(reconstructed_images, output_last_feature=False)
        perceptual_loss = self.calc_perceptual_loss(reconstructed_images_latent, content_images_latent, lam1)

        # total variation loss
        tv_loss = self.calc_tv_loss(reconstructed_images, lam2)

        loss = content_loss + perceptual_loss + tv_loss

        return loss

    def generate(self, content_images, style_images, patch_size, alpha):
        content_features = self.encoder(content_images)
        style_features = self.encoder(style_images)
        style_intermediate_features = self.encoder(style_images, output_last_feature=False)[:3]
        batch_size = content_features.shape[0]
        reconstructed_features = []
        for i in range(batch_size):
            c = content_features[i].unsqueeze(0)
            s = style_features[i].unsqueeze(0)
            res = style_decorator(c, s, patch_size, alpha)
            reconstructed_features.append(res)

        reconstructed_features = torch.cat(reconstructed_features, 0)

        # # the following code are only used to test AutoEncoder
        # reconstructed_features = self.encoder(content_images)
        # style_intermediate_features = self.encoder(style_images, output_last_feature=False)[:3]
        reconstructed_images = self.decoder(reconstructed_features, style_intermediate_features)
        return reconstructed_images


# x = torch.randn(2, 3, 256, 256)
# y = torch.randn(2, 3, 256, 256)
# ae = AutoEncoder()
# ae(x, x, 5, 0.8)
