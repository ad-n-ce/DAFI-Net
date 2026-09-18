import torch
import torch.nn as nn


class IFFTLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, kspace):

        real = kspace[:, 0]
        imag = kspace[:, 1]

        
        k_complex = torch.complex(
            real,
            imag
        )

        

        k_complex = torch.fft.ifftshift(
            k_complex,
            dim=(-2, -1)
        )

        image = torch.fft.ifft2(
            k_complex
        )

        image = torch.fft.fftshift(
            image,
            dim=(-2, -1)
        )

        image = torch.abs(
            image
        )
        

        image = image / (
            image.amax(
                dim=(-2, -1),
                keepdim=True
            ) + 1e-8
        )
        
        image = image.unsqueeze(1)

        return image
