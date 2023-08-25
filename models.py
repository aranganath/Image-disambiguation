import torch
from torch import nn
from pdb import set_trace


# Only disambiguates
class Transformer(nn.Module):
    '''
        Any
    '''
    def __init__(self, 
                d_model: int, 
                encoder_feedforward: int, 
                decoder_feedforward: int,
                encoder_heads: int,
                decoder_heads: int,
                num_encoder_layers: int,
                num_decoder_layers: int,
                upscale: int
            ):
        super(Transformer,self).__init__()
        self.upscale = upscale

        self.d_model = d_model
        self.Downsampler = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, stride=1, padding=1, kernel_size=(3,3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, stride=1, padding=1, kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5,5), stride=2, padding=1),
            nn.Sigmoid()
        )

        TransformerEncoder = nn.TransformerEncoderLayer(d_model = self.upscale*d_model,nhead=encoder_heads,dim_feedforward=encoder_feedforward)
        self.Encoder = nn.TransformerEncoder(encoder_layer=TransformerEncoder, num_layers=num_encoder_layers)
        DecoderLayer = nn.TransformerDecoderLayer(d_model=self.upscale*d_model, nhead=decoder_heads, dim_feedforward=decoder_feedforward)

        self.Decoder = nn.TransformerDecoder(decoder_layer=DecoderLayer, num_layers=num_decoder_layers)

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, stride=2, padding=1 , kernel_size=(5,5)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 32, stride=1, kernel_size = (3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5,5), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, stride=1, padding=1, kernel_size=(3,3)),
            nn.BatchNorm2d(2),
            nn.Upsample(scale_factor = 28/27, mode='bilinear', align_corners=True),
        )
        
    def forward(self,x):

        shape = x.shape

        
        # First downsample to space of [Batch, 64]
        output = self.Downsampler(x)

        shape = output.shape

        # We want to retain the same shape
        output = output.reshape(-1, self.d_model)

        src = self.Encoder(output)

        output = self.Decoder(src, memory = x)

        output = output.reshape(shape)

        output = self.upsampler(output)

        output = output.reshape((-1, 2, 28, 28))

        return output


# Upsamples and disambiguates
class UpsamplingTransformer(nn.Module):
    '''
        Any
    '''
    def __init__(self, 
                d_model: int, 
                encoder_feedforward: int, 
                decoder_feedforward: int,
                encoder_heads: int,
                decoder_heads: int,
                num_encoder_layers: int,
                num_decoder_layers: int,
                upscale: int
            ):
        super(UpsamplingTransformer,self).__init__()
        self.upscale = upscale

        self.d_model = d_model
        self.Downsampler = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size = 4, stride =2, padding = 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=16, stride=1, padding=1, kernel_size=(3,3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, stride=1, padding=1, kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5,5), stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5,5), stride=1, padding=1),
            nn.Sigmoid()

        )

        TransformerEncoder = nn.TransformerEncoderLayer(d_model = self.upscale*d_model,nhead=encoder_heads,dim_feedforward=encoder_feedforward)
        self.Encoder = nn.TransformerEncoder(encoder_layer=TransformerEncoder, num_layers=num_encoder_layers)
        DecoderLayer = nn.TransformerDecoderLayer(d_model=self.upscale*d_model, nhead=decoder_heads, dim_feedforward=decoder_feedforward)

        self.Decoder = nn.TransformerDecoder(decoder_layer=DecoderLayer, num_layers=num_decoder_layers)

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=(5,5), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, stride=2, padding=1 , kernel_size=(5,5)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 32, stride=1, kernel_size = (3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5,5), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, stride=1, padding=1, kernel_size=(4,4)),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

    def forward(self,x):

        shape = x.shape

        
        # First downsample to space of [Batch, 64]
        output = self.Downsampler(x)

        shape = output.shape

        # We want to retain the same shape
        output = output.reshape(-1, self.d_model)

        src = self.Encoder(output)

        output = self.Decoder(src, memory = src)

        output = output.reshape(shape)

        output = self.upsampler(output)

        output = output.reshape((-1, 2, 28, 28))

        return output