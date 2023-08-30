import torch
from torch import nn
from pdb import set_trace
import os


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

        output = self.Decoder(src, memory = src)

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

        output = self.Decoder(src, memory = output)

        output = output.reshape(shape)

        output = self.upsampler(output)

        output = output.reshape((-1, 2, 28, 28))

        return output
    


class TransferredCIFAR10(nn.Module):
    def __init__(self, gray= False, load_dict=True, PATH='./checkpoints/DisambDenoiseCheckpoint99.pt') -> None:
        super(TransferredCIFAR10, self).__init__()
        # Do a downsampler which will bring the network from 32 \times 32 to 28 \times 28
        
        if not gray:
            self.Downsampler = nn.Conv2d(3, 1, kernel_size=(5,5), stride=1)

            self.Upsampler = nn.Sequential(
                nn.ConvTranspose2d(2,4, kernel_size=(3,3), stride=1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.ConvTranspose2d(4,6, kernel_size=(3,3), stride = 1),
                nn.BatchNorm2d(6),
                nn.ReLU(),
            )
        else:
            self.Downsampler = nn.Conv2d(1, 1, kernel_size=(5,5), stride=1)
            
            self.Upsampler = nn.Sequential(
                nn.ConvTranspose2d(2,2, kernel_size=(3,3), stride=1),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(2,2, kernel_size=(3,3), stride = 1),
                nn.BatchNorm2d(2),
                nn.Sigmoid()
            )

        transformer = Transformer(d_model=288,
                    encoder_feedforward=2048,
                    decoder_feedforward=2048,
                    encoder_heads=144,
                    decoder_heads=144,
                    num_decoder_layers=1,
                    num_encoder_layers=1,
                    upscale=1
                )
        if load_dict:
            if not os.path.isfile(PATH):
                raise RuntimeError(PATH + ' does not exist')
            
            transformer.load_state_dict(torch.load(PATH))


        self.Transformer = transformer

        if load_dict:
            self.Transformer.requires_grad_(False)
    
    def forward(self, input):
        output = self.Downsampler(input)
        output = self.Transformer(output)
        output = self.Upsampler(output)

        return output


# RNN competing model
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        # Without downsample : input shape [batch, 1, 28, 28]
        # With downsample : input shape [batch, 1, 14, 14]
        self.Encoder = nn.Sequential(
            # batchnorm
            nn.Conv2d(1,32,kernel_size=2, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size =3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size =5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size =3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size =5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,kernel_size =3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,kernel_size =3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,8,kernel_size =3, padding=1, stride=1),
            nn.ReLU())

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(8,64,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,128,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,32,kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32,16,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=2,  stride=1),
            nn.LeakyReLU())

    
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

        self.disambiguate1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7, padding=3, stride=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, padding=3, stride=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, padding=3, stride=3, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.Sigmoid()
        )
        
        self.disambiguate2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7, padding=3, stride=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, padding=3, stride=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, padding=3, stride=3, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self,x, flag=True):
        outputs = []
        if flag:
            batch  = x[0].shape[0]
            output = self.Encoder[:5](x[0])
            output = self.Encoder[6](output + self.Encoder[5](output))
            output = self.Encoder[7](output)
            output = self.Encoder[9:](self.Encoder[8](output) + output)
            output = self.Decoder[:5](output)
            output = self.Decoder[6](output + self.Decoder[5](output))
            output = self.Decoder[7](output)
            output = self.Decoder[9:](self.Decoder[8](output) + output)
            
            output = self.upsample[:](output)
            output = torch.flatten(output, 2)
            outputs.append(output)

            output = self.disambiguate1[:](x[1])
            I1 = output
            output = torch.flatten(output, 2)
            outputs.append(output.reshape(((batch, 1,784))))
            
            output = self.disambiguate2[:](x[1])
            output = torch.flatten(output, 2)
            outputs.append(output.reshape(((batch, 1,784))))
            
            
        else:
            
            
            batch  = x.shape[0]
            output = self.Encoder[:5](x)
            output = self.Encoder[6](output + self.Encoder[5](output))
            output = self.Encoder[7](output)
            output = self.Encoder[9:](self.Encoder[8](output) + output)
            output = self.Decoder[:5](output)
            output = self.Decoder[6](output + self.Decoder[5](output))
            output = self.Decoder[7](output)
            output = self.Decoder[9:](self.Decoder[8](output) + output)
            output = self.upsample[:](output)
            upsampled = output
            output = torch.flatten(output, 2)
            outputs.append(output)
            output = self.disambiguate1[:](upsampled)
            I1 = output
            output = torch.flatten(output, 2)
            outputs.append(output.reshape(((batch, 1,784))))
            
            output = self.disambiguate2[:](upsampled)
            output = torch.flatten(output, 2)
            outputs.append(output.reshape(((batch, 1,784))))
            
            
        return outputs