import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
              # Use sigmoid if your input data is normalized between 0 and 1
        )
    
    def forward(self, x):
        # Pass through encoder
        encoded = self.encoder(x)
        # Pass through decoder
        decoded = self.decoder(encoded)
        return decoded




class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder_manipulated = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        

        self.encoder_error = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 🛠️ Solución agregando padding
        )


        self.middle = nn.Sequential(
            nn.Conv2d(64+64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, e):
        x1 = self.encoder_manipulated(x)
        e1 = self.encoder_error(e)
        import torch.nn.functional as F
        e1 = F.interpolate(e1, size=(x1.shape[2], x1.shape[3]), mode="bilinear", align_corners=False)

        if e1.shape[3] > x1.shape[3]:
            e1 = e1[:, :, :, :x1.shape[3]] 
        elif e1.shape[3] < x1.shape[3]:
            e1 = F.pad(e1, (0, x1.shape[3] - e1.shape[3]))  
        cat = torch.cat((x1, e1), dim=1)
        x2 = self.middle(cat)
        x3 = self.decoder(x2)

        out = self.relu(x-e) + x3
        out = self.sigmoid(out)
        return out
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class AUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AUNet, self).__init__()
        
        self.encoder_manipulated = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_error = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.middle = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, e):
        x1 = self.encoder_manipulated(x)
        e1 = self.encoder_error(e)
        
        cat = torch.cat((x1, e1), dim=1)
        x2 = self.middle(cat)
        x3 = self.decoder(x2)
        
        out = self.sigmoid(x - e + x3)
        return out

import torch
import torch.nn as nn

class ARModelDetection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        """
        in_channels: número de canales por línea (C)
        out_channels: tamaño de la firma (C)
        La entrada al modelo se concatenará, resultando en 2C canales.
        """
        super(ARModelDetection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels*2, hidden_channels, kernel_size=3, padding=1)
        self.conv_expand = nn.Conv2d(120, 400, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, current_line, prev_line):
        # current_line, prev_line: (B, C, W) -- cada línea es 1xW con C bandas
        # Concatenar a lo largo del canal: (B, 2C, W)
        x = torch.cat([prev_line, current_line], dim=1)
        # Añadimos una dimensión de altura: (B, 2C, 1, W)
        if x.dim() == 3:
            x = x.unsqueeze(2)
        x = self.conv_expand(x) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # (B, hidden_channels, 1, 1)
        x = x.view(x.size(0), -1)  # (B, hidden_channels)
        out = self.fc(x)  # (B, out_channels) -> Estimación de la firma espectral
        return out

import torch
import torch.nn as nn
import torch.optim as optim

class AutoRegressiveReconstructionModel(nn.Module):
    def __init__(self, input_size=200, hidden_size=256, output_size=200):
        super(AutoRegressiveReconstructionModel, self).__init__()

        # Capa convolucional para procesar las entradas de forma espectral
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=200, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(200, 60, kernel_size=3, padding=1)  # Salida esperada: [1, 60, 200]
        
        # LSTM para procesamiento autoregresivo (secuencia)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Capa de salida
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, manipulated_line, prev_line):
        # Aseguramos que ambas líneas tengan 4 dimensiones
        if manipulated_line.dim() == 3:
            manipulated_line = manipulated_line.unsqueeze(1)  # Añadir dimensión de canal (batch_size, 1, 120, 200)
        
        if prev_line.dim() == 3:
            prev_line = prev_line.unsqueeze(1)  # Añadir dimensión de canal (batch_size, 1, 120, 200)
        
        # Concatenamos las dos líneas espectrales (manipulada y previa)
        x = torch.cat((manipulated_line, prev_line), dim=1)  # Formato: [batch_size, 120, 200]
        
        # Ahora podemos aplicar la convolución y el resto del modelo
        x = self.conv1(x)  # Capa convolucional
        x = self.conv2(x)  # Capa convolucional final
        x = x.mean(dim=1, keepdim=True)  # Promediamos sobre el canal; ahora x es [1, 1, 60, 200]
        x = x.squeeze(1)  # Ahora x es [1, 60, 200]
        # Reorganizamos a [batch_size, seq_len, features] -> [1, 60, 200]

        # LSTM: procesamos la secuencia autoregresiva
        lstm_out, (hn, cn) = self.lstm(x)  # LSTM procesa la secuencia (batch_size, seq_len, hidden_size)

        # Reconstruimos la salida de la línea espectral
        reconstructed_line = self.fc(lstm_out)   # Eliminamos la dimensión de canal, ahora tenemos [batch_size, seq_len, 200]
         # Salida final: [batch_size, seq_len, output_size]
        return reconstructed_line
