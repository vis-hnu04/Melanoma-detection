import torch.nn as nn
import pretrainedmodels
import torch

class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        #print (self.base_model)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)
    
    def forward(self, image):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = self.l0(x)
      #  loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))
        out = torch.sigmoid(out)
        return out

