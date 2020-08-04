from torch.utils.data import Dataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import torch
import cv2


class MelenomaDataset(Dataset):
    def __init__(self, img_path, df,transform = None):
        self.image_path = img_path
        
        self.data_frame = df
        self.transforms = transform
    def __len__(self):
        return (len(self.data_frame))
   
    def __getitem__(self, index):
       # print(df["image_name"][index])
        path = os.path.join(self.image_path, self.data_frame["image_name"][index]+".png")
        
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image /= 255.0

        
        label = self.data_frame["target"][index]
        if self.transforms is not None:
            image = self.transforms(image)
        
        return {"image" :image,"label" : label}
        

"""        
use_gpu = torch.cuda.is_available()
root_dir = "../input/siic-isic-224x224-images/train/"

train_dataset = MelenomaDataset(root_dir,df)
image , label = train_dataset.__getitem__(1)
#print(image.shape)
for ind,pack in enumerate(train_dataset):
    data = pack["image"]  
        # print(data.shape)
    #data=data.numpy()
    #print(data.shape)
    # data=data.reshape(data.shape[1],data.shape[2],data.shape[0])
    data = data.permute(1, 2, 0)
   # print(data.shape)
    plt.imshow(data)
    plt.show()       
    """


