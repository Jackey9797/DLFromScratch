import gzip
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Sampler

root_path = "data/" 
file_name = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"), 
             "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}


def get_data(data_type): 
    image_file = root_path + file_name[data_type][0] 
    label_file = root_path + file_name[data_type][1] 

    with gzip.open(label_file, "rb") as f: 
        f.read(8) 
        labels = np.frombuffer(f.read(), dtype=np.uint8) 

    with gzip.open(image_file, "rb") as f: 
        f.read(16) 
        content = np.frombuffer(f.read(), dtype=np.uint8) 
        images = content.reshape(-1, 28, 28)

    return images, labels

# plt.imshow(x_test[2])


class MNIST: 
    def __init__(self, data_type, transform=None, inbalance=False):
        super().__init__() 
        self.data, self.labels = get_data(data_type)  
        if inbalance == True : # 构造类别不平衡数据
            label_idx = {}; label_num = {}  # 每个数字对应的数据下标；构造的inbalance数据每个数字样本个数 
            for i in range(0, 10): 
                label_idx[i] = [j for j in range(0, len(self.labels)) if self.labels[j] == i] 
                # print(len(label_idx[i]))
                label_num[i] = int(len(label_idx[i]) / (i % 3 * 2 + 1))
                label_idx[i] = label_idx[i][:label_num[i]] # 每个数字样本集取前num个 
                # print(len(label_idx[i]), label_num[i])


            new_idx = []
            for i in range(0, 10): 
                new_idx += label_idx[i]
            # print(new_idx, len(new_idx))
            # print(np.array(self.data).shape)
            # print(np.array(self.labels).shape)
            self.data = np.array(self.data)[new_idx]
            self.labels = np.array(self.labels)[new_idx]
            
            # print(np.array(self.data).shape)
            # print(np.array(self.labels).shape)
            # 重新索引
        self.data = torch.tensor(self.data, dtype=torch.float)
        self.num_classes = 10
        self.labels = torch.tensor(self.labels, dtype=torch.long) # ? float & float64
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if self.transform : 
            img = self.transform(img)
        img = img.unsqueeze(0)
        # print(label)
        return img, label 


class CustomSampler(Sampler):
    def __init__(self, data, resampling=False, shuffle=False):
        self.data = data
        self.shuffle = shuffle
        self.resampling = resampling
        print("test:", self.data)

    def __iter__(self): 
        indices = list(range(0, len(self.data)))
        max_L = 0
        if self.resampling == True: 
            idx = {}
            for i in range(self.data.num_classes): 
                idx[i] = np.where(self.data.labels == i) 
                idx[i] = idx[i][0]
                max_L = max(max_L, len(idx[i])) 

            for i in range(self.data.num_classes): 
                idx_ex = np.random.choice(idx[i], max_L - len(idx[i])) 
                # print(idx_ex.shape) 
                idx[i] = np.concatenate((idx[i], idx_ex), axis=0)  
                
            indices = np.concatenate([idx[i] for i in range(self.data.num_classes)], axis=0).tolist()
            # print(indices.shape)
            

        if self.shuffle == True: np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data)

# dl = DataLoader(test_ds, batch_size=256, sampler=CustomSampler(test_ds,resampling=True, shuffle=True)) 

# cnt = 0 

# for i in dl:
#     cnt += len(i[1]) 
#     # print(i[1])

# print("sample num: " + str(cnt))

# for i in dl: 
#     print(i[1])
#     break

def show_dataset(dataset):
    col = 3; row = 4
    fig = plt.figure(figsize=(9, 12))

    j = 0
    for i in range(col * row):
        idx = j
        j += 1
        
        x = dataset[idx][0];  y = dataset[idx][1].item()
        fig.add_subplot(row, col, i + 1)
        plt.axis("off")
        plt.imshow(x.squeeze())
        plt.title(y)