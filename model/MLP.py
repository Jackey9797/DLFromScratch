
from torch import nn

class MLP(nn.Module): 
    def __init__(self, input_size, output_size, layer_num, hidden_size, activation): ##TODO multi参数删了
        super().__init__()

        if activation == 'S' :
            self.activation = nn.Sigmoid() 
        elif activation == 'R': 
            self.activation = nn.ReLU() 
        
        if layer_num == 0: 
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_size, output_size), self.activation)
        else : 
            self.layer_list = [nn.Flatten(), nn.Linear(input_size, hidden_size), self.activation]

            for i in range(layer_num - 1): 
                self.layer_list += [nn.Linear(hidden_size, hidden_size), nn.Dropout(p=0.2), self.activation]

            self.layer_list += [nn.Linear(hidden_size, output_size), self.activation]
            self.net = nn.Sequential(*self.layer_list) 

    def info(self):
        for i, layer in enumerate(self.net): 
            print(("第%d层: " + layer._get_name()) % i) #可以删了， 展示还是建议用原来函数及 keras包 

    def forward(self, x): 
        return self.net(x) 
