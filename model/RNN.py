import torch 
from torch import nn

class RNN(nn.Module): 
    def __init__(self, input_size, output_size, layer_num, hidden_size, activation) -> None: 
        super().__init__() 
        if activation == 'R': self.activation = nn.ReLU() 
        if activation == 'T': self.activation = nn.Tanh() 
        if activation == 'S': self.activation = nn.Sigmoid() 
        
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_size = input_size
        def normal(shape):
            import math
            return (torch.rand(shape) - 0.5) * 1.0 / math.sqrt(150)
        self.W_hx = nn.Parameter(normal((self.layer_num, self.input_size, self.hidden_size)))  
        self.b_hx = nn.Parameter(normal((self.layer_num, self.hidden_size)))
        self.W_hh = nn.Parameter(normal((self.layer_num, self.hidden_size, self.hidden_size))) 
        self.b_hh = nn.Parameter(normal((self.layer_num, self.hidden_size)))
        # self.W_ho = {i : nn.Linear(self.hidden_size, self.hidden_size) for i in range(0, self.layer_num)}
        self.O = nn.Linear(hidden_size, output_size) 
        
        # for x in self.W_hx.values(): 
        # self.reset_parameters(self.W_hx) 
        # for x in self.W_hh.values(): 
            # self.reset_parameters(x) 
        # for x in self.W_ho.values(): 
            # self.reset_parameters(x) 
        # self.reset_parameters(self.O) 

    
    def reset_parameters(self, x):
        # print(x)
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in x.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
            # weight = nn.Parameter(weight)
            # print(weight.requires_grad)
            # print(weight)


    def forward(self, X): 
        X = {i : X[:, :, i] for i in range(self.layer_num)}
        H = {}; O = {}
        H[0] = torch.zeros(self.hidden_size)#, requires_grad=True) # 

        for i in range(0, self.layer_num): 
            # if i == 14:
                # print(self.W_hh[i](H[i]).shape, self.W_hx[i](X[i])[13], self.W_hx[i](X[i])[17])
            H[i + 1] = self.activation(H[i] @ self.W_hh[i] + self.b_hh[i] + X[i] @ self.W_hx[i] + self.b_hx[i]) 
            # O[i] = self.activation(self.W_ho[i](H[i]))
        # print(H[3][1], H[3][2],H[3].shape)
        # print(self.O(O[self.layer_num - 1])[:5]) 
        return self.O(H[self.layer_num]).squeeze() 
