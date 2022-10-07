import torch 
import numpy as np

def test_model(model, test_dl):
    correct = 0
    sample_sum = 0
    for test_step, (x, y) in enumerate(test_dl, 1): 
        model.eval() #? 实现

        with torch.no_grad(): 
            pred = model(x)
            correct += np.sum(torch.argmax(pred.detach(), axis=1).numpy() == y.detach().numpy()) 
            sample_sum += len(y)
            
    print("Test ACC: ", correct / sample_sum)
    return correct / sample_sum