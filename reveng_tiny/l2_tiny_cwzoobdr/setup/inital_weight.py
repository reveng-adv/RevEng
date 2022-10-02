from setup_vgg import vgg16, vgg16noise
import collections
import torch

def initial(file_name, state_dict, model):
    new_keys = list(model.state_dict().keys())
    keys = list(state_dict.keys())
    new_state_dict = collections.OrderedDict()
    for i in range(len(keys)):
            new_state_dict[new_keys[i]] = state_dict[keys[i]]
    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(),'./models/cifar10/'+file_name)   
    return

model1 = vgg16()
model2 = vgg16noise(0.2,0.1)
state_dict1 = torch.load('./models/cifar10/cifar10_vgg_plain.pth')
state_dict2 = torch.load('./models/cifar10/cifar10_vgg_rse.pth')

initial('vgg16',state_dict1, model1)
initial('vgg16_rse',state_dict2, model2)

def initial_no_back_track(file_name, state_dict, model):
    new_keys = list(model.state_dict().keys())
    keys = list(state_dict.keys())
    new_state_dict = collections.OrderedDict()
    count = 0
    for i in range(len(new_keys)):
        j = i - count
        key_name = new_keys[i]
        if 'num_batches_tracked' in key_name:
            count += 1
            new_state_dict[new_keys[i]] = model.state_dict()[new_keys[i]]
            continue
        new_state_dict[new_keys[i]] = state_dict[keys[j]]
    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(),file_name)  
    return  

initial_no_back_track('vgg16',state_dict1, model1)