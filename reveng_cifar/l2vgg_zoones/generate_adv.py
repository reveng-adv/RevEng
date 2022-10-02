import os
import torch
import numpy as np
import eagerpy as ep
import foolbox as fb
from tqdm import tqdm
from setup.utils import loadmodel, loaddata,randomdata
from art.attacks.evasion.carlini_ori import CarliniL2Method
from art.attacks.evasion.zoo import ZooAttack
from art.attacks.evasion.square_attack import SquareAttack
from art.attacks.evasion.boundary import BoundaryAttack
#from carlart_xw/art/attacks/evasionini import CarliniL2Method
from art.estimators.classification import PyTorchClassifier

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_adv(args, fmodel, attack, test_loader):
    count = 0
    success = []
    adv_samples = []
    label_list = []
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        images, labels = ep.astensors(*(images,labels))
        _, advs, success_batch = attack(fmodel, images, labels, epsilons=[args.epsilon])
        adv_samples.append(advs[0].raw.detach().cpu().numpy())
        success.append(success_batch[0].raw.detach().cpu().numpy()[0])
        label_list.append(labels.raw.detach().cpu().numpy()[0])
        count += images.shape[0]
        if count >= args.n_samples:
            break
    adv_samples = np.concatenate(adv_samples,axis=0)
    success = np.array(success)
    label_list = np.array(label_list)
    robust_accuracy = 1.0 - success.mean(axis=-1)
    print("Method={},num.sample={}, robust.acc={:.4f}".format(args.method, args.n_samples, robust_accuracy))
    return adv_samples, success, label_list


def get_all_samples(test_loader):
    img = []
    lab = []
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        img.append(images.detach().numpy())
        lab.append(labels.detach().numpy())
    x_test = np.concatenate(img)
    lab = np.concatenate(lab)
    y_test = np.zeros((lab.size, lab.max()+1))
    y_test[np.arange(lab.size),lab] = 1
    return x_test, y_test, lab



def main(args):
    print('==> Loading data..')
    _, test_loader, = loaddata(args)#, np.load("./features/"+args.dataset+'/indices.npy'))
    #_, test_loader, = randomdata(args, np.load("./features/"+args.dataset+'/indices.npy'))
    #np.save("./features/"+args.dataset+'/indices.npy', indices)
    
    if args.n_samples == None:
        args.n_samples = len(test_loader.dataset)
    print('==> Loading model..')
    model = loadmodel(args)
    model = model.to(device)
    model = model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=args.input_shape,
        nb_classes=10,
        )
    
    if args.method == 'CW':
        attack = CarliniL2Method(classifier=classifier,max_iter=args.cw_steps)
    if args.method == 'ZOO':
        attack = ZooAttack(classifier=classifier, max_iter=args.zoo_query, use_resize=False, use_importance=False)
    if args.method == 'SQUARE':
        attack = SquareAttack(estimator=classifier, eps=1, max_iter=args.square_query, norm=2)
    if args.method == 'BOUNDARY':
        attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=args.boundary_query)
    
    x_test, y_test, labels = get_all_samples(test_loader)
    predictions = classifier.predict(x_test)
    success = (np.argmax(predictions, axis=1) != np.argmax(y_test,axis=1))
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Base accuracy on all 10k: {:.4f}".format(accuracy))
    #print(np.sum(np.argmax(y_test,axis=1)==labels))
    
    x_test = x_test[:args.n_samples]
    y_test = y_test[:args.n_samples]
    labels = labels[:args.n_samples]
    #x_test = x_test[args.n_samples:2*args.n_samples]
    #y_test = y_test[args.n_samples:2*args.n_samples]
    #labels = labels[args.n_samples:2*args.n_samples]
    #x_test = x_test[2*args.n_samples:3*args.n_samples]
    #y_test = y_test[2*args.n_samples:3*args.n_samples]
    #labels = labels[2*args.n_samples:3*args.n_samples]
    #x_test = x_test[3*args.n_samples:4*args.n_samples]
    #y_test = y_test[3*args.n_samples:4*args.n_samples]
    #labels = labels[3*args.n_samples:4*args.n_samples]
    
    predictions_test = classifier.predict(x_test)
    success = (np.argmax(predictions_test, axis=1) != labels)
    accuracy = np.sum(np.argmax(predictions_test, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Base accuracy on current samples: {:.4f}".format(accuracy))

    adv_samples = attack.generate(x=x_test)
    predictions_adv = classifier.predict(adv_samples)
    success = (np.argmax(predictions_adv, axis=1) != labels) & (np.argmax(predictions_test,axis=1) == labels)
    accuracy = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        
    #print(np.sum(success)/len(success))
    distortion = []
    for i in range(len(adv_samples)):
        distortion.append(np.sum((adv_samples[i]-x_test[i])**2)**.5)
        print("Total distortion:", i, distortion[-1])
    print('max',np.max(distortion))
    print('mean',np.mean(distortion))
    print("median",np.median(distortion))
    
    if args.method =='CW':
        print("Query:{}, and Robust Accuracy: {:.4f}".format(args.cw_steps, accuracy))
        #np.save('./features/cifar10/distortion_cw1',distortion)
    if args.method =='ZOO':
        print("Query:{}, and Robust Accuracy: {:.4f}".format(args.zoo_query, accuracy))
        #np.save('./features/cifar10/distortion_zoo1',distortion)
    if args.method =='SQUARE':
        print("Query:{}, and Robust Accuracy: {:.4f}".format(args.square_query, accuracy))
        #np.save('./features/cifar10/distortion_square1',distortion)
    if args.method =='BOUNDARY':
        print("Query:{}, and Robust Accuracy: {:.4f}".format(args.boundary_query, accuracy))
        #np.save('./features/cifar10/distortion_boundary1',distortion)

    np.save("./features/"+args.dataset+'/'+args.method+'_adv1', adv_samples)
    np.save("./features/"+args.dataset+'/'+args.method+'_labels1', labels)
    np.save("./features/"+args.dataset+'/'+args.method+'_success1', success)
    np.save('./features/'+args.dataset+'/'+args.method+'_distortion1',distortion)
    np.save('./features/'+args.dataset+'/'+args.method+'_pred_test1',predictions_test)
    np.save('./features/'+args.dataset+'/'+args.method+'_pred_adv1',predictions_adv)
    # adv_samples = np.load("./features/"+args.dataset+'/'+args.method+'_adv.npy')
    # labels = np.load("./features/"+args.dataset+'/'+args.method+'_labels.npy')
    # success = np.load("./features/"+args.dataset+'/'+args.method+'_success.npy')
       
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Model Evaluation')
    parser.add_argument("-d", '--dataset', type=str, choices=["mnist", "cifar10"], default="cifar10")   
    parser.add_argument("--method", type=str, default="CW", choices=['ZOO','BOUNDARY','CW','SQUARE'])
    parser.add_argument("--root", type=str, default="D:/yaoli")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--cw_steps", type=int, default=10)
    parser.add_argument("--zoo_query",type = int, default = 100)
    parser.add_argument("--square_query",type = int, default = 10000)
    parser.add_argument("--boundary_query",type = int, default = 500)
    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.batch_size = 1
        args.init = "/cnn"
        args.epsilon = 0.3
        args.steps = 40
        args.alpha = 0.03
        args.input_shape = (1,28,28)
    elif args.dataset == 'cifar10':
        args.batch_size = 1
        args.init = "/vgg16"
        args.epsilon =.031
        args.steps = 20
        args.alpha = 0.003
        args.input_shape = (3,32,32)
    else:
        print('invalid dataset')
    print(args)
    main(args)
