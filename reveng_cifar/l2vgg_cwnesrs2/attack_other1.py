import torch
import numpy as np
from setup.utils import loadmodel
from setupcw.utils import loaddata

from art.attacks.evasion import SquareAttack, ZooAttack, BoundaryAttack
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10


def attackmodel(args, classifier, x_test, y_test, queries):
    acc = []
    for num_query in queries:
        if args['method'] == 'square':
            attack = SquareAttack(estimator=classifier, eps=args['epsilon'], max_iter=num_query, norm=2)
        elif args['method'] == 'zoo':
            attack = ZooAttack(classifier=classifier, max_iter=num_query, use_resize=False, use_importance=False)
        elif args['method'] == 'boundary':
            attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=num_query)
        else:
            print("wrong method")
        x_test_adv = attack.generate(x=x_test)
        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Query:{}, and Accuracy: {:.4f}".format(num_query, accuracy))
        acc.append(accuracy)
    return x_test_adv,acc


def main(args):
    print('==> Loading data..')
    if args['dataset'] == 'mnist':
        (_, _), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        input_shape = (1,28,28)
    else:
        
        (_,_),(x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        input_shape = (3,32,32)

    
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    b = 2500
    x_test = x_test[:b]
    y_test = y_test[:b]
    #x_test = x_test[k:2k]
    #x_test = x_test[2k:3k]
    #x_test = x_test[3k:4k]
    print(x_test.shape)
    print('==> Loading model..')
    model = loadmodel(args)
    model = model.cuda()
    model = model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=10,
        )

    predictions = classifier.predict(x_test)
    clean_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign all test examples: {}%".format(clean_accuracy * 100)) 
    predictions = classifier.predict(x_test[:args['n_samples']])
    clean_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:args['n_samples']], axis=1)) / len(y_test[:args['n_samples']])
    print("Accuracy on benign test examples: {}%".format(clean_accuracy * 100))
    
    print("==> Evaluate the classifier on adversarial test examples")
    #queries = [100,200,500]
    queries = [100]
    x_test_adv,acc = attackmodel(args, classifier, x_test[:args['n_samples']], y_test[:args['n_samples']], queries)
    #np.save("./pgd_results/"+args['dataset']+args['save'],np.array(acc))
    print("The adjusted accuracies are:")
    print(acc)
    adv_examples = x_test_adv#.cpu().data.numpy()
    np.save("./features/cifar10/"+args['method']+'_adv'+args['k'], adv_examples)
    distortion = []
    adv_samples = adv_examples
    for i in range(len(adv_samples)):
        #print("Total distortion:", np.sum((adv_samples[i]-x_test[i])**2)**.5)
        distortion.append(np.sum((adv_samples[i]-x_test[i])**2)**.5)
        print("Total distortion:", i, distortion[-1])
    print('max',np.max(distortion))
    print('mean',np.mean(distortion))
    print('median',np.median(distortion))
    np.save('./features/cifar10/distortion_'+args['method']+args['k'],distortion)
    predictions = classifier.predict(adv_samples)
    success = (np.argmax(predictions, axis=1) != y_test[:args['n_samples']])
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:args['n_samples']], axis=1)) / len(y_test[:args['n_samples']])
    print(accuracy)
    np.save("./features/"+args['dataset']+'/'+args['method']+'_success'+args['k'], success)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Model Evaluation')
    parser.add_argument("-d", '--dataset', type=str, choices=["mnist", "cifar10"], default="mnist")   
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--method", type=str, default="square", choices=['zoo','square','boundary'])
    parser.add_argument("--root", type=str, default="D:/yaoli")#required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--k", type=str, default="1")
    args = vars(parser.parse_args())
    args['save'] = '/acc_' + args['method']
    if args['dataset'] == 'mnist':
        args['init'] = "/m_cnn"
        args['epsilon'] = 0.3
    elif args['dataset'] == 'cifar10':
        args['init'] = "/vgg16"
        args['epsilon'] =0.5# 0.03
    else:
        print('invalid dataset')
    print(args)
    main(args)
