import numpy as np
def combine(method = "CW"):
    path = "./features/cifar10/"
    path += method
    
    a1 = np.load(path+"_adv1.npy")
    a2 = np.load(path+"_adv2.npy")
    a3 = np.load(path+"_adv3.npy")
    a4 = np.load(path+"_adv4.npy")
    aa = np.concatenate((a1,a2,a3,a4))
    #print(aa.shape)
    np.save(path+"_adv10k.npy",aa)
    
    a1 = np.load(path+"_distortion1.npy")
    a2 = np.load(path+"_distortion2.npy")
    a3 = np.load(path+"_distortion3.npy")
    a4 = np.load(path+"_distortion4.npy")
    aa = np.concatenate((a1,a2,a3,a4))
    print(aa.shape)
    np.save(path+"_distortion10k.npy",aa)
    
    a1 = np.load(path+"_pred_adv1.npy")
    a2 = np.load(path+"_pred_adv2.npy")
    a3 = np.load(path+"_pred_adv3.npy")
    a4 = np.load(path+"_pred_adv4.npy")
    aa = np.concatenate((a1,a2,a3,a4))
    print(aa.shape)
    np.save(path+"_pred_adv10k.npy",aa)
    
    a1 = np.load(path+"_pred_test1.npy")
    a2 = np.load(path+"_pred_test2.npy")
    a3 = np.load(path+"_pred_test3.npy")
    a4 = np.load(path+"_pred_test4.npy")
    aa = np.concatenate((a1,a2,a3,a4))
    print(aa.shape)
    np.save(path+"_pred_test10k.npy",aa)
    
    a1 = np.load(path+"_success1.npy")
    a2 = np.load(path+"_success2.npy")
    a3 = np.load(path+"_success3.npy")
    a4 = np.load(path+"_success4.npy")
    aa = np.concatenate((a1,a2,a3,a4))
    print(aa.shape)
    np.save(path+"_success10k.npy",aa)
    
    a1 = np.load(path+"_labels1.npy")
    a2 = np.load(path+"_labels2.npy")
    a3 = np.load(path+"_labels3.npy")
    a4 = np.load(path+"_labels4.npy")
    aa = np.concatenate((a1,a2,a3,a4))
    print(aa.shape)
    np.save(path+"_labels10k.npy",aa)


def triple_true():
    path = "./features/cifar10/"
    
    p_test = np.load(path+"CW_pred_test10k.npy")
    labels = np.load(path+"CW_labels10k.npy")
    #sq_p_test = np.load(path+"SQUARE_pred_test10k.npy")
    #bdr_p_test = np.load(path+"BOUNDARY_pred_test10k.npy")
    cw_p_adv = np.load(path+"CW_pred_adv10k.npy")
    zoo_p_adv = np.load(path+"ZOO_pred_adv10k.npy")
    bdr_p_adv = np.load(path+"BOUNDARY_pred_adv10k.npy")
    
    cw_s = (np.argmax(cw_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)
    #n = np.load("./CW_success10k.npy")
    zoo_s = (np.argmax(zoo_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)
    bdr_l2 = np.load(path+"BOUNDARY_distortion10k.npy")
    bdr_s = (bdr_l2 <=1) &(np.argmax(bdr_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)

    #bdr_s = (np.argmax(bdr_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)
    
    ind = cw_s & zoo_s & bdr_s
    np.save(path+"ind_triple_true",ind)
    print(np.max(np.load(path+"CW_distortion10k.npy")[ind]))
    print(np.max(np.load(path+"ZOO_distortion10k.npy")[ind]))
    print(np.max(np.load(path+"BOUNDARY_distortion10k.npy")[ind]))

def split(): 
    '''
    concatenate first and then split
    '''
    path = "./features/cifar10/"
    ind = np.load(path+"ind_triple_true.npy")
    '''
    cw_adv = np.load(path+"CW_adv10k.npy")[ind]
    zoo_adv = np.load(path+"SQUARE_adv10k.npy")[ind]
    bdr_adv = np.load(path+"BOUNDARY_adv10k.npy")[ind]
    cw_pertb = np.load(path+"CW_adv10k.npy")[ind] - np.load(path+"x_test.npy")[ind]
    sq_pertb = np.load(path+"SQUARE_adv10k.npy")[ind] - np.load(path+"x_test.npy")[ind]
    bdr_pertb = np.load(path+"BOUNDARY_adv10k.npy")[ind] - np.load(path+"x_test.npy")[ind]
    '''
    methods = ['CW', 'ZOO','BOUNDARY']
    comb = []
    pertb =[]
    attk_method = []
    label = []
    i= 0
    for method in methods:
        comb_data = np.load(path+method+"_adv10k.npy")
        x_data = np.load(path+"x_test.npy")[:comb_data.shape[0]][ind]
        comb_data = comb_data[ind]
        lab = np.load(path+method+"_labels10k.npy")[ind]
        comb.append(comb_data)
        pertb_data = comb_data-x_data
        pertb.append(pertb_data)
        attk_method.append(np.repeat(i, comb_data.shape[0]))
        label.append(lab)
        i += 1

    comb = np.concatenate(comb)
    pertb= np.concatenate(pertb)
    cate = np.concatenate(attk_method)
    label = np.concatenate(label)

    print('comb.shape', comb.shape)
    
    print('pertb.shape', pertb.shape)
    print('cate.shape', cate.shape)
    print('label.shape',label.shape)
    
    np.save("./features/combined/comb.npy",comb)
    np.save("./features/combined/pertb.npy",pertb)
    np.save("./features/combined/label.npy",label)
    np.save("./features/combined/cate.npy",cate)
    
    
    ##split 
    np.random.seed(2020)
    n = np.sum(ind)
    ind_train = np.random.choice(n,int(n*0.7),replace= False)
    np.save("./features/cifar10/ind_train.npy",ind_train)
    mask = np.random.rand(n) >1
    mask[ind_train] = True
    
    mask = np.concatenate([list(mask),list(mask),list(mask)])
    
    adv_train = comb[mask]
    adv_test = comb[~mask]
    pertb_train = pertb[mask]
    pertb_test = pertb[~mask]
    cate_train = cate[mask]
    cate_test = cate[~mask]
    label_train = label[mask]
    label_test = label[~mask]
    np.save("./features/cifar10/train_adv.npy",adv_train)
    np.save("./features/cifar10/test_adv.npy",adv_test)
    np.save("./features/cifar10/train_pertb.npy",pertb_train)
    np.save("./features/cifar10/test_pertb.npy",pertb_test)
    np.save("./features/cifar10/train_label.npy",label_train)
    np.save("./features/cifar10/test_adv.npy",label_test)
    np.save("./features/cifar10/train_cate.npy",cate_train)
    np.save("./features/cifar10/test_cate.npy",cate_test)
    

    np.save("./features/combined/comb_train.npy",adv_train)
    np.save("./features/combined/comb_test.npy",adv_test)
    np.save("./features/combined/pertb_train.npy",pertb_train)
    np.save("./features/combined/pertb_test.npy",pertb_test)
    np.save("./features/combined/label_train.npy",label_train)
    np.save("./features/combined/label_test.npy",label_test)
    np.save("./features/combined/cate_train.npy",cate_train)
    np.save("./features/combined/cate_test.npy",cate_test)


def main():
    # combine batch 1,2,3,4 for different attacks respectively
    combine("CW")
    combine("ZOO")
    combine("BOUNDARY")
    
    # select triple success 
    triple_true()
    
    # Concatenate triple true, and split train/test in terms of adv, pertb, add labels for diff attacks
    split()
  

if __name__=="__main__":
    main()
    

