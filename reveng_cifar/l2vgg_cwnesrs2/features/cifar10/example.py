import numpy as np
#a = np.load("./indices.npy")
#print(np.max(a))
#print(a.shape)
#a = np.load("./BOUNDARY_distortion10k.npy")
#print("bdrmaxl2",np.max(a))
#a = np.load("./SQUARE_distortion10k.npy")
#print("squaremaxl2",np.max(a))
a = np.load("./CW_distortion1.npy")
print("cwmaxl2",np.max(a))
print(len(a))
print(a)
print("cwsuccess",np.sum(np.where(a!=0)))
a = np.load("./CWrs_distortion10k.npy")
print("cwmaxl2",np.max(a),np.sum(a<5))
path = "./"
p_test = np.load(path+"CW_pred_test10k.npy")
labels = np.load(path+"CW_labels10k.npy")
#    #sq_p_test = np.load(path+"SQUARE_pred_test10k.npy")
#    #bdr_p_test = np.load(path+"BOUNDARY_pred_test10k.npy")
cw_p_adv = np.load(path+"CW_pred_adv10k.npy")
cwnes_p_adv = np.load(path+"CWnes_pred_adv10k.npy")
cwrs_p_adv = np.load(path+"CWrs_pred_adv10k.npy")

#    cw_l2 = np.load(path+"CW_distortion10k.npy")
#    cw_s = (cw_l2 <=5.001)& (np.argmax(cw_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)
#n = np.load("./CW_success10k.npy")
#cwnes_l2 = np.load(path+"CWnes_distortion10k.npy")
#cwnes_s = (cwnes_l2 < 5.001)&(np.argmax(cwnes_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)
cwrs_l2 = np.load(path+"CWrs_distortion10k.npy")
cwrs_s = (cwrs_l2 <=10) &(np.argmax(cwrs_p_adv, axis=1) != labels) & (np.argmax(p_test,axis=1) == labels)
print("hh",np.sum(cwrs_s),np.sum(cwrs_l2<10))
#b = np.load("./CW_adv.npy")
#print("CW_adv",b.shape)
h3 = np.load("./CW_success1.npy")
print("CW_success",np.sum(h3))
h3 = np.load("./CW_success2.npy")
print("CW_success",np.sum(h3))
h3 = np.load("./CW_success3.npy")
print("CW_success",np.sum(h3))
h3 = np.load("./CW_success4.npy")
print("CW_success",np.sum(h3))
c  = np.load("./PGD_adv.npy")
print("PGD",c.shape)

d = np.load("./FGSM_adv.npy")
print("FGSM_adv",d.shape)

e = np.load("./feat_FGSM.npy")
print("feat_FGSM", e.shape)

f = np.load("./feat_CW.npy")
print("feat_CW", f.shape)

g = np.load("./feat_PGD.npy")
print("feat_PGD", g.shape)

h = np.load("./feat_natural.npy")
print("feat_natural.npy",h.shape)
noise = g-h
print("noise.shape",noise.shape)

h = np.load("./adv_test.npy")
print("adv_test.npy",h.shape)

h  = np.load("./adv_train.npy")
print("adv_train",h.shape)
h1 = np.load("./FGSM_success.npy")
print("FGSM_success",np.sum(h1))

h2 = np.load("./PGD_success.npy")
print("PGD_success",np.sum(h2))

h3 = np.load("./CW_success.npy")
print("CW_success",np.sum(h3))

h4 = h1&h2&h3
print("triple_true",np.sum(h4))

#aa = np.load("./feat_CW_pertb.npy")
#print("feat_CW_pertb", aa.shape)

aaa = np.load("./pertb_CW.npy")
print("CW_pertb", aaa.shape)

#aa = np.load("./feat_PGD_pertb.npy")
#print("feat_PGD_pertb", aa.shape)

aaa = np.load("./pertb_PGD.npy")
print("PGD_pertb", aaa.shape)

#aa = np.load("./feat_FGSM_pertb.npy")
#print("feat_FGSM_pertb", aa.shape)

aaa = np.load("./pertb_FGSM.npy")
print("FGSM_pertb", aaa.shape)
a = np.load("./pertb_natural_labels.npy")
b = np.load("./FGSM_labels.npy")
c = np.load("./PGD_labels.npy")
d = np.load("./CW_labels.npy")
print(np.sum(c-b))
print(np.sum(d-c))
#print(np.sum(a-d))
print(a[230])
print(b[230])
'''
    aa = np.load("./feat_PGD_pertb.npy")
    print("feat_PGD_pertb", aa.shape)

    aaa = np.load("./PGD_pertb.npy")
    print("PGD_pertb", aaa.shape)

    aa = np.load("./feat_FGSM_pertb.npy")
    print("feat_FGSM_pertb", aa.shape)

    aaa = np.load("./FGSM_pertb.npy")
    print("FGSM_pertb", aaa.shape
            ''' 
