import numpy as np
np.set_printoptions(threshold=np.nan)

t_train = np.load('data/train_t.npy')
# t_test = np.load('data/test_t.npy')
# s_train = np.load('data/train_s.npy')
# s_test = np.load('data/test_s.npy')

# t_x = np.reshape(t_train[:, 1:694], [-1, 33, 21])
# t_y = t_train[:, 0]
# np.save('data/s_te_x', s_x)
# np.save('data/s_te_y', s_y)
# print(np.shape(t_x))

label_0=[]
label_1=[]

for i, data in enumerate(t_train):
    if data[0]==0: label_0.append(i)
    elif data[0]==1: label_1.append(i)

print(len(label_0), len(label_1))
data_0 = np.array(t_train[label_0])
data_1 = np.array(t_train[label_1])

shuffle_indexi_0 = np.random.permutation(len(data_0))
shuffle_indexi_1 = np.random.permutation(len(data_1))

data_0 = data_0[shuffle_indexi_0]
data_1 = data_1[shuffle_indexi_1]

# data_0_te = data_0
# data_1_te = data_1
data_0_tr = data_0[:5000]
data_1_tr = data_1[:5000]
data_0_vali = data_0[5000:]
data_1_vali = data_1[5000:]

print(np.shape(data_0_tr))
print(np.shape(data_1_tr))
print(np.shape(data_0_vali))
print(np.shape(data_1_vali))

# data_0_te_x = np.reshape(data_0_te[:, 1:694], [-1, 33, 21])
# data_0_te_y = data_0_te[:, 0]
# data_1_te_x = np.reshape(data_1_te[:, 1:694], [-1, 33, 21])
# data_1_te_y = data_1_te[:, 0]
data_0_tr_x = np.reshape(data_0_tr[:, 1:694], [-1, 33, 21])
data_0_tr_y = data_0_tr[:, 0]
data_1_tr_x = np.reshape(data_1_tr[:, 1:694], [-1, 33, 21])
data_1_tr_y = data_1_tr[:, 0]
data_0_vali_x = np.reshape(data_0_vali[:, 1:694], [-1, 33, 21])
data_0_vali_y = data_0_vali[:, 0]
data_1_vali_x = np.reshape(data_1_vali[:, 1:694], [-1, 33, 21])
data_1_vali_y = data_1_vali[:, 0]

# data_te_x = np.concatenate((data_0_te_x, data_1_te_x), axis=0)
# data_te_y = np.concatenate((data_0_te_y, data_1_te_y), axis=0)
data_tr_x = np.concatenate((data_0_tr_x, data_1_tr_x), axis=0)
data_tr_y = np.concatenate((data_0_tr_y, data_1_tr_y), axis=0)
data_vali_x = np.concatenate((data_0_vali_x, data_1_vali_x), axis=0)
data_vali_y = np.concatenate((data_0_vali_y, data_1_vali_y), axis=0)

# np.save('data/t_te_x', data_te_x)
# np.save('data/t_te_y', data_te_y)
np.save('data/t_tr_x', data_tr_x)
np.save('data/t_tr_y', data_tr_y)
np.save('data/t_vali_x', data_vali_x)
np.save('data/t_vali_y', data_vali_y)

print(np.shape(data_tr_x))
print(np.shape(data_tr_y))
print(np.shape(data_vali_x))
print(np.shape(data_vali_y))

