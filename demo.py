import numpy as np
import os
import matplotlib.pyplot as plt

lr = 0.1
n=100
class my_neural_network:
    def __init__(self):
        self.w1 = np.random.rand(28 * 28, 128)*0.01
        self.w2 = np.random.rand(128, 10)*0.01
        self.b1 = np.random.rand(1, 128)
        self.b2 = np.random.rand(1, 10)

    def process(self, x, y):

        # x为n*784矩阵，y为n*1的列向量
        h = x.dot(self.w1) + self.b1

        # sigmoid激活函数
        sigmoid = 1 / (1 + np.exp(-h))
        self.y_pre = sigmoid.dot(self.w2) + self.b2

        # 对y_pre进行Softmax
        for i in range(self.y_pre.shape[0]):
            max=np.max(self.y_pre[i,:])
            for j in range(self.y_pre.shape[1]):
                self.y_pre[i, j] = np.exp(self.y_pre[i, j]-max)
            self.y_pre[i, :] /= self.y_pre[i, :].sum()
        # 使用交叉熵损失函数
        loss=0
        for i in range(self.y_pre.shape[0]):
            loss += -np.log(self.y_pre[i, y[i, 0]]+1e-15)/10
        print("loss:",loss)

        # 反向传播
        y_label = np.zeros_like(self.y_pre)
        for i in range(len(y)):
            y_label[i, y[i, 0]] = 1
        dy_pre = self.y_pre - y_label
        dw2 = sigmoid.T.dot(dy_pre)/10
        db2 = np.sum(dy_pre,axis=0,keepdims=True)/10
        dsigmoid = dy_pre.dot(self.w2.T)
        dh = dsigmoid * (sigmoid * (1 - sigmoid))
        dw1 = x.T.dot(dh)/10
        db1 = np.sum(dh, axis=0, keepdims=True) / 10

        # 梯度下降
        self.w1 += -dw1 * lr
        self.w2 += -dw2 * lr
        self.b1 += -db1 * lr
        self.b2 += -db2 * lr

    def predict(self,x,y):
        # x为n*784矩阵，y为n*1的列向量
        h = x.dot(self.w1) + self.b1

        # sigmoid激活函数
        sigmoid = 1 / (1 + np.exp(-h))
        self.y_pre = sigmoid.dot(self.w2) + self.b2

        # 归一化,采用Softmax
        # 对y_pre进行归一化
        for i in range(self.y_pre.shape[0]):
            for j in range(self.y_pre.shape[1]):
                self.y_pre[i, j] = np.exp(self.y_pre[i, j])
            self.y_pre[i, :] /= self.y_pre[i, :].sum()
        predict = self.y_pre.argmax(axis=1)
        return predict.reshape(-1,1)

#数据加载函数
def load_image(file_name):
    file=open(file_name,'rb')
    #跳过前 16 字节是因为 MNIST 的二进制文件遵循 IDX3 格式标准，其文件头部的元数据占用了前 16 个字节
    data = np.frombuffer(file.read(), dtype=np.uint8, offset=16).reshape(-1,28*28)
    #除以255归一化
    return data.astype(np.float32)/ 255.0

def load_labels(filename):
    file= open(filename, "rb")
    data = np.frombuffer(file.read(), dtype=np.uint8, offset=8).reshape(-1,1)
    return data

class data_loader:
    def __init__(self,images_file,labels_file):
        self.images=load_image(images_file)
        self.labels=load_labels(labels_file)
        #随机打乱顺序
        temp=np.hstack((self.images,self.labels))
        np.random.shuffle(temp)
        self.images=temp[:,:-1]
        self.labels=temp[:,-1:]
        #组合数组后label会变成float类型，需转化为int
        self.labels=self.labels.astype(np.int8)

#数据集路径
base="D:\\代码\\PythonProject\\Machine Learning\\Handwiriting_Number_Recognition\\data\\MNIST\\raw"
files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
]

#加载数据
train_loader=data_loader(os.path.join(base,files[0]),os.path.join(base,files[1]))
test_loader=data_loader(os.path.join(base,files[2]),os.path.join(base,files[3]))

#展示图片
# print(train_loader.images[0])
# plt.imshow(train_loader.images[0]reshape(28,28),'gray')
# plt.show()

#训练模型
model=my_neural_network()
images=train_loader.images
labels=train_loader.labels
for epoch in range(10):
    for i in range(images.shape[0]//100):
        model.process(images[i*100:100*i+100,:],labels[i*100:100*i+100,:])
        print(f"epoch:{epoch+1}/10,num:{i+1}")

#测试模型
total=0
correct=0
images=test_loader.images
labels=test_loader.labels
for i in range(images.shape[0]//100):
    images1=images[i*100:100*i+100,:]
    labels1=labels[i*100:100*i+100,:]
    y_pre=model.predict(images1,labels1)
    total+=n
    for j in range(n):
        correct+=1 if y_pre[j,0]==labels1[j,0] else 0
print(correct/total)


