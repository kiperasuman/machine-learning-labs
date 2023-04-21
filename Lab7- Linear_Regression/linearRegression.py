import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

# Klasörün içine attığımız csv uzantılı dosyayı pandas kütüphanesi kullanrak import ediyoruz

dosya = pd.read_csv('diabetes.csv')
dosya.head()
# Gelen verileri işleme koyup içeriğine bakıyoruz yani sütun adlarına ..
dosya.info()
# Silinecek boş verimiz olmadığı için boş bir veri ekleyip bunu drop ediyoruz
#axis de hangi değeri alırsa ona göre satır ya da sütun işlemi yapar
# axis  = 0 ise satır axis = 1 ise sütun işlemi 
# inplace ile de True  olduğunda tamamen kaldırır

#dosya.drop(["Unnamed Data:","id"],axis=1, inplace=True)

#fonksiyon oluşturup Outcome verisinden gelen değere bakıyoruz
dosya.Outcome = [1 if each == "0" else 0 for each in dosya.Outcome]
#gelen verileri arraye çevirir
y = dosya.Outcome.values
x_data  = dosya.drop(["Outcome"],axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
print(x)

# ilk önce veri seti için import işlemini gerçekleştiriyoruz daha sonra 
# train  ve test olarak ayrılan veri setlerine oran veriyoruz mesela 
# tarin_set = 0.80 ise %80 'i train %20 si ise test setini oluşturur
# random_state atamasını yapmalıyız çünkü modelideğerlendirirken sürekli farklı değerlerden alacağı için sonuç yanlış çıkabilir

# T =  tranpose
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.30,random_state=100)
x_train = x_train.T
print("\n X TRAIN SHAPE")
print(x_train.shape)
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
print("\n X TEST SHAPE")
print(x_test.shape)
print("\n Y TRAIN SHAPE")
print(y_train.shape)
print("\n Y TEST SHAPE")
print(y_test.shape)



#weight değeri 0 olurse feature ile çarpılınca tüm değerler 0 olur ve model için öğrenme gerçekleşmez
#O yüzden çok küçük bir değer ile çarparız (0.01)
#bias değeri de 0 olarak verilebilir


# 0 ile 1 arasında olasılıksal bir değer verir
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b=0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

print("Sigmoid Value : " ,sigmoid(0))

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation
    #features ile transpose çarpıp bias ile toplarız daha sonra z değeri elde edilir ve bu değer de sigmoid fonk verilir
    z = np.dot(w.T,x_train) + b 
    y_head = sigmoid(z) 
   # tek veri için uygulanırsa 
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) 
    #tüm veriler için uygulanırsa
    cost = (np.sum(loss))/x_train.shape[1]   

    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]  
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]              
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients

#güncellenen weight değerlerini  train verisine uygularız.
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion): 
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train) 
        cost_list.append(cost) 
        w = w - learning_rate * gradients["derivative_weight"]  
        b = b - learning_rate * gradients["derivative_bias"]   
        if i % 10 == 0:
            cost_list2.append(cost) 
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))

    parameters = {"weight": w,"bias": b} 
    plt.plot(index,cost_list2) 
    plt.xticks(index,rotation='vertical') 
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list 

def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b) 
    Y_prediction = np.zeros((1,x_test.shape[1])) #(1,114) 
    
    for i in range(z.shape[1]): 
        if z[0,i]<= 0.5: 
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
                        
    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    
    dimension =  x_train.shape[0]  
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
   
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 4, num_iterations = 150)

from sklearn import linear_model
lr = linear_model.LogisticRegression(random_state=100,max_iter=15)
print(lr.fit(x_train.T,y_train.T))
print(y_pred = lr.predict(x_test.T))
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))