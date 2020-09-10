import tkinter as tk
from tkinter import ttk 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
import keras
import itertools
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from sklearn.model_selection import cross_val_score    
from tkinter import filedialog
from tkinter import *
from sklearn.model_selection import GridSearchCV

print("----------------------")

window = tk.Tk()
window.geometry("1500x800") 
# window['background']='#e6c1c1'
window.title("Samed Hıra")
def plotConfusionMatrix(cm,
               normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def importData():
    global dataframe
    csv_file_path = filedialog.askopenfilename(filetypes=(("CSV Files", "*.csv"), ("All", "*.*")))
    dataframe = pd.read_csv(csv_file_path)

def readCsvFile():
    list_all.delete(0, END)
    for col in dataframe.columns:
        list_all.insert(END, col)
              
def addButton(self):
    self.insert(END, list_all.get(ACTIVE))

def deleteButton(self):
    self.delete(ACTIVE)

def getTarget():  
    targets = list_target.get(0, END)
    targets = np.asarray(targets)
    print("targets = ",targets)
    y = dataframe[targets]
    print("chosen targets = ",y)
    return y

def getPredictor():   
    predictors = list_predictor.get(0, END)
    predictors = np.asarray(predictors)
    print("predictors = ",predictors)
    Xx = dataframe[predictors]
    print("chosen predictors = ",Xx)
    global input_num
    input_num=len(Xx.columns)
    return Xx
              
def save_button():
      predictor = getPredictor()
      predictor = np.asarray(predictor)
      target = getTarget()
      target = np.asarray(target)

def runButtonClick():
    Xx = getPredictor()  
    y = getTarget()
    y = y.values.ravel()
    combo_value = problem.get()
    cross_random_value = int(cross_random.get()) 
    
    if combo_value == "SVM":
        if cross_random_value == 1:
              Cv=int(cross.get())
              kernel = svm_kernel.get()
              if kernel == "linear":                  
                    c_min=float(min_c.get())
                    c_max=float(max_c.get())                                                    
                    C = np.linspace(c_min, c_max, 3)  
                    param_grid = {'C': C,'kernel': ['linear']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

              elif kernel == "sigmoid":
                    c_min=float(min_c.get())
                    c_max=float(max_c.get()) 
                    coef_min=float(min_coef0.get())
                    coef_max=float(max_coef0.get())
                    g_min=float(min_g.get())
                    g_max=float(max_g.get())
                    C = np.linspace(c_min, c_max, 3)
                    Coef0=np.linspace(coef_min, coef_max, 3)
                    Gamma = np.linspace(g_min, g_max, 3)
                    param_grid = {'C': C,'coef0':Coef0,'gamma':Gamma,'kernel': ['sigmoid']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
                  
              elif kernel == "rbf":
                    c_min=float(min_c.get())
                    c_max=float(max_c.get())                    
                    g_min=float(min_g.get())
                    g_max=float(max_g.get())
                    C = np.linspace(c_min, c_max, 3)                  
                    Gamma = np.linspace(g_min, g_max, 3)
                    param_grid = {'C': C,'gamma':Gamma,'kernel': ['rbf']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
                    
              elif kernel == "poly":
                    c_min=float(min_c.get())
                    c_max=float(max_c.get()) 
                    coef_min=float(min_coef0.get())
                    coef_max=float(max_coef0.get())
                    g_min=float(min_g.get())
                    g_max=float(max_g.get())
                    d_min = float(min_d.get())
                    d_max = float(max_d.get())
                    C = np.linspace(c_min, c_max, 3)
                    Coef0=np.linspace(coef_min, coef_max, 3)
                    Gamma = np.linspace(g_min, g_max, 3)
                    Degree = np.linspace(d_min, d_max, 3)
                    param_grid = {'C': C,'coef0':Coef0,'gamma':Gamma,'degree':Degree,'kernel': ['poly']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
                    
                    
              cvs = cross_val_score(svm_model_linear, Xx , y ,cv = Cv)
              print(cvs)
              svm_model_linear.fit(Xx,y)
              svm_predictions = svm_model_linear.predict(Xx) 
              accuracy = svm_model_linear.score(Xx, y) 
              print("accuracy= ",accuracy)
              print ('c.r = ',classification_report(y, svm_predictions))
              
              np.set_printoptions(precision=2)


              titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
              for title, normalize in titles_options:
                    
                          disp = plot_confusion_matrix(svm_model_linear, Xx, y,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
                          disp.ax_.set_title(title)

                          print(title)
   
              print ('Accuracy Score :',accuracy_score(y, svm_predictions))
              plt.show()
              report=classification_report(y, svm_predictions)        
              tk.Label(window, text = report).place(x = 1060 ,y = 50)
              
        elif cross_random_value == 2:
              random_percent=float(random.get())
              X_train, X_test, y_train, y_test = train_test_split(Xx, y, test_size=random_percent, random_state=1)

              kernel = svm_kernel.get()
              if kernel == "linear":                   
                    c_min=float(min_c.get())
                    c_max=float(max_c.get())                    
                    C = np.linspace(c_min, c_max, 3)                   
                    param_grid = {'C': C,'kernel': ['linear']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
              elif kernel == "sigmoid":                  
                    c_min=float(min_c.get())
                    c_max=float(max_c.get()) 
                    coef_min=float(min_coef0.get())
                    coef_max=float(max_coef0.get())
                    g_min=float(min_g.get())
                    g_max=float(max_g.get())
                    C = np.linspace(c_min, c_max, 3)
                    Coef0=np.linspace(coef_min, coef_max, 3)
                    Gamma = np.linspace(g_min, g_max, 3)
                    param_grid = {'C': C,'coef0':Coef0,'gamma':Gamma,'kernel': ['sigmoid']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
              elif kernel == "rbf":                   
                    c_min=float(min_c.get())
                    c_max=float(max_c.get())                    
                    g_min=float(min_g.get())
                    g_max=float(max_g.get())
                    C = np.linspace(c_min, c_max, 3)                  
                    Gamma = np.linspace(g_min, g_max, 3)
                    param_grid = {'C': C,'gamma':Gamma,'kernel': ['rbf']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
              elif kernel == "poly":
                    c_min=float(min_c.get())
                    c_max=float(max_c.get()) 
                    coef_min=float(min_coef0.get())
                    coef_max=float(max_coef0.get())
                    g_min=float(min_g.get())
                    g_max=float(max_g.get())
                    d_min = float(min_d.get())
                    d_max = float(max_d.get())
                    C = np.linspace(c_min, c_max, 3)
                    Coef0=np.linspace(coef_min, coef_max, 3)
                    Gamma = np.linspace(g_min, g_max, 3)
                    Degree = np.linspace(d_min, d_max, 3)
                    param_grid = {'C': C,'coef0':Coef0,'gamma':Gamma,'degree':Degree,'kernel': ['poly']}
                    svm_model_linear = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
                    
        
       
              svm_model_linear.fit(X_train, y_train)
              svm_predictions = svm_model_linear.predict(X_test) 
              
#accuracy = (tp+tn)/tp+tn+fp+fn

              np.set_printoptions(precision=2)

              titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
              for title, normalize in titles_options:
                       disp = plot_confusion_matrix(svm_model_linear, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
                       disp.ax_.set_title(title)

                       print(title)
    # print(disp.confusion_matrix)

              plt.show()
              print ('Accuracy Score :',accuracy_score(y_test, svm_predictions)) 
# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
#f1 ikisinin harmonik ortalaması
        
              report=classification_report(y_test, svm_predictions)
        
              label_report = tk.Label(window, text = report)
              label_report.place(x = 1060 ,y = 50)
    elif combo_value == "MLP":
          
        layers=int(mlp_layers.get())    
        epoch_num=int(itera.get())
        optm=float(opt.get())
        
        
        if layers == 1:
              first_number = int(first_layer.get())
              act_func1=activation_func1.get()
              model = keras.Sequential([
                    keras.layers.Dense(first_number,input_dim=input_num, activation=act_func1),
                    keras.layers.Dense(24,activation='softmax')
                    ]) 
      
        elif layers == 2:
              first_number = int(first_layer.get())
              second_number = int(second_layer.get())
              act_func1=activation_func1.get()
              act_func2=activation_func2.get()
              model = keras.Sequential([
                    keras.layers.Dense(first_number,input_dim=input_num, activation=act_func1),
                    keras.layers.Dense(second_number, activation=act_func2),                    
                    keras.layers.Dense(24,activation='softmax')
                    ]) 
        elif layers == 3:
              first_number = int(first_layer.get())
              second_number = int(second_layer.get())  
              third_number = int(third_layer.get())
              act_func1=activation_func1.get()
              act_func2=activation_func2.get()
              act_func3=activation_func3.get()
              model = keras.Sequential([
                    keras.layers.Dense(first_number,input_dim=input_num, activation=act_func1),
                    keras.layers.Dense(second_number, activation=act_func2),
                    keras.layers.Dense(third_number, activation=act_func3),
                    keras.layers.Dense(24,activation='softmax')
                    ])  
          
        opt2 = keras.optimizers.Adam(lr=optm)  
        print("MLP seçildi")
        model.compile(optimizer=opt2,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
        if cross_random_value == 1:
              Cv=int(cross.get())
        
              MLP = model.fit(Xx, y,epochs=epoch_num,validation_split=Cv)
              y_pred = model.predict(Xx)

              y_pred=np.argmax(y_pred,axis=1)

              from sklearn.metrics import confusion_matrix
              cm = confusion_matrix(y, y_pred)
              report1=classification_report(y, y_pred)
              print ('Accuracy Score :',accuracy_score(y, y_pred))
        elif cross_random_value == 2:
              random_percent=float(random.get())
              X_train, X_test, y_train, y_test = train_test_split(Xx, y, test_size=random_percent, random_state=1)
              MLP = model.fit(X_train, y_train,epochs=epoch_num)
        
              y_pred = model.predict(X_test)
        
              y_pred=np.argmax(y_pred,axis=1)

              from sklearn.metrics import confusion_matrix
              cm = confusion_matrix(y_pred, y_test)    
              report1=classification_report(y_test, y_pred)
              print ('Accuracy Score :',accuracy_score(y_test, y_pred))
        plotConfusionMatrix(cm=cm, title='Confusion Matrix')
               
        np.set_printoptions(precision=2)
               
        label_report = tk.Label(window, text = report1)
        label_report.place(x = 1060 ,y = 50)

    print(problem.get())
   
def disableKernelFunc():    
    kernel = svm_kernel.get()
    if kernel == "linear":
            svm_coef0.configure(state='disabled')  
            svm_d.configure(state='disabled')
            svm_g.configure(state='disabled')
            svm_c.configure(state='normal')
                            
    elif kernel == "sigmoid":
            svm_d.configure(state='disabled')
            svm_c.configure(state='normal')
            svm_coef0.configure(state="normal")
            svm_g.configure(state="normal")
                   
    elif kernel == "rbf":
            svm_coef0.configure(state="disabled")
            svm_d.configure(state="disabled")
            svm_c.configure(state="normal")
            svm_g.configure(state="normal")
              
    elif kernel == "poly":
            svm_c.configure(state="normal")
            svm_coef0.configure(state="normal")
            svm_g.configure(state="normal")
            svm_d.configure(state="normal")  
            
def disableCrossRandom():
    cross_random_value = int(cross_random.get())  
    if cross_random_value == 1:
            cross.configure(state="normal")
            random.configure(state="disabled")
    elif cross_random_value == 2:
            cross.configure(state="disabled")
            random.configure(state="normal")
    
def disableMlp():
    layers=int(mlp_layers.get())
    if layers == 1:
          first_layer.configure(state="normal")
          second_layer.configure(state="disabled")
          third_layer.configure(state="disabled")
          comboBox2.configure(state="disabled")
          comboBox3.configure(state="disabled")
    elif layers == 2:
          first_layer.configure(state="normal")
          second_layer.configure(state="normal")
          third_layer.configure(state="disabled")
          comboBox2.configure(state="normal")
          comboBox3.configure(state="disabled")
    elif layers == 3:
          first_layer.configure(state="normal")
          second_layer.configure(state="normal")
          third_layer.configure(state="normal")
          comboBox2.configure(state="normal")
          comboBox3.configure(state="normal")
 

pw = ttk.Panedwindow(window, orient = tk.HORIZONTAL)
pw.pack(fill = tk.BOTH, expand = True)

m2 = ttk.Panedwindow(pw, orient = tk. VERTICAL)

frame2 = ttk.Frame(pw, width = 500, height = 430, relief = tk.RAISED)
frame3 = ttk.Frame(pw, width = 500, height = 240, relief = tk.RAISED)

m2.add(frame2)
m2.add(frame3)

frame1 = ttk.Frame(pw, width = 450, height = 640, relief = tk.RAISED)
frame4 = ttk.Frame(pw, width = 360, height = 640, relief = tk.RAISED)

pw.add(m2)
pw.add(frame1)      
pw.add(frame4)
problem = tk.StringVar()
comboBox = ttk.Combobox(window, textvariable = problem, values = ("SVM","MLP"), state= "readonly")
comboBox.place(x=15,y=15)

##########▓

tk.Button(window, text='Browse DataSet', command=importData).place(x=215,y=50)
tk.Button(window, text='Read CSV', command=lambda: readCsvFile()).place(x=365,y=50)

list_all = Listbox(window)
list_all.place(x=15,y=100)

list_predictor = Listbox(window)
list_predictor.place(x=165,y=100)
tk.Button(window, text="Add Predictor", command=lambda: addButton(list_predictor)).place(x=175,y=320)
tk.Button(window, text="Delete Predictor", command=lambda: deleteButton(list_predictor)).place(x=175,y=355)

list_target = Listbox(window)
list_target.place(x=305,y=100)
tk.Button(window, text="Add Target", command=lambda: addButton(list_target)).place(x=315,y=320)
tk.Button(window, text="Delete Target", command=lambda: deleteButton(list_target)).place(x=315,y=355)

button = tk.Button(window, text = "Save", activebackground = "blue",
                        activeforeground = "black",
                      command = save_button).place(x= 700,y=700)

#############        
button = tk.Button(window, text = "Run", activebackground = "blue",
                        activeforeground = "black",
                      command = runButtonClick)

button.place(x = 900, y = 750)

tk.Label(window, text = "SVM").place(x = 535, y = 15)

label_func = tk.Label(window, text = "Kernel Func:")
label_func.place(x = 515, y = 45)
svm_kernel = tk.StringVar()
tk.Radiobutton(window,text = "Linear", value = "linear",command=disableKernelFunc,variable = svm_kernel).place(x=515, y= 70)
tk.Radiobutton(window,text = "RBF", value = "rbf", command=disableKernelFunc,variable = svm_kernel).place(x= 515, y= 95)
tk.Radiobutton(window,text = "Polynomial", value = "poly",command=disableKernelFunc,variable = svm_kernel).place(x= 515, y= 120)
tk.Radiobutton(window,text = "Sigmoid", value = "sigmoid", command=disableKernelFunc,variable = svm_kernel).place(x= 515, y= 145)

tk.Label(window, text = "Model Parameters").place(x = 515, y = 195)
tk.Label(window, text = "Search Range").place(x = 690, y = 195)

cross_random = tk.StringVar()
tk.Radiobutton(window, text = "Cross Validation:", value = 1,command=disableCrossRandom,variable = cross_random).place(x=15, y= 320)
tk.Radiobutton(window,text = "Random Percent:", value = 2, command=disableCrossRandom,variable = cross_random).place(x= 15, y= 370)

cross = tk.Entry(window, width = 12)
cross.insert(string = "",index = 0)
cross.place(x = 15,y = 345)

random = tk.Entry(window, width = 12)
random.insert(string = "",index = 0)
random.place(x = 15,y = 395)
#############################################  C  ##########################################
tk.Label(window, text = "C:").place(x = 515, y = 220)
svm_c = tk.Entry(window, width = 10)
svm_c.insert(string = "",index = 0)
svm_c.place(x = 515,y = 245)
min_c = tk.Entry(window, width = 10)
min_c.insert(string = "",index = 0)
min_c.place(x = 650,y = 245)
max_c = tk.Entry(window, width = 10)
max_c.insert(string = "",index = 0)
max_c.place(x = 740,y = 245)
############################################# Coef0   ##########################################
tk.Label(window, text = "Coef0:").place(x = 515, y = 270)
svm_coef0 = tk.Entry(window, width = 10)
svm_coef0.insert(string = "",index = 0)
svm_coef0.place(x = 515,y = 295)
min_coef0 = tk.Entry(window, width = 10)
min_coef0.insert(string = "",index = 0)
min_coef0.place(x = 650,y = 295)
max_coef0 = tk.Entry(window, width = 10)
max_coef0.insert(string = "",index = 0)
max_coef0.place(x = 740,y = 295)
############################################# Gamma   ##########################################
tk.Label(window, text = "Gamma:").place(x = 515, y = 320)
svm_g = tk.Entry(window, width = 10)
svm_g.insert(string = "",index = 0)
svm_g.place(x = 515,y = 345)
min_g = tk.Entry(window, width = 10)
min_g.insert(string = "",index = 0)
min_g.place(x = 650,y = 345)
max_g = tk.Entry(window, width = 10)
max_g.insert(string = "",index = 0)
max_g.place(x = 740,y = 345)
############################################# Degree   ##########################################
tk.Label(window, text = "Degree:").place(x = 515, y = 370)
svm_d = tk.Entry(window, width = 10)
svm_d.insert(string = "",index = 0)
svm_d.place(x = 515,y = 395)
min_d = tk.Entry(window, width = 10)
min_d.insert(string = "",index = 0)
min_d.place(x = 650,y = 395)
max_d = tk.Entry(window, width = 10)
max_d.insert(string = "",index = 0)
max_d.place(x = 740,y = 395)

tk.Label(window, text = "MLP").place(x = 65, y = 445)
tk.Label(window, text = "# of hidden layer:").place(x = 15, y = 470)

mlp_layers = tk.StringVar()
tk.Radiobutton(window, text = "1", value = 1,command=disableMlp,variable = mlp_layers,state="normal").place(x=15, y= 495)
tk.Radiobutton(window,text = "2", value = 2, command=disableMlp,variable = mlp_layers,state="normal").place(x= 65, y= 495)
tk.Radiobutton(window,text = "3", value = 3,command=disableMlp,variable = mlp_layers,state="normal").place(x= 115, y= 495)

tk.Label(window, text = "Neurons in 1st layer:").place(x = 15, y = 545)
first_layer = tk.Entry(window, width = 10)
first_layer.insert(string = "",index = 0)
first_layer.place(x = 160,y = 545)
tk.Label(window,text = "Neurons in 2nd layer:").place(x = 15, y = 570)
second_layer = tk.Entry(window, width = 10)
second_layer.insert(string = "",index = 0)
second_layer.place(x = 160,y = 570)
label_third_layer = tk.Label(window, text = "Neurons in 3rd layer:")
label_third_layer.place(x = 15, y = 595)
third_layer = tk.Entry(window, width = 10)
third_layer.insert(string = "",index = 0)
third_layer.place(x = 160,y = 595)

tk.Label(window, text = "Activation Functions:").place(x = 270, y = 515)
activation_func1 = tk.StringVar()
comboBox1 = ttk.Combobox(window, textvariable = activation_func1, values = ("relu","linear","sigmoid","softmax","tanh"), state= "readonly")
comboBox1.place(x=250,y=545)
activation_func2 = tk.StringVar()
comboBox2 = ttk.Combobox(window, textvariable = activation_func2, values = ("relu","linear","sigmoid","softmax","tanh"), state= "readonly")
comboBox2.place(x=250,y=570)
activation_func3 = tk.StringVar()
comboBox3 = ttk.Combobox(window, textvariable = activation_func3, values = ("relu","linear","sigmoid","softmax","tanh"), state= "readonly")
comboBox3.place(x=250,y=595)

tk.Label(window, text = "Epoch:").place(x = 15, y = 645)
itera = tk.Entry(window, width = 10)
itera.insert(string = "",index = 0)
itera.place(x = 160,y = 645)

tk.Label(window, text = "Learning rate:").place(x = 15, y = 670)
opt = tk.Entry(window, width = 10)
opt.insert(string = "",index = 0)
opt.place(x = 160,y = 670)

tk.Label(window, text = "Outputs:").place(x = 1180, y = 15)
        
tk.Frame(window, width = 400, height = 625, bg="white").place(x=1020,y=50)

window.mainloop()
