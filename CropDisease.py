from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from sklearn.metrics import accuracy_score
import webbrowser
import os
import cv2
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json

main = Tk()
main.title("Agriculture Crop Image Classification")
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy

plants = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
          'Potato___Late_blight', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
          'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy',
          'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
          'Tomato_Spider_mites_Two_spotted_spider_mite',
          'Brinjal_Epilachna_Beetle', 'Brinjal_Flea_Beetle', 'Brinjal_Healthy', 'Brinjal_Jassid', 'Brinjal_Mite',
          'Brinjal_Mite_and_Epilachna_Beetle', 'Brinjal_Nitrogen_and_Potassium_Deficiency', 'Brinjal_Nitrogen_Deficiency',
          'Brinjal_Potassium_Deficiency']

disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'Precautions': ['Plant disease-resistant pepper varieties', 'Avoid overhead watering'],
        'Pesticides': 'Double Nickel 55 LC ,WDG,EcoSwing Botanical Fungicide,GreenFurrow BacStop.'
    },
    'Pepper__bell___healthy': {
        'Precautions': ['Practice good pepper plant hygiene', 'Control pests'],
        'Pesticides': 'No Pesticides needed, as the plant is healthy.'
    },
    'Potato___Early_blight': {
        'Precautions': ['Plant disease-resistant potato varieties', 'Apply fungicides'],
        'Pesticides': 'Chlorothalonil, Famoxadone/Cymoxanil (Tanos),Mancozeb.'
    },
    'Potato___healthy': {
        'Precautions': ['Practice good potato plant hygiene', 'Control pests'],
        'Pesticides': 'No Pesticides needed, as the plant is healthy.'
    },
    'Potato___Late_blight': {
        'Precautions': ['Plant disease-resistant potato varieties', 'Apply fungicides'],
        'Pesticides': 'Fenamidone (Reason 500SC),Azoxystrobin (Quadris), Boscalid (Endura).'
    },
    'Tomato__Target_Spot': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Prune affected leaves'],
        'Pesticides': 'chlorothalonil, mancozeb, and copper oxychloride.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'Precautions': ['Control aphid vectors', 'Remove infected plants'],
        'Pesticides': 'There is no Pesticides for this viral disease; remove and destroy infected plants.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Prune affected leaves'],
        'Pesticides': 'cypermethrin, deltamethrin, bifenthrin.'
    },
    'Tomato_Bacterial_spot': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Prune infected branches'],
        'Pesticides': 'acibenzolar-S-methyl (Actigard®), Serenade Opti®, and Sporan EC2.'
    },
    'Tomato_Early_blight': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Apply fungicides'],
        'Pesticides': 'copper products mixed with mancozeb, as well as biopesticides like Serenade Opti®.'
    },
    'Tomato_healthy': {
        'Precautions': ['Practice good tomato plant hygiene', 'Control pests'],
        'Pesticides': 'No Pesticides needed, as the plant is healthy.'
    },
    'Tomato_Late_blight': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Apply fungicides'],
        'Pesticides': 'fungicides like copper (Champ), chlorothalonil (Bravo).'
    },
    'Tomato_Leaf_Mold': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Prune affected leaves'],
        'Pesticides': 'chemical fungicides based on sulfur, as well as natural fungicides like neem oil, rosemary oil.'
    },
    'Tomato_Septoria_leaf_spot': {
        'Precautions': ['Plant disease-resistant tomato varieties', 'Prune infected leaves'],
        'Pesticides': 'maneb, mancozeb, and benomyl.'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'Precautions': ['Monitor and control spider mite populations', 'Maintain good plant health'],
        'Pesticides': 'Apply insecticidal soap or neem oil,insecticidal soaps, horticultural oils.'
    }
}

def uploadDataset():
    global X, Y
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, 'dataset loaded\n')

def imageProcessing():
    text.delete('1.0', END)
    global X, Y
    X = np.load("model/myimg_data.txt.npy")
    Y = np.load("model/myimg_label.txt.npy")
    Y = to_categorical(Y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    X = X.astype('float32')
    X = X / 255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END, 'image processing completed\n')
    img = X[20].reshape(64, 64, 3)
    cv2.imshow('ff', cv2.resize(img, (250, 250)))##########################################
    cv2.waitKey(0)

def cnnModel():
    global model
    global accuracy
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END, "CNN Crop Disease Recognition Model Prediction Accuracy = " + str(acc))
    else:
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim=256, activation='relu'))
        model.add(Dense(output_dim=25, activation='softmax'))  # Updated output_dim to 25
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        hist = model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        model.save_weights('model/model_weights.h5')
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END, "CNN Crop Disease Recognition Model Prediction Accuracy = " + str(acc))

selected_disease = ""  # Declare selected_disease as a global variable

def predict():
    global model
    global selected_disease  # Access the global variable
    
    filename = filedialog.askopenfilename(initialdir="testImages") ############################tkinter
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test / 255
    preds = model.predict(test)
    predict = np.argmax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 400))
    cv2.putText(img, 'Crop Disease Recognize as : ' + plants[predict], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)
    cv2.imshow('Crop Disease Recognize as : ' + plants[predict], img)
    cv2.waitKey(0)
    
    # Set the selected disease
    selected_disease = plants[predict]
    
    # Display precautions and Pesticides for the detected disease
    if selected_disease in disease_info:
        precautions = "\n".join(disease_info[selected_disease]['Precautions'])
        Pesticides = disease_info[selected_disease]['Pesticides']
        text.insert(END, f"\nDisease: {selected_disease}\n", 'disease')
        text.insert(END, f"Precautions:\n{precautions}\n", 'precaution')
        text.insert(END, f"Pesticides:\n{Pesticides}\n", 'Pesticides')
        text.tag_configure("disease", foreground="red")
        text.tag_configure("precaution", foreground="blue")
        text.tag_configure("Pesticides", foreground="green")

def open_video():
    global selected_disease  # Access the global variable
    
    # Dictionary mapping diseases to YouTube video links
    disease_videos = {
        'Pepper__bell___Bacterial_spot': 'https://youtu.be/1HgsMF4gd7U?si=xJPxwxKyXfHyEh5W',
        'Pepper__bell___healthy': 'https://www.youtube.com/watch?v=Mr6i4s5bSA',
        'Potato___Early_blight': 'https://youtu.be/6i5_sLY_pWc?si=tq_rzMe1YPkeCf5w',
        'Potato___healthy': 'https://www.youtube.com/watch?v=bGq3bNWwGAs',
        'Potato___Late_blight': 'https://youtu.be/PSXXoGrOyDg?si=-vpOEjSSBnuMG1oB',
        'Tomato__Target_Spot': 'https://youtu.be/jONPVKSvFW0?si=L5lULWoGYDZf_KvL',
        'Tomato__Tomato_mosaic_virus': 'https://youtu.be/n8PyKuGcURY?si=TbNT8ZwIUsiE8Ves',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'https://youtu.be/cM5jEqd5n5A?si=NHsNhqiVRysx_uAb',
        'Tomato_Bacterial_spot': 'https://youtu.be/JT1gGgJmOBM?si=FdqcM-ngtoUDixiY',
        'Tomato_Early_blight': 'https://youtu.be/2GYD7aVBFtg?si=-lACqKF_KVUuAF2G',
        'Tomato_healthy': 'No video link available',
        'Tomato_Late_blight': 'https://youtu.be/Djd7z03iSYE?si=7NVGT-ArQnS9K-z_',
        'Tomato_Leaf_Mold': 'https://youtu.be/0lZOboTH8m4?si=2cwr43FsMfgXUxrq',
        'Tomato_Septoria_leaf_spot': 'No video link available',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'https://youtu.be/iRYvw9vRguk?si=cRLo6w3-hbQXyVtX'
    }
    
    # Check if the selected disease has a corresponding video link
    if selected_disease in disease_videos:
        video_url = disease_videos[selected_disease]
        webbrowser.open_new(video_url)
    else:
        messagebox.showinfo("Video Not Available", "Sorry, video tutorial for this disease is not available.")
        

def graph():
    acc = accuracy['accuracy']
    loss = accuracy['loss']
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color='green')
    plt.plot(loss, 'ro-', color='blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.title('Iteration Wise Accuracy & Loss Graph')
    plt.show()

def close():
    main.destroy()
    text.delete('1.0', END)

font = ('times', 15, 'bold')
title = Label(main, text='Agriculture Crop Image Classification')
title.config(font=font)
title.config(height=3, width=120)
title.config(bg="#87CEEB")
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')


uploadButton = Button(main, text="1.Upload Crop Disease Dataset", command=uploadDataset)
uploadButton.place(x=300, y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="2.Image Processing & Normalization", command=imageProcessing)
processButton.place(x=600, y=100)
processButton.config(font=ff)

modelButton = Button(main, text="3.Build Crop Disease Recognition Model", command=cnnModel)
modelButton.place(x=900, y=100)
modelButton.config(font=ff)

predictButton = Button(main, text="4.Upload Test Image & Predict Disease", command=predict)
predictButton.place(x=300, y=150)
predictButton.config(font=ff)

graphButton = Button(main, text="5.Accuracy & Loss Graph", command=graph)
graphButton.place(x=600, y=150)
graphButton.config(font=ff)

videoButton = Button(main, text="6.video", command=open_video)
videoButton.place(x=900, y=150)
videoButton.config(font=ff)

exitButton = Button(main, text="7.Exit", command=close)
exitButton.place(x=250, y=700)
exitButton.config(font=ff)



font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=350, y=200)
text.config(font=font1)

main.config(bg='#002147')
main.mainloop()



############use canvas for background image
