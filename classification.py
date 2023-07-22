import customtkinter
from tkinter import messagebox, filedialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models


class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# load the model
model = models.load_model('img_class.model')


def classification(model, img_root):
    img = cv.imread(img_root)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img, cmap=plt.cm.binary)
    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    prediction = class_name[index]
    messagebox.showinfo(message=prediction)



def path():
    rootfile = filedialog.askopenfilename(initialdir='C:/Users/Mohamed/Desktop/computer_vision2/Images'
                                          , title='Select Image'
                                          , filetypes=(("jpg files", "*.jpg"), ("png files", "*.png")
                                                       , ("all files", "*.*")))
    return rootfile


def activeclass():
    classification(model, root1)


def change():
    global root1
    root1 = path()
    return root1


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("Imgae_Classification")
root.geometry('600x600')

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

lable = customtkinter.CTkLabel(master=frame, text='Imgae_Classification', text_color='#095783', font=('', 40))
lable.pack(pady=10, padx=20)

lable = customtkinter.CTkLabel(master=frame, text='Choose The Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)

btn1 = customtkinter.CTkButton(master=frame, text="show", command=activeclass)
btn1.pack(pady=10)

btn1 = customtkinter.CTkButton(master=frame, text="Choose Image", command=change)
btn1.pack(pady=10)


def change_appearance_mode_event(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)


appearance_mode_label = customtkinter.CTkLabel(root, text="Appearance Mode:", anchor="w")
appearance_mode_label.pack()
appearance_mode_optionemenu = customtkinter.CTkOptionMenu(root, values=["Light", "Dark"]
                                                          , command=change_appearance_mode_event)
appearance_mode_optionemenu.pack(pady=10)

appearance_mode_optionemenu.set("Dark")

root.mainloop()
