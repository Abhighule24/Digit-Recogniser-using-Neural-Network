import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
import Train

def preprocess(image):
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 50, 255, cv2.THRESH_BINARY_INV)

    contours, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = [cv2.contourArea(cnt) for cnt in contours]
    c = cnts.index(max(cnts))
    x,y,w,h = cv2.boundingRect(contours[c])
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)

    if h>=w:
        digit = thresh[y:y+h, x+(w//2)-(h//2):x+(w//2)+(h//2)]
    else:
        digit = thresh[y+(h//2)-(w//2):y+(h//2)+(w//2), x:x+w]

    kernel = np.ones((9,9),np.uint8)
    digit = cv2.dilate(digit,kernel,iterations = 2)

    resized_digit = cv2.resize(digit, (18,18), cv2.INTER_LINEAR)

    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

    return padded_digit

def predict(path):
    img = cv2.imread(path)
    pre_img = preprocess(img)
    img = pre_img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    digit = model.predict_classes(img)
    return digit[0]

def browse_img():
    img_path.set(filedialog.askopenfilename(title='Please Select a File'))
    guess = predict(img_path.get())
    img = Image.open(img_path.get())
    img = img. resize((300, 300), Image. ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    l0['text'] = "The Image You Choose is"
    l1.configure(width = "300", height = "300", image=img)
    l1.image = img
    l2["text"] = guess

from keras.models import load_model
try:
    model = load_model('modelx.h5')
except:
    Train.training()
    model = load_model('modelx.h5')

root = Tk()
root.geometry("500x600+700+200")
root.configure(bg = "black")
root.title("Digit Recogniser")

img_path = StringVar()
img_path.set("Select a Image with Number")
frm1 = Frame(root)
Entry(frm1, textvariable = img_path , width = 60,bg = "white", font = "consolas 11 bold").pack(pady = (10, 0))
Button(frm1, text = "BROWSE", bg = "white", command = lambda :browse_img(), font = "consolas 12 bold").pack(pady = 10)
l0 = Label(frm1,bg = "black", fg = "white", font = "consolas 24 bold")
l0.pack()
l1 = Label(frm1)
l1.configure(width = "100", height = "100",bg = "black")
l1.pack()
Label(frm1, text = "The Number is ", bg = "black", fg = "white", font = "consolas 24 bold").pack(pady = (10, 0))
l2 = Label(frm1)
l2.configure(width = "20", height = "20",bg = "black", fg = "red", font = "consolas 36 bold")
l2.pack()
frm1.configure(width = "500", height = "600",bg = "black")
frm1.pack()
root.resizable(False, False)
root.mainloop()


