from flask import Flask, render_template, request
import pickle
import os
from PIL import Image 
from numpy import asarray 

app = Flask(__name__,template_folder='html')
app.config['UPLOAD_FOLDER']='uploads'

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/ocr',methods=['POST'])
def get_ocr():
    # with open('final.model', 'rb') as file:  
    #     modell = pickle.load(file)
    if(request.method=="POST"):
        image= request.files['image']
        path= os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path)
        return get_character(path)
        # result="Prediction: {}".format(modell.predict(input_test[k].reshape(1,-1))[0])  

def get_character(pathvar):
    image= Image.open(pathvar)
    arrx= asarray(image)
    with open('final.model', 'rb') as file:  
        modell = pickle.load(file)
    return "{"+"char:"+modell.predict(arrx.reshape(1,-1))[0]+"}"
