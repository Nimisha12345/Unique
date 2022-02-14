from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 



img_size=100

app = Flask(__name__)   #flask is used to connect html with python code

model=load_model('model-007.model')

label_dict = {'COVID19': 0, 'NORMAL': 1, 'PNEUMONIA': 2}

def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped

def models(X_train,Y_train):
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    print('Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    return tree


def fromsymptoms(cough,fever,sore_throat,breath,headache,age):
	df = pd.read_csv('covid_symptoms.csv')

	X = df.iloc[:, 1:7].values #parameters will be added
	Y = df.iloc[:, -1].values #covid result
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

	model = models(X_train,Y_train)
	predic = [cough,fever,sore_throat,breath,headache,age]
	p = np.array(predic)

	if model.predict(p.reshape(1,-1))==[1]:
		return "COVID POSITIVE"
	else:
		return "COVID NEGATIVE"


@app.route("/")
def index():
	return(render_template("nirmiti.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)  #to store input data
	encoded = message['image']
	age = int(message['age'])
	if age<60:
		age = 0
	else:
		age = 1

	temp = float(message['temp'])
	if temp<98.6:
		fever = 0
	else:
		fever = 1
	
	cough = message['cough']
	sore_throat= message['sore_throat']
	breathing = message['breathing']
	headache = message['headache']
	sym = fromsymptoms(cough,fever,sore_throat,breathing,headache,age)
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]   #maximium value of accuracy is considered
	accuracy=float(np.max(prediction,axis=1)[0])

	for k, v in label_dict.items():
         if v == result:
            label = k

	print(headache)
	
	print(prediction,result,accuracy)
	

	response = {'prediction': {'result': label,'accuracy': accuracy, 'symptoms': sym}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">