import streamlit as st;
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pickle
import os
from my_model.model import FacialExpressionModel
import time
from bokeh.models.widgets import Div

# importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("my_model/model.json", "my_model/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


# face exp detecting function
def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
		fc = gray[y:y + h, x:x + w]
		roi = cv2.resize(fc, (48, 48))
		pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
		cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		return img, faces, pred
	# the main function


def main():
	"""Face Expression Detection App"""
	# setting the app title & sidebar

	activities = ["Home", "Questions based predictor", "Detect your Facial expressions", "CNN Model Performance"]
	choice = st.sidebar.selectbox("Select Activity", activities)

	if choice == 'Home':
		st.title("Welcome to Happiness Index Predictor\n 😊😒😡😂")
		st.subheader("We predict your happiness index and give you recommendations to improve it just by taking few inputs from you")
		st.write("CREATED BY: Soham Kulkarni, Khemchandra Chaudhari , Mayur Patil, Nikhil Sangle")

	# loading questions based module
	if choice == 'Questions based predictor':
		def load_model():
			with open('ideal_model_2.pkl', 'rb') as file:
				data = pickle.load(file)
			return data

		def recommendations():
			st.subheader("To uplift your happiness index you can try the following Measures 💕")
			if Family_support < 5:
				st.write("Ohh your family support is less 😔 you can try following measures 👍 :\n"
						 "1. Don't be afraid to take social risks\n"
						 "2. Spend more time with your family members\n"
						 "2. Avoid negative relationships\n"
						 "3. Take care of your relationships\n"
						 "4. Try to think about other Person's perspective before coming to any diseason")
			if life_expectency < 75:
				st.write("um uhh  Your Life Expectancy is less than the average,Please try following things :\n"
						 "1. Eat Healthy food 🍅\n"
						 "1. Hang Out With Friends\n"
						 "2. Get Daily Exercise\n"
						 "3. Give more attention towards regular Checkups 🩺\n")
			if Freedom_of_choices < 5:
				st.write("Your Freedom of choices is less pal,try following things :\n"
						 "1. Always be honest with yourself\n"
						 "2. Try to surround yourself with competant people\n"
						 "3. If you want freedom then you have to grant others equal freedom \n")
			if Generosity < 4:
				st.write("Well now, you are one of less generous ones you can try following things :\n"
						 "1. Volunteer in your community\n"
						 "2. Use your skills and expertise to help\n"
						 "3. Check in on someone who might need help\n")
			if level_of_corruption > 4:
				st.write("Your are getting affected by corruption you can try following things to fight corruption  :\n"
						 "1. Report corruption arround you 👮‍\n"
						 "2. expose corrupt activities and risks that may otherwise remain hidden\n"
						 "3. ensure that public sector employees act in the public interest\n")
			else:
				st.write("Please Remain as Awesome as you are right now ")

		data = load_model()

		regressor = data["model"]
		le_country = data["country"]
		le_Family_support = data["Family_support"]
		le_life_expectency = data["life_expectency"]
		le_Freedom_of_choices = data["Freedom_of_choices"]
		le_Generosity = data["Generosity"]
		le_level_of_corruption = data["level_of_corruption"]

		st.title("Happiness Index prediction 😊")

		st.write("""### We need some information to predict the your happiness index""")

		countries = (
			'Finland', 'Denmark', 'Switzerland', 'Iceland', 'Netherlands',
			'Norway', 'Sweden', 'Luxembourg', 'New Zealand', 'Austria',
			'Australia', 'Israel', 'Germany', 'Canada', 'Ireland',
			'Costa Rica', 'United Kingdom', 'Czech Republic', 'United States',
			'Belgium', 'France', 'Bahrain', 'Malta',
			'Taiwan Province of China', 'United Arab Emirates', 'Saudi Arabia',
			'Spain', 'Italy', 'Slovenia', 'Guatemala', 'Uruguay', 'Singapore',
			'Kosovo', 'Slovakia', 'Brazil', 'Mexico', 'Jamaica', 'Lithuania',
			'Cyprus', 'Estonia', 'Panama', 'Uzbekistan', 'Chile', 'Poland',
			'Kazakhstan', 'Romania', 'Kuwait', 'Serbia', 'El Salvador',
			'Mauritius', 'Latvia', 'Colombia', 'Hungary', 'Thailand',
			'Nicaragua', 'Japan', 'Argentina', 'Portugal', 'Honduras',
			'Croatia', 'Philippines', 'South Korea', 'Peru',
			'Bosnia and Herzegovina', 'Moldova', 'Ecuador', 'Kyrgyzstan',
			'Greece', 'Bolivia', 'Mongolia', 'Paraguay', 'Montenegro',
			'Dominican Republic', 'North Cyprus', 'Belarus', 'Russia',
			'Hong Kong S.A.R. of China', 'Tajikistan', 'Vietnam', 'Libya',
			'Malaysia', 'Indonesia', 'Congo (Brazzaville)', 'China',
			'Ivory Coast', 'Armenia', 'Nepal', 'Bulgaria', 'Maldives',
			'Azerbaijan', 'Cameroon', 'Senegal', 'Albania', 'North Macedonia',
			'Ghana', 'Niger', 'Turkmenistan', 'Gambia', 'Benin', 'Laos',
			'Bangladesh', 'Guinea', 'South Africa', 'Turkey', 'Pakistan',
			'Morocco', 'Venezuela', 'Georgia', 'Algeria', 'Ukraine', 'Iraq',
			'Gabon', 'Burkina Faso', 'Cambodia', 'Mozambique', 'Nigeria',
			'Mali', 'Iran', 'Uganda', 'Liberia', 'Kenya', 'Tunisia', 'Lebanon',
			'Namibia', 'Palestinian Territories', 'Myanmar', 'Jordan', 'Chad',
			'Sri Lanka', 'Swaziland', 'Comoros', 'Egypt', 'Ethiopia',
			'Mauritania', 'Madagascar', 'Togo', 'Zambia', 'Sierra Leone',
			'India', 'Burundi', 'Yemen', 'Tanzania', 'Haiti', 'Malawi',
			'Lesotho', 'Botswana', 'Rwanda', 'Zimbabwe', 'Afghanistan',
			'Trinidad and Tobago', 'Macedonia', 'Congo (Kinshasa)',
			'Central African Republic', 'South Sudan', 'Taiwan', 'Qatar',
			'Trinidad & Tobago', 'Northern Cyprus', 'Hong Kong', 'Bhutan',
			'Somalia', 'Syria', 'Belize', 'Sudan', 'Angola',
			'Hong Kong S.A.R., China', 'Puerto Rico', 'Suriname',
			'Somaliland Region', 'Oman', 'Somaliland region', 'Djibouti'
		)

		country = st.selectbox("Select your country", countries)
		Family_support = st.slider("How much your family supports you ? ", 0, 10, 5)
		life_expectency = st.number_input(" What is your healthy life expectency (In Years)  ", 0, 100)
		Freedom_of_choices = st.slider("How much Freedom of choices do you have ", 0, 10, 5)
		Generosity = st.slider("How generous you are ? ", 0, 10, 5)
		level_of_corruption = st.slider("level by which corruption around you affects you ", 0, 10, 5)

		le = {'Finland': 1, 'Denmark': 2, 'Switzerland': 3, 'Iceland': 4, 'Netherlands': 5,
			  'Norway': 6, 'Sweden': 7, 'Luxembourg': 8, 'New Zealand': 9, 'Austria': 10,
			  'Australia': 11, 'Israel': 12, 'Germany': 13, 'Canada': 14, 'Ireland': 14,
			  'Costa Rica': 15, 'United Kingdom': 16, 'Czech Republic': 17, 'United States': 18,
			  'Belgium': 19, 'France': 20, 'Bahrain': 21, 'Malta': 22,
			  'Taiwan Province of China': 23, 'United Arab Emirates': 24, 'Saudi Arabia': 25,
			  'Spain': 26, 'Italy': 27, 'Slovenia': 28, 'Guatemala': 29, 'Uruguay': 30, 'Singapore': 31,
			  'Kosovo': 32, 'Slovakia': 33, 'Brazil': 34, 'Mexico': 35, 'Jamaica': 36, 'Lithuania': 37,
			  'Cyprus': 38, 'Estonia': 39, 'Panama': 40, 'Uzbekistan': 41, 'Chile': 42, 'Poland': 43,
			  'Kazakhstan': 44, 'Romania': 45, 'Kuwait': 46, 'Serbia': 47, 'El Salvador': 48,
			  'Mauritius': 49, 'Latvia': 50, 'Colombia': 51, 'Hungary': 52, 'Thailand': 53,
			  'Nicaragua': 54, 'Japan': 55, 'Argentina': 56, 'Portugal': 57, 'Honduras': 58,
			  'Croatia': 59, 'Philippines': 60, 'South Korea': 61, 'Peru': 62,
			  'Bosnia and Herzegovina': 63, 'Moldova': 64, 'Ecuador': 65, 'Kyrgyzstan': 66,
			  'Greece': 67, 'Bolivia': 68, 'Mongolia': 69, 'Paraguay': 70, 'Montenegro': 71,
			  'Dominican Republic': 72, 'North Cyprus': 73, 'Belarus': 74, 'Russia': 75,
			  'Dominican Republic': 76, 'North Cyprus': 77, 'Belarus': 78, 'Russia': 79,
			  'Dominican Republic': 80, 'North Cyprus': 81, 'Belarus': 82, 'Russia': 83,
			  'Hong Kong S.A.R. of China': 84, 'Tajikistan': 85, 'Vietnam': 86, 'Libya': 87,
			  'Malaysia': 88, 'Indonesia': 89, 'Congo (Brazzaville)': 90, 'China': 91,
			  'Ivory Coast': 92, 'Armenia': 93, 'Nepal': 94, 'Bulgaria': 95, 'Maldives': 96,
			  'Azerbaijan': 97, 'Cameroon': 98, 'Senegal': 99, 'Albania': 100, 'North Macedonia': 101,
			  'Ghana': 102, 'Niger': 103, 'Turkmenistan': 104, 'Gambia': 105, 'Benin': 106, 'Laos': 107,
			  'Bangladesh': 108, 'Guinea': 109, 'South Africa': 110, 'Turkey': 111, 'Pakistan': 112,
			  'Morocco': 113, 'Venezuela': 114, 'Georgia': 115, 'Algeria': 116, 'Ukraine': 117, 'Iraq': 118,
			  'Gabon': 119, 'Burkina Faso': 120, 'Cambodia': 121, 'Mozambique': 122, 'Nigeria': 123,
			  'Mali': 124, 'Iran': 125, 'Uganda': 126, 'Liberia': 127, 'Kenya': 128, 'Tunisia': 129, 'Lebanon': 130,
			  'Namibia': 131, 'Palestinian Territories': 132, 'Myanmar': 133, 'Jordan': 134, 'Chad': 135,
			  'Sri Lanka': 136, 'Swaziland': 137, 'Comoros': 138, 'Egypt': 139, 'Ethiopia': 140,
			  'Mauritania': 141, 'Madagascar': 142, 'Togo': 143, 'Zambia': 144, 'Sierra Leone': 145,
			  'India': 146, 'Burundi': 147, 'Yemen': 148, 'Tanzania': 149, 'Haiti': 150, 'Malawi': 151,
			  'Lesotho': 152, 'Botswana': 153, 'Rwanda': 154, 'Zimbabwe': 155, 'Afghanistan': 156,
			  'Trinidad and Tobago': 157, 'Macedonia': 158, 'Congo (Kinshasa)': 159,
			  'Central African Republic': 160, 'South Sudan': 161, 'Taiwan': 162, 'Qatar': 163,
			  'Trinidad & Tobago': 164, 'Northern Cyprus': 165, 'Hong Kong': 166, 'Bhutan': 167,
			  'Somalia': 168, 'Syria': 169, 'Belize': 170, 'Sudan': 171, 'Angola': 172,
			  'Hong Kong S.A.R., China': 173, 'Puerto Rico': 174, 'Suriname': 175,
			  'Somaliland Region': 176, 'Oman': 177, 'Somaliland region': 178, 'Djibouti': 179}
		ok = st.button("Calculate Happiness Index")
		if ok:
			x = np.array(
				[[country, Family_support, life_expectency, Freedom_of_choices, Generosity, level_of_corruption]])
			temp = x[:, 0]
			temp1 = temp[0]
			x[:, 0] = le.get(temp1)
			x = x.astype(float)

			happines_index = regressor.predict(x)
			st.subheader(f"The estimated Happiness index is {happines_index[0]:.2f}")
			recommendations()

	# if choosing to consult the cnn model performance

	if choice == 'CNN Model Performance':
		st.title("Face Expression WEB Application :")
		st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
		st.subheader("CNN Model :")
		st.image('images/model.png', width=700)
		st.subheader("FER2013 Dataset from:")
		st.text(
			" https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
		st.image('images/dataframe.png', width=700)
		st.subheader("Model training results:")
		st.markdown("Accuracy :chart_with_upwards_trend: :")
		st.image("images/accuracy.png")
		st.markdown("Loss :chart_with_downwards_trend: : ")
		st.image("images/loss.png")
	# if choosing to detect your face exp , give access to upload the image

	if choice == 'Detect your Facial expressions':
		st.title("Detect emotion using uploaded image ")
		st.subheader("Please Fill the survey form to get best recommendations to Uplift your Happiness:")
		st.write("Please Enter only one input per datafield")
		hobby = st.text_input("What is your Hobby")
		t_h = st.text_input("What things do you do when you are happy")
		t_s = st.text_input("What things do you do when you are sad")
		music = st.text_input("Which genre of music do you like")
		books = st.text_input("Which kind of books do you like ")
		st.subheader("Upload a Image to know happiness index")
		# st.subheader(":smile: :worried: :fearful: :rage: :hushed:")

		image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

		# if image if uploaded,display the progress bar +the image
		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Original Image")
			progress = st.progress(0)
			for i in range(100):
				time.sleep(0.01)
				progress.progress(i + 1)
			st.image(our_image)
		if image_file is None:
			st.error("No image uploaded yet")

		# Face Detection
		task = ["Faces"]
		feature_choice = st.sidebar.selectbox("Find Features", task)
		if st.button("Process"):
			if feature_choice == 'Faces':

				# process bar
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.05)
					progress.progress(i + 1)
				# end of process bar

				result_img, result_faces, prediction = detect_faces(our_image)
				if st.image(result_img):

					if prediction == 'Happy':
						st.subheader("YeeY!  You are Happy :smile: today , Always Be ! ")

					elif prediction == 'Angry':
						st.subheader("You seem to be angry :rage: today ,Take it easy! ")
						st.subheader("Recommendation for you : \n"
									 "1. Try doing meditation 🧘‍\n"
									f"2. Why don't you  try listning to your favourite {music} music"
									 )

					elif prediction == 'Disgust':
						st.subheader("You seem to be Disgust :rage: today! ")

					elif prediction == 'Fear':
						st.subheader("You seem to be Fearful :fearful: today ,Be couragous! ")

					elif prediction == 'Neutral':
						st.subheader("You seem to be Neutral today , Have a good day!!")
						st.subheader("Recommendation for you : \n"
									 f"1. Why don't you  try reading your favourite {books} books to add more Spice  "
									 )


					elif prediction == 'Sad':
						st.subheader("You seem to be Sad :sad: today ,Smile and be happy! ")
						st.subheader("Recommendation for you : \n"
									 f"1. Why don't Try doing {hobby}\n"
									 f"2. Why don't you  try listning to your favourite {music} music"
									 )

					elif prediction == 'Surprise':
						st.subheader("You seem to be surprised today ! ")

					else:
						st.error("Your image does not match the training dataset's images! Try an other image!")


main()
