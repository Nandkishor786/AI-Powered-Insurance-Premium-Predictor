
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field
from typing import Literal,Optional,Annotated
import pandas as pd
import pickle
import joblib
#to coonect the deployed frontent with deployed backend we need cors 
from fastapi.middleware.cors import CORSMiddleware


#pickle and joblib are library used to load and save python object and ml models in porjects
# 1..............................................
#pickle library used to load(import) and save the python object ,ml models etc.
# way1 with  sckit-learn 1.6.1
# with open('model.pkl',"rb") as f:
#  model  =pickle.load(f)

# way2 with  sckit-learn 1.7.2
# Load the model
model = joblib.load("model.joblib")

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

app  =FastAPI()


# Streamlit URL ya "*" (dev) add karo
origins = [
    "https://ai-powered-insurance-premium-predictor-mzenjse9ftcyvkt6dxwgdt.streamlit.app",
    "https://your-custom-domain.com",  # agar koi ho
    "http://localhost:8501",            # local testing
    "https://localhost:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production ke liye specific origins list karo instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2........................................
#  pydantic model to validate incomming data....
# Literals -used where we need optional values
# ... denoted required field
#raw data filed validation comming from clients
class UserInput(BaseModel):
  age:Annotated[int,Field(...,gt=0,lt=120,description="Age of the user")]
  weight:Annotated[float,Field(...,gt=0,description="Weight of the user")]
  height:Annotated[float,Field(...,gt=0,lt=2.5, description="Height of the user")]
  income_lpa:Annotated[float,Field(...,gt=0,description="Annual salary of the user in lpa")]
  smoker:Annotated[bool,Field(...,description="Is user a smoker")]
  city:Annotated[str,Field(...,description="The city in which the user belongs to")]
  occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'],Field(...,description="Occupation of the user")]

  @computed_field
  @property
  def bmi(self)->float:
   return self.weight/(self.height**2)
  
  @computed_field
  @property
  def lifestyle_risk(self)->str:
    if self.smoker and self.bmi > 30:
        return "high"
    elif self.smoker or self.bmi > 27:
        return "medium"
    else:
        return "low"
  @computed_field
  @property
  def age_group(self)->str:
    if self.age < 25:
        return "young"
    elif self.age < 45:
        return "adult"
    elif self.age < 60:
        return "middle_aged"
    return "senior"

  @computed_field
  @property
  def city_tier(self)->int:
    if self.city in tier_1_cities:
        return 1
    elif self.city in tier_2_cities:
        return 2
    else:
        return 3
     

@app.get('/')
def hello():
  return {"message":"Wellcome to my IPP Model"}

@app.get('/about')
def about():
  return {"message":"Insurance Premium Predictor"}  


# 3................................
# ENDPOINT-API FOR ACCESS ML MODEL BY USER
# here we create routes for user like '/predict'
#    on which user hit and get acess of ml model api

#data:UserInput-data of requestbody(from client) is store in data variable and which is a object variable of pydantic class model where it validate

@app.post('/predict')
def predict_premium(data:UserInput):
 
# now here we create first input data formate for ml  model 
# we have to send one row wise data into ml model
# as our model trained using randomforest algo in data frame object formate 
# so we send data as pandas data frames  to the ml model
 input_df = pd.DataFrame([{
    'bmi':data.bmi,
    'age_group':data.age_group,
    'lifestyle_risk':data.lifestyle_risk,
    'city_tier':data.city_tier,
    'income_lpa':data.income_lpa,
    'occupation':data.occupation
 }])


 # server(give) proper input data to ml model
 prediction = model.predict(input_df)[0]
 # predict() is method of ml model  which take input as dict and give response as list we our result at 0 index of list

 return JSONResponse(status_code=200,content={"Predicted_category":prediction})



#4. FRONTEND PART:.........................
# we create ml model and end point also
# now we create frontends for it 
# we can use Streamlit(python library for ui) or html css js
# create frontend.py
# pip install streamlit
# streamlit run frontend.py


