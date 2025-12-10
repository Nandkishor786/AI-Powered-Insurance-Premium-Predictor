
# we provide the access of ml models to users(clients) through fastapi(serving ml models using fastapi)

# INSORANCE PREMIUM(IP) PREDICTION (IPP) ML MODEL...(RandomForest Algo)
 IP is amount which is user have to pay annualy based on its insurance policy 
 IP depends on multiple factors 
 like health insurance
 its depends on life styles,age health etc.
so we create a ml model which predict the IP  of any  user based on its factor(features) given by him
it predict categorical data(IP) like high,medium,low

# 3 parts of this video
1. build ml model(using google collapse)
2. endpoints by fastapi to provide access to users
3. UI for user (frontend)-either using streamlet or html,css,js
   

# working: ml models <->api <-> frontend
1. client send the data from frontent to api
2. api valiadte the data and interact with ml model
3. ml models give predictions based on that data and resonse send to api 
4. then api send this response to frontent where user can see its ul predictions


# ML MODEL:(uisng google collapse notebook)
strategy:
# prepare input data for traing  ml 
we do the feature engineering means we create a new feature using exising feature either by combining them or modfiying them 
<!-- we take raw data from user but we convert it into new formate for  taining ml model -->
# input data:
1. age_group:we make it categorical from numerical
2. bmi:create using combining height and weight
lifestyle_risk:create using combining bmi and smoker fields 
bmi high +smoker =high risk
bmi perfect +no smoker =low risk
3. city_tier: make numerical from categorical city feature
4. income_lpa and occupation we same as given by user

# we create a ml models stored in model.pkl file...



# FASTAPI-endpoints
we use POST methods:means post is used to send data to server(ml model) to processed there and give result back

working: raw data from user->valiadaet data->transform into ml input data  -> send to ml models

