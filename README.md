# Disaster Response Pipeline Project

### Project Summary:
The project is part of the Data Science Nanodegree offered by Udacity in collaboration with Figure8.

Following a desaster, there are millions of communications via social media or directly. When a disaster occurs, disaster response organizations have the least capacity to filter for relevant messages. 

Machine Learning solutions can help categorizing the messages correctly and distribute them to the responsible organization, to quickly react where it is indeed needed.

### Files:

- ./data/disaster_messages.csv: contains messages provided by figure8 sent during natural desasters, either via social media or directly to disaster response organizations

- ./data/disaster_categories.csv: comprises the relevant categories of each message in the former file (csv files can be joined on "id" column)

- ./data/process_data.py: the ETL pipeline which extracts the data, cleans and transforms it, and finally loads it into a SQLight database

- ./data/DisasterResponse.db: SQLight database, where the merged, cleaned and transformed data is stored/loaded

- ./models/train_classifier.py: the ML pipeline, which loads the data from the database and trains a multi-output supervised learning model (Multinomial Naive Bayes classifier )

- ./models/classifier.pkl: pickle file, where the trained model is stored

- ./app/run.py: launches the web app to classify messages

- ./app/templates/go.html and ./app/templates/master.html: html templates for the web app

- Contains the web app which extracts data from the database and uses the ML model to classify new messages for each of the categories

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Acknowledgements:

**Udacity** for setting up this awesome project in their [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025?promo=year_end&coupon=SKILLS50&utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=143524475679&utm_keyword=data%20scientist%20udacity_e&utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=143524475679&utm_keyword=data%20scientist%20udacity_e&gad_source=1&gclid=EAIaIQobChMIlNyctJC_hgMVJ6loCR3eowY6EAAYASAAEgJhzPD_BwE).

**Figure8** for providing the data and context.
