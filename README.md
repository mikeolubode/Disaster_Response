# Disaster Response Pipeline Project
## Project Summary
The project creates a web for disaster response in a time of crisis. It builds a machine learning model on the likely response to a disaster message which is used to predict what is needed in a time of disaster. This is needed because the time of disaster is when response team are least capable of filtering through disaster messages to identify what is needed.

### Files
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py  # python file for processing data
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py # python script for training classifier model
|- classifier.pkl  # saved model in pickle

- README.md # contains documentation of the project


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run the web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
