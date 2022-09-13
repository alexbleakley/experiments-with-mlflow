# Experiments in CML with MLflow

This repository demonstrates use of CML's Experiments v2 functionality using the MLflow 
client library to log metrics and models while iterating on a model training script.


## Prerequisites

This project is designed to be used in a CML workspace with Experiments v2 enabled.
The steps detailed below were tested using a Python 3.7 CML Runtime.


## Setup

1. Create a new CML Project from this repository by providing its Git URL.

2. Start a new Session within the Project.

3. In the command prompt, type `!pip3 install -r requirements.txt` to install the 
Python dependencies for this project.


## Demonstration

4. Navigate to the Project's "Experiments" page by selecting it in the left navigation bar. 
Note that no Experiments have been created yet.

5. In a second tab, start a new Session within the Project.

6. Open the `train_model.py` file. 
The code in this file trains a simple linear regression model to predict the number of 
riders using a bikeshare program on a given day, provided certain attributes of the day 
such as the weather, and whether it is a weekday. 
Note that the code in this file makes calls using 
the `mlflow` library to log an MLflow Experiment Run, with a combination of autologging 
(using `.sklearn.autolog`) and manual logging of metrics (using `.log_metric`).

7. Run all code in the file (Run> Run All).

8. Switching back to the first tab, refresh the Experiments table to show the new 
`bikeshare` Experiment. Click on the Experiment to show the Runs table, and note the 
Run that we just created, including the latest Git commit hash, 
various autologged metrics, and the `train_score` and `test_score` metrics that we 
manually logged.

9. We think that we can get a higher accuracy than this, so we'll switch back to the 
Session tab and make some changes to our code. Uncomment line 34, where we define 
`one_hot_encoder`, and line 40, where we include it in the pipeline. Make sure that you 
have the correct whitespace on these two lines (none for line 34, 2 chars for line 40). 
This change will replace the "Season" column with a one-hot array, reflecting the fact 
that "Season" is not suitable as a linear variable.

10. Commit the change by typing `!git commit -am "Add one hot encoder"` in the command 
prompt. There is no need to push the commit for the purposes of this demonstration.

11. Run the code to create a new Run based on this modified model training script.

12. Switching back to the first tab, refresh the Experiments table to show the new 
Run. Note the new Run that we just created, including the latest Git commit hash, 
various autologged metrics, and the `train_score` and `test_score` metrics that we 
manually logged. Note that our small feature engineering step increased both the train 
and test scores.

13. If a little feature engineering is good, hopefully a lot is even better.
Switching back to the Session tab, uncomment line 35, where we define 
`polynomial_features`, and line 41, where we include it in the pipeline. Make sure that you 
have the correct whitespace on these two lines (none for line 35, 2 chars for line 41). 
This change will create many new features by multiplying together pairs of the 
existing features.

14. Commit the change by typing `!git commit -am "Add polynomial features"` in the command 
prompt. There is no need to push the commit for the purposes of this demonstration.

15. Run the code to create a new Run based on this modified model training script.

16. Switching back to the first tab, refresh the Experiments table to show the new 
Run. Note the new Run that we just created, including the latest Git commit hash, 
various autologged metrics, and the `train_score` and `test_score` metrics that we 
manually logged. Unfortunately our latest attempt at feature engineering created an 
overfitting problem: we increased the train score but decreased the test score.

17. Note that we can evaluate all of the models we have created so far, comparing 
them in the table and using charts. Even though the last model we created was not 
the best model, we can choose the model that performed best, the second 
model, and easily identify the Git commit used to create it. We can also register 
that model to our Model Registry (future demo).
