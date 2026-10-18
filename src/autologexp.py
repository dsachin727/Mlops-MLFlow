import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


##using dagshub for mlflow tracking
import dagshub
dagshub.init(repo_owner='dsachin727', repo_name='Mlops-MLFlow', mlflow=True)

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

X_train, x_test, Y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 42)

max_depth = 5
n_estimeators = 50

#setting url for dagshub mlflow server
mlflow.set_tracking_uri("https://dagshub.com/dsachin727/Mlops-MLFlow.mlflow")

#autologging
mlflow.autolog()

#For naming own experiment to get runs in it (otherwise it will be in default experiment)
#Or Create experiment in mlflow ui and then use that name here
mlflow.set_experiment("Autolog experiment")

experiment = mlflow.get_experiment_by_name("Autolog experiment")

print("Experiment name:", experiment.name)
print("Experiment ID:", experiment.experiment_id)

with mlflow.start_run(): #Also we can add exp id inside start_run() to log runs in that experiment
    # Create and train the model
    model = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimeators, random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    #confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    #save plot as an image
    plt.savefig("confusion_matrix.png")

    # Log the confusion matrix plot as an artifact in MLFlow
    mlflow.log_artifact(__file__)

#tags
    mlflow.set_tags({"Author": "Sachin", "Model": "RandomForestClassifier"})

    print(f"Accuracy: {accuracy}")
