import mlflow
print(mlflow.get_tracking_uri())

#getting Error
'''When an mlflow-artifacts URI was supplied, 
the tracking URI must be a valid http or https URI, 
but it was currently set to 
file:///Users/sacx/MLOps%20Series%20-%20Vikas%20Das/Mlops-MLFlow/mlruns.'''

'''so I have to set the tracking URI to a valid http or https URI in Expfile.py'''
to '''mlflow.set_tracking_uri("http://127.0.0.1:5000")'''