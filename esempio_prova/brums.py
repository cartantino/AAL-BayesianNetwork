import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel

values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])

train_data = values[:800]
predict_data = values[800:]

model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
model.fit(values)

predict_data = predict_data.copy()
e = predict_data.loc[:, 'E']
predict_data.drop('E', axis=1, inplace=True)

print("predict data E", e)
y_pred = model.predict_probability(predict_data)


print("y pred\n", y_pred)
