import pandas as pd
import random
from pathlib import Path

################################################################################
Path('./tests').mkdir(exist_ok=True)

# Number of samples
num_samples = 1500

# Lists for generating random data
sol_pred = []
submission_data = []

for i in range(num_samples):
    sol_pred.append([random.randint(0, 9)])
    submission_data.append([random.randint(0, 9), random.random()])

solution = pd.DataFrame(sol_pred, columns=['label'])
submission = pd.DataFrame(submission_data, columns=['label', 'latency'])

# Modify DataFrames for writing to CSV
solution.insert(1, 'Usage', 'Private')

################################################################################
# Count the number of correctly predicted samples
acc_dict = (solution.iloc[:, 0] ==
            submission.iloc[:, 0]).value_counts().to_dict()

# Compute the accuracy, latency, and score
accuracy = 100*(acc_dict[True] / len(solution))
latency = submission.iloc[:, 1].mean()
score = accuracy / latency
################################################################################

# Print the expected results
print(accuracy)
print(latency)
print(score)
print(len(solution))

# NOTE: Keep in mind these CSV files will NOT have the ID column name
# The name should be manually edited as 'id'
solution.to_csv('./tests/solution.csv')
submission.to_csv('./tests/submission.csv')

# Add the id column name
lines = []
with open('./tests/solution.csv', 'r') as f:
    lines = f.readlines()

with open('./tests/solution.csv', 'w+') as f:
    f.write('id,label,Usage\n')
    f.writelines(lines[1:])

lines = []
with open('./tests/submission.csv', 'r') as f:
    lines = f.readlines()

with open('./tests/submission.csv', 'w+') as f:
    f.write('id,label,latency\n')
    f.writelines(lines[1:])

# Write the expected results
with open('./tests/score.txt', 'w+') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Latency: {latency}\n')
    f.write(f'Score: {score}\n')
