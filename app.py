from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
app = Flask(__name__)


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def remove_collinear_features(x, threshold):
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    return x


@app.route('/')
def hello():
    return jsonify({'message': 'Hello, World!'})


@app.route('/predict', methods=['POST'])
def getPrediction():
    request_data = request.json
    credit = pd.DataFrame([request_data['data']], index=['ROW1'])
    credit[credit['Years of Credit History'].isnull() == True]
    credit.dropna(subset=['Maximum Open Credit'], inplace=True)
    for i in credit['Tax Liens'][credit['Tax Liens'].isnull() == True].index:
        credit.drop(labels=i, inplace=True)
    for i in credit['Bankruptcies'][credit['Bankruptcies'].isnull() == True].index:
        credit.drop(labels=i, inplace=True)
    # credit.fillna(credit.mean(), inplace=True)

    credit.fillna('10+ years', inplace=True)

    # # Encoding categorical data & Feature Scaling

    # Select the categorical columns
    categorical_subset = credit[[
        'Term', 'Years in current job', 'Home Ownership', 'Purpose']]

    print(credit.shape)
    # One hot encode
    categorical_subset = pd.get_dummies(categorical_subset)
    print(categorical_subset.shape)

    credit.drop(labels=['Term', 'Years in current job',
                        'Home Ownership', 'Purpose'], axis=1, inplace=True)
    credit = pd.concat([credit, categorical_subset], axis=1)
    print(credit.shape)

    credit = remove_collinear_features(credit, 0.6)

    sc = StandardScaler()
    transformed_data = sc.fit_transform(credit)
    model = joblib.load('./model.joblib')
    prediction = model.predict(transformed_data)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
