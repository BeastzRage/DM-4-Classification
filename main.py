from pathlib import Path

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42

def evaluate_model(name, model, x_train, y_train, x_valid, y_valid):
    """
     Print the performance metrics for a given classifier on both training and validation data.

     :param name : str
        Name of the model.
     :param model : sklearn classifier
        Trained classifier.
     :param x_train : pandas.DataFrame
        Processed training data.
     :param y_train : pandas.Series
        Training labels, where 0 means low income and 1 means high income.
     :param x_valid : pandas.DataFrame
        Processed validation data.
     :param y_valid : pandas.Series
        Validation labels, where 0 means low income and 1 means high income.
     """
    train_predictions = model.predict(x_train)
    valid_predictions = model.predict(x_valid)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    valid_probabilities = model.predict_proba(x_valid)[:, 1]

    train_accuracy = accuracy_score(y_train, train_predictions)
    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    train_auc = roc_auc_score(y_train, train_probabilities)
    valid_auc = roc_auc_score(y_valid, valid_probabilities)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Train accuracy:      {train_accuracy:.3f}")
    print(f"Validation accuracy: {valid_accuracy:.3f}")
    print(f"Train AUC:           {train_auc:.3f}")
    print(f"Validation AUC:      {valid_auc:.3f}")
    print(f"AUC overfit gap:     {train_auc - valid_auc:.3f}")

    print("\nClassification report on validation set:")
    print(classification_report(y_valid, valid_predictions, target_names=["low", "high"], zero_division=0)
    )


def select_features(x_train, y_train, x_valid, number_of_features):
    """
    Select and returns a subset of features using the training data.

    Parameters
    ----------
    :param x_train : pandas.DataFrame
        Processed training data.
    :param y_train : pandas.Series
        Training labels, used to score how useful each feature is.
    :param x_valid : pandas.DataFrame
        Processed validation data.
    :param number_of_features : int or str
        Number of features to keep. "all" keeps all features.

    :return pandas.DataFrame
        Training data with only the selected features.
    :return pandas.DataFrame
        Validation data with only the selected features.
    :return list
        Names of the selected features.
    """
    if number_of_features == "all":
        return x_train, x_valid, x_train.columns.tolist()

    selector = SelectKBest(k=number_of_features)
    selector.fit(x_train, y_train)

    selected_columns = x_train.columns[selector.get_support()].tolist()
    x_train_selected = x_train[selected_columns]
    x_valid_selected = x_valid[selected_columns]

    return x_train_selected, x_valid_selected, selected_columns


def tune_model(model, parameter_grid, x_train, y_train):
    """
    Find the best hyperparameters for a model using cross-validation.

    :param model : sklearn classifier
        Classifier whose hyperparameters should be tuned.
    :param parameter_grid : dict
        Dictionary containing the hyperparameter values to try.
    :param x_train : pandas.DataFrame
        Processed training data.
    :param y_train : pandas.Series
        Training labels, where 0 means low income and 1 means high income.

    :return sklearn.model_selection.GridSearchCV
        Fitted grid search object. get best model using object.best_estimator_
    """
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring="roc_auc", cv=cross_validation, n_jobs=1, refit=True)

    search.fit(x_train, y_train)
    return search

def explain_prediction_with_lime(explainer, prediction_function, validation_probabilities, x_valid_lime, x_valid_display, row_number):
    """
    Explain one validation prediction with LIME. saves the explanation to an html file

    :param explainer : lime.lime_tabular.LimeTabularExplainer
        Fitted LIME explainer.
    :param prediction_function : method
        Function that receives LIME rows and returns class probabilities from the final model.
    :param validation_probabilities : numpy.ndarray
        Predicted probabilities for all validation rows.
    :param x_valid_lime : pandas.DataFrame
        Validation data in LIME format.
    :param x_valid_display : pandas.DataFrame
        Validation data in its original readable format.
    :param row_number : int
        Index of the validation row to explain.
    """
    lime_row = x_valid_lime.iloc[row_number]
    display_row = x_valid_display.iloc[row_number]

    predicted_probabilities = validation_probabilities[row_number]
    predicted_class = predicted_probabilities.argmax()

    class_names = {0: "low", 1: "high"}

    explanation = explainer.explain_instance(data_row=lime_row.values, predict_fn=prediction_function, labels=[1], num_features=10)

    print("\n" + "=" * 70)
    print(f"LIME explanation for validation row {row_number}")
    print("=" * 70)

    print("\nOriginal person:")
    print(display_row)

    print("\nPrediction:")
    print("Predicted class:", class_names[predicted_class])
    print("Probability low: ", round(predicted_probabilities[0], 3))
    print("Probability high:", round(predicted_probabilities[1], 3))

    print("\nFeatures that influenced the prediction of high income:")
    for feature, weight in explanation.as_list(label=1):
        if weight > 0:
            direction = "increases probability of high income"
        else:
            direction = "decreases probability of high income"

        print(f"{feature}: {weight:.4f} ({direction})")

    html_file = f"data/lime_explanation_row_{row_number}.html"
    explanation.save_to_file(str(html_file))

def main():
    # load data
    df = pd.read_csv("data/income.csv")

    # drop column as too few values are present
    df.drop(columns=["ability to speak english"], axis=1)

    # filling missing values
    df["ability to speak english"] = df["ability to speak english"].fillna(0.0)
    df["gave birth this year"] = df["gave birth this year"].fillna("No")

    x = df.drop(columns=["income"])
    y = df["income"].map({"low": 0, "high": 1})

    # Splitting into training and validation data
    x_train_raw, x_valid_raw, y_train, y_valid = train_test_split(x, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    numeric_columns = ['age', 'education', 'workinghours', 'ability to speak english']
    categorical_columns = ['workclass', 'marital status', 'occupation', 'sex', 'gave birth this year']

    #Step 1: standardize numeric columns
    scaler = StandardScaler()
    scaler.fit(x_train_raw[numeric_columns])

    x_train_numeric_scaled = pd.DataFrame(scaler.transform(x_train_raw[numeric_columns]), columns=numeric_columns, index=x_train_raw.index)
    x_valid_numeric_scaled = pd.DataFrame(scaler.transform(x_valid_raw[numeric_columns]), columns=numeric_columns, index=x_valid_raw.index)

    #Step 2: one-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(x_train_raw[categorical_columns])

    encoded_column_names = encoder.get_feature_names_out(categorical_columns)

    x_train_categorical_encoded = pd.DataFrame(encoder.transform(x_train_raw[categorical_columns]), columns=encoded_column_names, index=x_train_raw.index)
    x_valid_categorical_encoded = pd.DataFrame(encoder.transform(x_valid_raw[categorical_columns]), columns=encoded_column_names, index=x_valid_raw.index)

    #Step 3: combine numeric and categorical columns
    x_train_processed = pd.concat([x_train_numeric_scaled, x_train_categorical_encoded], axis=1)
    x_valid_processed = pd.concat([x_valid_numeric_scaled, x_valid_categorical_encoded], axis=1)




    print("\n" + "=" * 70)
    print("Logistic regression")
    print("=" * 70)

    logistic_train, logistic_valid, logistic_features = select_features(x_train_processed, y_train, x_valid_processed, number_of_features="all")


    logistic_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)

    logistic_search = tune_model(model=logistic_model, parameter_grid={"C": [0.1, 1.0, 10.0]}, x_train=logistic_train, y_train=y_train)

    evaluate_model("Logistic regression", logistic_search.best_estimator_, logistic_train, y_train, logistic_valid, y_valid)

    print("\n" + "=" * 70)
    print("Decision tree")
    print("=" * 70)

    decision_tree_train, decision_tree_valid, decision_tree_features = select_features(x_train_processed, y_train, x_valid_processed, number_of_features="all")

    decision_tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    decision_tree_search = tune_model(model=decision_tree_model, parameter_grid={
            "max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 10, 25, 50], }, x_train=decision_tree_train, y_train=y_train)

    evaluate_model("Decision tree", decision_tree_search.best_estimator_, decision_tree_train, y_train, decision_tree_valid, y_valid)

    print("\n" + "=" * 70)
    print("Random forest")
    print("=" * 70)

    random_forest_train, random_forest_valid, random_forest_features = select_features(x_train_processed, y_train, x_valid_processed, number_of_features="all")

    random_forest_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)

    random_forest_search = tune_model(model=random_forest_model, parameter_grid={
            "max_depth": [6, 10, None], "min_samples_leaf": [1, 10, 25], "max_features": ["sqrt", 0.6], }, x_train=random_forest_train, y_train=y_train)

    evaluate_model("Random forest", random_forest_search.best_estimator_, random_forest_train, y_train, random_forest_valid, y_valid)



    print("\n" + "=" * 70)
    print("Feature selection example")
    print("=" * 70)

    selected_train, selected_valid, selected_features = select_features(x_train_processed, y_train, x_valid_processed, number_of_features=16)

    print("Selected 16 features:")
    print(selected_features)

    selected_forest = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=10, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1)
    selected_forest.fit(selected_train, y_train)

    evaluate_model("Random forest with 16 selected features", selected_forest, selected_train, y_train, selected_valid, y_valid)



    print("\n" + "=" * 70)
    print("Overfitting comparison: unrestricted models")
    print("=" * 70)

    unrestricted_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    unrestricted_tree.fit(x_train_processed, y_train)

    evaluate_model("Decision tree without overfitting controls", unrestricted_tree, x_train_processed, y_train, x_valid_processed, y_valid)

    unrestricted_forest = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)
    unrestricted_forest.fit(x_train_processed, y_train)

    evaluate_model("Random forest without overfitting controls", unrestricted_forest, x_train_processed, y_train, x_valid_processed, y_valid)







    # model explainability with LIME

    final_model = random_forest_search.best_estimator_

    # retrieve the importance scores the model assigned to each feature and save it to a csv
    feature_importance = pd.DataFrame({"feature": x_train_processed.columns, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
    feature_importance.to_csv("data/feature_importance.csv", index=False)


    # prepare data in such a way that lime shows the actual numerical/categorical values instead of the standardized scalar ones
    x_train_lime = x_train_raw.copy()
    x_valid_lime = x_valid_raw.copy()
    x_valid_display = x_valid_raw.copy()

    categorical_value_names = {}
    categorical_value_to_number = {}

    for column in categorical_columns:
        #get all possible values e.g. ['No', 'Yes'] for gave birth this year
        values = sorted(x_train_raw[column].unique().tolist())
        categorical_value_names[column] = values

        #make a mapping from number to value e.g. {0: 'no', 1: 'yes'}
        categorical_value_to_number[column] = {value: number for number, value in enumerate(values)}

        # apply mapping to columns
        x_train_lime[column] = x_train_lime[column].map(categorical_value_to_number[column])
        x_valid_lime[column] = x_valid_lime[column].map(categorical_value_to_number[column])

    # column indexes with categorical features
    categorical_feature_indexes = [1, 3, 4, 6, 8]

    # dictionary mapping categorical column indexes to list of possible values in that column
    categorical_names_for_lime = {x_train_lime.columns.get_loc(column): categorical_value_names[column] for column in categorical_columns}

    def prepare_lime_rows_for_model(lime_rows):
        """
        preforms normalization on scalar columns and one hot encoding on categorical columns on lime formatted data to be given to a model
        :param lime_rows: data in lime format
        :return: processed data
        """

        lime_rows_as_dataframe = pd.DataFrame(lime_rows, columns=x_train_lime.columns)
        original_style_rows = lime_rows_as_dataframe.copy()

        for column in categorical_columns:
            code_to_value = {number: value for value, number in categorical_value_to_number[column].items()}
            original_style_rows[column] = (original_style_rows[column].round().astype(int).map(code_to_value))

        numeric_part = pd.DataFrame(scaler.transform(original_style_rows[numeric_columns]), columns=numeric_columns)

        categorical_part = pd.DataFrame(encoder.transform(original_style_rows[categorical_columns]), columns=encoded_column_names)

        processed_rows = pd.concat([numeric_part, categorical_part], axis=1)
        return processed_rows

    def predict_for_lime(lime_rows):
        processed_rows = prepare_lime_rows_for_model(lime_rows)
        return final_model.predict_proba(processed_rows)

    # build lime explainer
    lime_explainer = LimeTabularExplainer(training_data=x_train_lime.values, feature_names=x_train_lime.columns.tolist(),
                                          class_names=["low", "high"], categorical_features=categorical_feature_indexes,
                                          categorical_names=categorical_names_for_lime, mode="classification", random_state=RANDOM_STATE)

    # get the 2 people who the model predicts have the highest probability of having high income and low income
    validation_probabilities = final_model.predict_proba(x_valid_processed)[:, 1]
    high_income_prediction = (validation_probabilities.argmax())
    low_income_prediction = (validation_probabilities.argmin())

    # explain prediction for person with predicted high income
    explain_prediction_with_lime(explainer=lime_explainer, prediction_function=predict_for_lime,
                                 validation_probabilities=final_model.predict_proba(x_valid_processed), x_valid_lime=x_valid_lime,
                                 x_valid_display=x_valid_display, row_number=high_income_prediction)
    # explain prediction for person with predicted low income
    explain_prediction_with_lime(explainer=lime_explainer, prediction_function=predict_for_lime,
                                 validation_probabilities=final_model.predict_proba(x_valid_processed), x_valid_lime=x_valid_lime,
                                 x_valid_display=x_valid_display, row_number=low_income_prediction)






    # make predictions for test data set

    #load test data
    test_data = pd.read_csv("data/income_test.csv")
    predictions_template = pd.read_csv("data/predictions_template.csv")

    #fill empty values
    test_data["ability to speak english"] = test_data["ability to speak english"].fillna(0.0)
    test_data["gave birth this year"] = test_data["gave birth this year"].fillna("No")

    #standardize and one hot encode
    test_numeric_scaled = pd.DataFrame(scaler.transform(test_data[numeric_columns]), columns=numeric_columns)
    test_categorical_encoded = pd.DataFrame(encoder.transform(test_data[categorical_columns]), columns=encoded_column_names)
    test_processed = pd.concat([test_numeric_scaled, test_categorical_encoded], axis=1)

    # get predictions from model, 0 = 'low', 1 = 'high'
    test_predictions_numeric = final_model.predict(test_processed)

    # get model likelihood for 'high' prediction (1 - 'high' chance = 'low' chance)
    test_probabilities_high = final_model.predict_proba(test_processed)[:, 1]

    # convert numeric label to actual label
    test_predictions_labels = pd.Series(test_predictions_numeric).map({0: "low", 1: "high"})

    # add predictions to template
    predictions_template["income"] = test_predictions_labels

    #save predictions to file
    predictions_template.to_csv("data/predictions_template.csv", index=False)

    print(predictions_template["income"].value_counts())


    # group prediction vs real income based on sex for validation data
    validation_fairness = x_valid_raw[["sex"]].copy()
    validation_fairness["true_income"] = y_valid.map({0: "low", 1: "high"}).values
    validation_predictions_final = final_model.predict(x_valid_processed)
    validation_fairness["predicted_income"] = pd.Series(validation_predictions_final).map({0: "low", 1: "high"}).values

    for sex_value, group in validation_fairness.groupby("sex"):
        # calculate overall accuracy per group
        group_accuracy = (group["true_income"] == group["predicted_income"]).mean()

        # calculate accuracy for predicting high and low separately
        predicted_high_rate = (group["predicted_income"] == "high").mean()
        true_high_rate = (group["true_income"] == "high").mean()

        # calculate recall for both high and low income
        high_income_people = group[group["true_income"] == "high"]
        low_income_people = group[group["true_income"] == "low"]

        high_recall = ((high_income_people["predicted_income"] == "high").mean() if len(high_income_people) > 0 else 0)
        low_recall = ((low_income_people["predicted_income"] == "low").mean() if len(low_income_people) > 0 else 0)

        print("\nSex:", sex_value)
        print("Accuracy:", round(group_accuracy, 3))
        print("True high-income rate:", round(true_high_rate, 3))
        print("Predicted high-income rate:", round(predicted_high_rate, 3))
        print("Recall for high income:", round(high_recall, 3))
        print("Recall for low income:", round(low_recall, 3))

if __name__ == "__main__":
    main()