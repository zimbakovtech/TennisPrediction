from sklearn.ensemble import RandomForestClassifier
import os

def process_y():
    dataset = []

    for i in range(15710):
        dataset.append(1)
        dataset.append(0)
    
    return dataset

def read_file():
    with open("data/processed/all_matches.csv", 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line.strip()]
    # Skip header and convert each line to a list of floats
    data = []
    for line in lines[1:]:
        values = line.split(',')
        float_values = [float(v) for v in values]
        data.append(float_values)
    return data


if __name__ == "__main__":
    dataset_y = process_y()
    dataset_x = read_file()

    

    threshold = int(len(dataset_x) * 0.8)

    train_x, test_x = dataset_x[:threshold], dataset_x[threshold:]
    train_y, test_y = dataset_y[:threshold], dataset_y[threshold:]

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    random_forest.fit(train_x, train_y)
    accuracy = random_forest.score(test_x, test_y)

    print(f"Accuracy: {accuracy * 100:.2f}%")



