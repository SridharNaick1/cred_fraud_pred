from cred_pred.data import load_data, validate_data
from cred_pred.config import load_config, PROJECT_ROOT


if __name__ == "__main__":
    # Load the training data
    config_path = PROJECT_ROOT/"configs"/"config.yaml"
    config = load_config(config_path)
    train_data = load_data(PROJECT_ROOT/ config["paths"]["train"])
    validated_train_data = validate_data(train_data, config_path, True)
    print("Training data loaded successfully.")
    print(validated_train_data.head())

