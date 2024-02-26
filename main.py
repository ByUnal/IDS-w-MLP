"""
Author: M.Cihat ÜNAL
"""

import numpy as np
import pandas as pd
import random
import time

from MLP2hidden import MLP
from MLP1hidden import MLP_oh
from functions import get_train_test_data, normalize, convert_numeric

rs = 1
random.seed(rs)
np.random.seed(rs)

if __name__ == "__main__":
    """# Load Data"""
    df, df_testing = get_train_test_data("./kdd_data/kddcup.names",
                                         "./kdd_data/training",
                                         "./kdd_data/testing")

    # Data Preprocessing
    # Remove useless features
    # According to the authors inspection of the data it turned out that the values
    # of six features (land, urgent, num_failed_logins,
    # num_shells, is_host_login, num_outbound_cmds) were
    # constantly zero over all data records.

    zero_columns = ["land", "urgent", "num_failed_logins", "num_shells", "is_host_login", "num_outbound_cmds"]

    # Remove columns with only 0s
    df = df.drop(columns=zero_columns)
    df_testing = df_testing.drop(columns=zero_columns)

    # Convert features into numerical form -> tcp=0, udp=1, icmp=2
    protocol_dict = {"tcp": 0, "udp": 1, "icmp": 2}
    df.protocol_type = df.protocol_type.apply(lambda x: protocol_dict[x])

    # Do the same for test sets
    df_testing.protocol_type = df_testing.protocol_type.apply(lambda x: protocol_dict[x])

    df = convert_numeric(df)
    df_testing = convert_numeric(df_testing)

    """ Normalization"""
    df = normalize(df)
    df_testing = normalize(df_testing)

    """ Data Split & Training
    ** normal : 0, satan: 1, neptune: 2}**
    
    * The amount of "Normal" label must be 5,922 in train, 300 in val, 3,608 in test sets.
    * The amount of "Satan" label must be 1,807 in train, 300 in val, 2,321 in test sets.
    * The amount of "Neptune" label must be 4,430 in train, 300 in val, 1,067 in test sets.
    """

    # --- TRAIN & VALIDATION ---
    # Create dataframe contains only specific labels
    df_normal = df[df['label'] == 0]
    df_satan = df[df['label'] == 1]
    df_neptun = df[df['label'] == 2]

    # "normal" train and validation set
    df_train_normal = df_normal.sample(n=5922, random_state=rs)
    df_val_normal = df_normal[~df_normal.index.isin(df_train_normal.index)].sample(n=300, random_state=rs)

    # "satan" train and validation set
    df_train_satan = df_satan.sample(n=1807, random_state=rs)
    df_val_satan = df_satan[~df_satan.index.isin(df_train_satan.index)].sample(n=300, random_state=rs)

    # "neptun" train and validation set
    df_train_neptun = df_neptun.sample(n=4430, random_state=rs)
    df_val_neptun = df_neptun[~df_neptun.index.isin(df_train_neptun.index)].sample(n=300, random_state=rs)

    df_train = pd.concat([df_train_normal, df_train_satan, df_train_neptun]
                         ).sample(frac=1, random_state=rs).reset_index(drop=True)
    df_val = pd.concat([df_val_normal, df_val_satan, df_val_neptun]
                       ).sample(frac=1, random_state=rs).reset_index(drop=True)

    # --- TEST ---
    df_normal_test = df_testing[df_testing['label'] == 0].sample(n=3608, random_state=rs)
    df_satan_test = df_testing[df_testing['label'] == 1].sample(n=2321, random_state=rs)
    df_neptun_test = df_testing[df_testing['label'] == 2].sample(n=1067, random_state=rs)

    df_test = pd.concat([df_normal_test, df_satan_test, df_neptun_test]).sample(frac=1).reset_index(drop=True)

    print("\nTrain Counts:")
    print(df_train.label.value_counts())

    print("\nVal Counts:")
    print(df_val.label.value_counts())

    print("\nTest Counts:")
    print(df_test.label.value_counts())

    """# Train"""
    # X_train, X_val, and X_test should have shape (num_samples, input_size)
    # y_train, y_val should have shape (num_samples, output_size)
    X_train = df_train.drop(["label"], axis=1)
    y_train = df_train["label"]

    X_val = df_val.drop(["label"], axis=1)
    y_val = df_val["label"]

    X_test = df_test.drop(["label"], axis=1)
    y_test = df_test["label"]

    """ Train - 2 hidden"""
    print("\n******* 2-hidden Layer Experiment *******\n")

    input_size = 35
    hidden_size1 = 35
    hidden_size2 = 35
    output_size = 3  # multi-class
    epochs = 200
    learning_rate = 0.0001
    early_stop_patience = 3

    # Create MLP model with two hidden layers
    mlp_model = MLP(input_size, hidden_size1, hidden_size2, output_size)

    y_train_one_hot = np.eye(output_size)[y_train]
    y_val_one_hot = np.eye(output_size)[y_val]
    y_test_one_hot = np.eye(output_size)[y_test]

    start_time = time.time()
    # Train the model
    mlp_model.train(X_train, y_train_one_hot, X_val, y_val_one_hot, epochs, learning_rate, early_stop_patience)
    print("--- Training took % seconds ---\n" % (time.time() - start_time))

    start_time = time.time()
    # Test the model
    mlp_model.test(X_test, y_test_one_hot)
    print("--- Testing took %s seconds ---\n\n" % (time.time() - start_time))

    """## 2nd training
    Training with 1 hidden-layer*
    """
    print("******* 1-hidden Layer Experiment *******\n")

    rs = 6
    random.seed(rs)
    np.random.seed(rs)

    input_size = 35
    hidden_size1 = 45
    output_size = 3  # multi-class
    epochs = 200
    learning_rate = 0.0001
    early_stop_patience = 3

    # Create MLP model with two hidden layers
    one_hidden_model = MLP_oh(input_size, hidden_size1, output_size)

    y_train_one_hot = np.eye(output_size)[y_train]
    y_val_one_hot = np.eye(output_size)[y_val]
    y_test_one_hot = np.eye(output_size)[y_test]

    start_time = time.time()
    # Train the model
    one_hidden_model.train(X_train, y_train_one_hot, X_val, y_val_one_hot, epochs, learning_rate, early_stop_patience)
    print("--- Training took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    # Test the model
    one_hidden_model.test(X_test, y_test_one_hot)
    print("--- Testing took %s seconds ---" % (time.time() - start_time))
