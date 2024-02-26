import pandas as pd
from utils import category_int2str, category_str2int
from sklearn.preprocessing import LabelEncoder


def get_train_test_data(kddcup_path, train_path, test_path):
    def filter_data(df):
        df.drop_duplicates(keep="first", inplace=True)
        df.label = df.label.str.replace(".", "")
        df.label = df.label.apply(lambda x: category_str2int[x])
        df = df[~df.label.isin([3, 4])]
        return df

    # where feature_type is either 'continuous' or 'symbolic'
    schema_key = {}

    # a list of all possible string labels
    labels = []

    # maps symbolic features to possible values
    symbolic_features = {}

    with open(kddcup_path) as f:
        line_num = 0
        for line in f:
            line = line.replace('.', '')
            if line_num == 0:
                labels = [s.strip() for s in line.split(',')]
            else:
                [feature_name, feature_type] = line.split(': ')
                feature_name = feature_name.strip()
                feature_type = feature_type.strip()
                schema_key[line_num - 1] = (feature_name, feature_type)
                if feature_type == 'symbolic':
                    symbolic_features[feature_name] = []
            line_num += 1

    num_features = len(schema_key.keys())
    num_classes = len(category_int2str.keys())

    feature_list = [schema_key[i][0] for i in range(num_features)]
    feature_list.append("label")

    df_train = pd.read_csv(train_path, sep=',', header=None, names=feature_list)
    df_test = pd.read_csv(test_path, sep=',', header=None, names=feature_list)

    return filter_data(df_train), filter_data(df_test)


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        # Do not normalize the label column
        if feature_name != "label":
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def convert_numeric(df):
    enc_service = LabelEncoder()
    enc_flag = LabelEncoder()

    enc_service.fit(df['service'])
    enc_flag.fit(df['flag'])

    df['service'] = enc_service.transform(df['service'])
    df['flag'] = enc_flag.transform(df['flag'])

    return df
