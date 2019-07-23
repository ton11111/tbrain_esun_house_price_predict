import numpy as np
import pandas as pd
from sklearn import preprocessing


class DatasetLoaderConfig(object):
    def __init__(
        self,
        transforms=[
            'reverse_engineer_total_price',
            'reverse_engineer_building_area',
            'norm_price_by_area',
            'zerofill_txn_floor',
            'zerofill_parking_area',
            'zerofill_parking_price',
            'village_income_median_missing_tag',
            'meanfill_village_income_median',
            # 'norm_rate', # CHENCHC
            'remove_town',
            'remove_village',
            'remove_town_population',
            'remove_town_area',
            'removeseries_n',
            'enumerate_to_onehot',
            'log_price',
            'log_distance',
            'log_area',
            'txn_floor_all_flag',
            'txn_floor_first_flag',
            'txn_floor_top_flag',
            'txn_floor_percentage',
            # 'neg_txn_floor',
            'removeseries_not_important',
            'sqrt_num_facility',
            # 'poly_XIII_10000',
            # 'poly_bachelor_rate',
            # 'poly_jobschool_rate',
            # 'poly_building_area',
            # 'poly_land_area',
            # 'poly_elementary_rate',
            # 'density',
            'have_parking',
            'standard_scalar',
        ],
        tape=None,
        read_mode=False
    ):
        self.transforms = transforms
        self.tape = tape if tape is not None else {}
        self.read_mode = read_mode

    def getTransforms(self):
        return self.transforms

    def setTapeRecord(self, transform, record):
        assert self.read_mode is False
        self.tape[transform] = record

    def getTapeRecord(self, transform):
        assert self.read_mode is True
        return self.tape[transform]

    def toggleReadMode(self, read_mode):
        self.read_mode = read_mode

    def isReadMode(self):
        return self.read_mode


class DatasetLoader(object):
    def __init__(
        self,
        config
    ):
        self.dataset = None
        self.config = config

    def load(self, filename):
        # Load the csv file
        self.dataset = pd.read_csv(filename)

        # Transform
        for transform in self.config.getTransforms():
            if transform == 'reverse_engineer_total_price':
                if 'total_price' not in self.dataset:
                    continue
                confused_total_price = self.dataset['total_price']
                true_total_price = ((confused_total_price - 196452) / 200) ** (2 / 3) * 10000
                self.dataset['total_price'] = true_total_price

            elif transform == 'reverse_engineer_building_area':
                confused_building_area = self.dataset['building_area']
                true_building_area = confused_building_area ** (1 / 1.31) / 11.61593
                self.dataset['building_area'] = true_building_area

            elif transform == 'norm_price_by_area':
                if 'total_price' not in self.dataset:
                    continue
                self.dataset['total_price'] = self.dataset['total_price'] / self.dataset['building_area']

            elif transform.startswith('zerofill_'):
                col = transform.replace('zerofill_', '')
                self.dataset[col].fillna(value=0.0, inplace=True)

            elif transform.startswith('remove_'):
                col = transform.replace('remove_', '')
                self.dataset.drop(labels=col, axis=1, inplace=True)

            elif transform.startswith('poly_'):
                col = transform.replace('poly_', '')
                self.dataset['{}-s2'.format(col)] = self.dataset[col] ** 2
                self.dataset['{}-s3'.format(col)] = self.dataset[col] ** 3
                self.dataset['{}-sq'.format(col)] = np.sqrt(self.dataset[col])

            elif transform == 'village_income_median_missing_tag':
                self.dataset['village_income_median_missing_tag'] = self.dataset['village_income_median'].isna()

            elif transform == 'meanfill_village_income_median':
                if self.config.isReadMode():
                    mean = self.config.getTapeRecord('meanfill_village_income_median')
                else:
                    mean = np.array(self.dataset['village_income_median'].dropna())
                    mean = np.exp(np.mean(np.log(mean)))
                    self.config.setTapeRecord('meanfill_village_income_median', mean)
                self.dataset['village_income_median'].fillna(value=mean, inplace=True)

            elif transform == 'norm_rate':
                cols = ['doc_rate', 'master_rate', 'bachelor_rate', 'jobschool_rate', 'highschool_rate',
                        'junior_rate', 'elementary_rate', 'born_rate', 'death_rate', 'marriage_rate',
                        'divorce_rate']
                if self.config.isReadMode():
                    mean_rate_dict = self.config.getTapeRecord('norm_rate')
                else:
                    mean_rate_dict = {}
                    for col in cols:
                        mean_rate = (
                            np.sum(self.dataset['town_population'] * self.dataset[col] * 0.01) /
                            np.sum(self.dataset['town_population'])
                        )
                        mean_rate_dict[col] = mean_rate
                    self.config.setTapeRecord('norm_rate', mean_rate_dict)

                for col in cols:
                    mean_rate = mean_rate_dict[col]
                    self.dataset[col] = (
                        (self.dataset[col] * 0.01 - mean_rate) /
                        np.sqrt(mean_rate * (1.0 - mean_rate) * self.dataset['town_population'])
                    )

            elif transform == 'removeseries_n':
                self.dataset.drop(
                    labels=['N_50', 'N_500', 'N_1000', 'N_5000', 'N_10000'], 
                    axis=1,
                    inplace=True
                )

            elif transform == 'enumerate_to_onehot':
                cols = [
                    'building_material', 'city', 'town', 'village', 'building_type', 
                    'building_use', 'parking_way'
                ]
                cols = [col for col in cols if col in self.dataset]

                if self.config.isReadMode():
                    possible_vals_dict = self.config.getTapeRecord('enumerate_to_onehot')
                else:
                    possible_vals_dict = {}
                    for col in cols:
                        possible_vals_dict[col] = list(self.dataset[col].unique())
                    self.config.setTapeRecord('enumerate_to_onehot', possible_vals_dict)

                for col in cols:
                    enum_val = self.dataset[col]
                    for possible_val in possible_vals_dict[col]:
                        new_col_name = '{}_{}'.format(col, possible_val)
                        self.dataset[new_col_name] = (enum_val == possible_val).astype(np.float32)
                    self.dataset.drop(col, axis=1, inplace=True)

            elif transform == 'log_price':
                cols = ['parking_price', 'total_price', 'village_income_median']
                for col in cols:
                    if col not in self.dataset.columns:
                        continue
                    self.dataset[col] = np.log(self.dataset[col] + 1.0)

            elif transform == 'log_distance':
                cols = [
                    'I_MIN', 'II_MIN', 'III_MIN', 'IV_MIN', 'V_MIN', 'VI_MIN', 'VII_MIN',
                    'VIII_MIN', 'IX_MIN', 'X_MIN', 'XI_MIN', 'XII_MIN', 'XIII_MIN', 'XIV_MIN'
                ]
                for col in cols:
                    self.dataset[col] = np.log(self.dataset[col] + 1.0)

            elif transform == 'log_area':
                cols = [
                    'land_area', 'building_area', 'parking_area'
                ]
                for col in cols:
                    if col == 'land_area':
                        self.dataset[col] = np.maximum(self.dataset[col], 2.222)
                    elif col == 'parking_area':
                        self.dataset[col] = np.maximum(self.dataset[col], 0.01)
                    self.dataset[col] = np.log(self.dataset[col])

            elif transform == 'txn_floor_all_flag':
                self.dataset['txn_floor_all_flag'] = (self.dataset['txn_floor'] == 0.0).astype(np.float32)

            elif transform == 'txn_floor_first_flag':
                self.dataset['txn_floor_first_flag'] = (self.dataset['txn_floor'] == 1).astype(np.float32)

            elif transform == 'txn_floor_top_flag':
                self.dataset['txn_floor_top_flag'] = (
                    self.dataset['txn_floor'] == self.dataset['total_floor']
                ).astype(np.float32)

            elif transform == 'txn_floor_percentage':
                self.dataset['txn_floor_percentage'] = (
                    self.dataset['txn_floor'] / self.dataset['total_floor']
                ).astype(np.float32)

            elif transform == 'neg_txn_floor':
                self.dataset['neg_txn_floor'] = self.dataset['total_floor'] - self.dataset['txn_floor']

            elif transform == 'standard_scalar':
                exclude_cols = ['total_price', 'building_id']
                if self.config.isReadMode():
                    record = self.config.getTapeRecord('standard_scalar')
                else:
                    record = {}
                    for col in self.dataset.columns:
                        if col in exclude_cols:
                            continue
                        scalar = preprocessing.StandardScaler().fit(self.dataset[[col]].astype(np.float64))
                        record[col] = scalar
                    self.config.setTapeRecord('standard_scalar', record)

                for col in self.dataset.columns:
                    if col in exclude_cols:
                        continue
                    self.dataset[col] = record[col].transform(self.dataset[[col]].astype(np.float64))

            elif transform == 'removeseries_not_important':
                remove_cols = [
                    'X_index_5000', 'X_index_10000', 'XI_index_10000',
                    'XIV_index_5000', 'XIV_index_10000', 'XIV_index_1000', 'XII_index_5000', 'XII_index_10000', 'V_index_5000',
                    'V_index_10000', 'VI_index_5000', 'VI_index_10000', 'VII_index_5000', 'VII_index_10000', 'VIII_index_5000',
                    'VIII_index_10000', 'I_index_5000', 'I_index_10000', 'IX_index_10000', 'IV_index_10000', 'II_index_5000',
                    'II_index_10000', 'III_index_5000', 'III_index_10000'
                ]
                self.dataset.drop(
                    labels=remove_cols,
                    axis=1,
                    inplace=True
                )

            elif transform == 'sqrt_num_facility':
                classe_list = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV']
                radius_list = [10, 50, 100, 250, 500, 1000, 5000, 10000]
                for cla in classe_list:
                    for r in radius_list:
                        col = '{}_{}'.format(cla, r)
                        self.dataset[col] = np.sqrt(self.dataset[col])

            elif transform == 'density':
                classe_list = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV']
                radius_list = [10000]
                for cla in classe_list:
                    for r in radius_list:
                        col = '{}_{}'.format(cla, r)
                        new_col = 'density_{}_{}'.format(cla, r)
                        self.dataset[new_col] = self.dataset[col] / (3.1415926 * (r**2))

            elif transform == 'have_parking':
                self.dataset['have_parking'] = (self.dataset['parking_area'] > 0.0).astype(np.float32)

            else:
                raise NotImplemented('{} is not a valid transform.'.format(transform))

    def getDataset(self, for_train=False):
        if not for_train:
            return self.dataset

        dataset = self.dataset.drop('building_id', axis=1).astype(np.float32)
        assert not np.isnan(np.array(dataset)).any()
        return dataset

    def getConfig(self):
        return self.config

    def split(self, val_percentage, rand=False, rand_seed=None):
        val_count = int(len(self.dataset) * val_percentage)
        assert val_count > 0
        train_count = len(self.dataset) - val_count
        assert train_count > 0

        if rand:
            val_list = np.random.choice(list(range(len(self.dataset))), val_count)
            train_list = [idx for idx in range(len(self.dataset)) if idx not in val_list]

            train_dataset = DatasetLoader(self.config)
            train_dataset.dataset = pd.DataFrame(self.dataset, index=train_list)

            val_dataset = DatasetLoader(self.config)
            val_dataset.dataset = pd.DataFrame(self.dataset, index=val_list)

        else:
            train_dataset = DatasetLoader(self.config)
            train_dataset.dataset = self.dataset[: train_count]
            val_dataset = DatasetLoader(self.config)
            val_dataset.dataset = self.dataset[train_count: ]

        return train_dataset, val_dataset

    def split_by_index(self, train_index, valid_index):
        train_dataset = DatasetLoader(self.config)
        train_dataset.dataset = pd.DataFrame(self.dataset, index=train_index)

        val_dataset = DatasetLoader(self.config)
        val_dataset.dataset = pd.DataFrame(self.dataset, index=valid_index)

        return train_dataset, val_dataset

    def getFeatureDataset(self):
        feature_dataset = self.dataset.drop('building_id', axis=1)
        if 'total_price' in feature_dataset.columns:
            feature_dataset.drop('total_price', axis=1, inplace=True)
        feature_dataset.astype(np.float32)
        feature_dataset = feature_dataset.reindex(sorted(feature_dataset.columns), axis=1)
        return feature_dataset

    def getLabelDataset(self):
        return self.dataset['total_price']

    def save(self, filename):
        self.dataset.to_csv(filename)

    def reverseTotalPrice(self, pred_total_price):
        confused_total_price = pred_total_price
        building_area = self.dataset['building_area']
        for transform in reversed(self.config.transforms):
            if transform == 'reverse_engineer_total_price':
                confused_total_price = 196452 + 200 * ((confused_total_price / 10000) ** (3 / 2))

            elif transform == 'norm_price_by_area':
                confused_total_price = confused_total_price * building_area

            elif transform == 'log_price':
                confused_total_price = np.exp(confused_total_price) - 1.0

            elif transform == 'standard_scalar':
                record = self.config.getTapeRecord('standard_scalar')
                building_area = record['building_area'].inverse_transform(building_area)

            elif transform == 'log_area':
                building_area = np.exp(building_area)

        return confused_total_price

    def __getitem__(self, key):
        dsl = DatasetLoader(self.config)
        dsl.dataset = self.dataset.loc[key, :]
        return dsl


if __name__ == '__main__':
    config = DatasetLoaderConfig()
    dsl = DatasetLoader(config=config)
