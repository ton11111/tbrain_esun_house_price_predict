import multiprocessing as mp

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import KFold
import scipy

import config as conf
from dataloader import DatasetLoaderConfig, DatasetLoader
from models import xgboost_param, lgb_param, build_dnn_model
from utils import score_this_game, calc_mse, ReverseEngineer


class CVRunner:
    def __init__(self, train_csv_dsl, num_cv, fit_model_fn):
        self._train_csv_dsl = train_csv_dsl
        self._num_cv = num_cv
        self._fit_model_fn = fit_model_fn
        self._losses = []
        self._scores = []

        self._all_feature = train_csv_dsl.get_feature().values

        index_list = []
        for train_index, valid_index in KFold(n_splits=num_cv).split(self._all_feature):
            index_list.append((train_index, valid_index))
        self._index_list = index_list

        self._out_of_fold_prediction = np.zeros((self._all_feature.shape[0]))

    def run(self, n_jobs=1):
        with mp.Pool(n_jobs) as pool:
            results = pool.map(self._worker, range(self._num_cv))
            for score, loss, valid_index, pred in results:
                self._losses.append(loss)
                self._scores.append(score)
                self._out_of_fold_prediction[valid_index] = pred

    def _worker(self, i):
        train_index, valid_index = self._index_list[i]
        train_dsl, valid_dsl = self._train_csv_dsl.split_by_index(train_index, valid_index)

        X_train_cv = train_dsl.get_feature().values
        y_train_cv = train_dsl.get_label().values
        X_valid_cv = valid_dsl.get_feature().values
        y_valid_cv = valid_dsl.get_label().values

        model = self._fit_model_fn(i, X_train_cv, y_train_cv)

        loss = calc_mse(
            y_true=y_valid_cv,
            y_pred=model.predict(X_valid_cv),
        )
        score = score_this_game(
            y_true=valid_dsl.reverse_total_price(y_valid_cv),
            y_pred=valid_dsl.reverse_total_price(model.predict(X_valid_cv)),
        )

        print(score, loss)
        pred = model.predict(self._all_feature[valid_index])
        if len(pred.shape) == 2:
            assert pred.shape[1] == 1
            pred.shape = (pred.shape[0], )
        return score, loss, valid_index, pred

    def is_finished(self):
        return (len(self._losses) == self._num_cv
                and len(self._scores) == self._num_cv)

    def get_result(self):
        if not self.is_finished():
            print('not finished')
            return
        return np.mean(self._scores), np.mean(self._losses), self._out_of_fold_prediction


class GPUController:
    def __init__(self, avail_gpu_list=None):
        if not avail_gpu_list:
            avail_gpu_list = [0]
        self._avail_gpu_list = avail_gpu_list
        self._queue = mp.Queue()
        for i in self._avail_gpu_list:
            self._queue.put(i)

    def get_gpu_id(self):
        gpu_id = self._queue.get()
        return gpu_id

    def release_gpu_id(self, gpu_id):
        self._queue.put(gpu_id)

    def max_size(self):
        return len(self._avail_gpu_list)


pd.set_option('display.max_columns', 5000)

gpu_controller = GPUController(avail_gpu_list=conf.AVAIL_GPU_LIST)

# Load data
config = DatasetLoaderConfig()
train_csv_dsl = DatasetLoader(config=config)
train_csv_dsl.load(conf.TRAIN_DATASET)
X_train_csv = train_csv_dsl.get_feature().values
y_train_csv = train_csv_dsl.get_label().values

config.toggle_read_mode(True)
test_csv_dsl = DatasetLoader(config=config)
test_csv_dsl.load(conf.TEST_DATASET)
X_test = test_csv_dsl.get_feature().values

out_of_fold_predictions_dict = {
    'xgb': None,
    'lgb': None,
    'dnn': None,
}


# XGBoost K-fold validation
print('XGBoost K-fold validation')


def fit_xgb(i, X_train_cv, y_train_cv):
    gpu_id = gpu_controller.get_gpu_id()
    model = xgb.XGBRegressor(gpu_id=gpu_id, **xgboost_param)
    model.fit(X_train_cv, y_train_cv)
    gpu_controller.release_gpu_id(gpu_id)
    return model


runner = CVRunner(train_csv_dsl, conf.NUM_CV, fit_xgb)
runner.run(n_jobs=gpu_controller.max_size())

mean_score, mean_loss, out_of_fold_prediction = runner.get_result()
out_of_fold_predictions_dict['xgb'] = out_of_fold_prediction
print('[xgb] score: {}, loss: {}'.format(mean_score, mean_loss))
print(out_of_fold_predictions_dict['xgb'])


# lightgbm K-fold validation
print('lightGBM K-fold validation')


def fit_lgb(i, X_train_cv, y_train_cv):
    model = lgb.LGBMRegressor(**lgb_param)
    model.fit(X_train_cv, y_train_cv)
    return model


runner = CVRunner(train_csv_dsl, conf.NUM_CV, fit_lgb)
runner.run(n_jobs=5)

mean_score, mean_loss, out_of_fold_prediction = runner.get_result()
out_of_fold_predictions_dict['lgb'] = out_of_fold_prediction
print('[lgb] score: {}, loss: {}'.format(mean_score, mean_loss))
print(out_of_fold_predictions_dict['lgb'])


# DNN K-fold validation
print('DNN k-fold validation')
cols = train_csv_dsl.get_feature().columns


def dnn_lr_schd(epoch, lr):
    if epoch == 100 or epoch == 150:
        return lr * 0.1
    else:
        return lr


def fit_dnn(i, X_train_cv, y_train_cv):
    graph = tf.Graph()
    graph.as_default()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, graph=graph)
    tf.keras.backend.set_session(session)

    gpu_id = gpu_controller.get_gpu_id()
    with tf.device('/gpu:%d' % gpu_id):
        model = build_dnn_model(cols)
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(dnn_lr_schd)
        model.fit(
            X_train_cv,
            y_train_cv,
            epochs=250,
            batch_size=64,
            verbose=0,
            callbacks=[reduce_lr]
        )
    gpu_controller.release_gpu_id(gpu_id)
    return model


runner = CVRunner(train_csv_dsl, conf.NUM_CV, fit_dnn)
runner.run(n_jobs=gpu_controller.max_size())

mean_score, mean_loss, out_of_fold_prediction = runner.get_result()
out_of_fold_predictions_dict['dnn'] = out_of_fold_prediction
print('[dnn] score: {}, loss: {}'.format(mean_score, mean_loss))
print(out_of_fold_predictions_dict['dnn'])


# Get meta weights for ensembling the 3 models.
print('Getting meta weights.')
meta_feature = np.column_stack(
    [val for name, val in sorted(out_of_fold_predictions_dict.items())]
)


def fun(x):
    x = np.exp(x) / np.sum(np.exp(x))
    pred = np.sum(meta_feature * x, axis=-1)
    mse = calc_mse(y_train_csv, pred)
    return mse


result = scipy.optimize.basinhopping(fun, (0.0, 0.0, 0.0), niter=1000, disp=True)
meta_weight = np.exp(result.x) / np.sum(np.exp(result.x))
print(meta_weight)


# Generate initial guess by ensembling the outputs of the 3 models, each with 8 bags.


def worker(i_bag):
    gpu_id = gpu_controller.get_gpu_id()
    base_models = {
        'xgb': xgb.XGBRegressor(random_state=hash(i_bag), gpu_id=gpu_id, **xgboost_param),
        'lgb': lgb.LGBMRegressor(random_state=hash(i_bag), **lgb_param),
        'dnn': build_dnn_model(train_csv_dsl.get_feature().columns, seed=hash(i_bag)),
    }

    for name, model in base_models.items():
        if name == 'dnn':
            graph = tf.Graph()
            graph.as_default()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config, graph=graph)
            tf.keras.backend.set_session(session)

            with tf.device('/gpu:%d' % gpu_id):
                reduce_lr = tf.keras.callbacks.LearningRateScheduler(dnn_lr_schd)
                model.fit(
                    X_train_csv,
                    y_train_csv,
                    epochs=250,
                    batch_size=64,
                    verbose=0,
                    callbacks=[reduce_lr]
                )
        else:
            model.fit(X_train_csv, y_train_csv)
    gpu_controller.release_gpu_id(gpu_id)

    meta_feature_test = []
    for name, model in sorted(base_models.items()):
        if name == 'dnn':
            results = []
            results.append([val[0] for val in model.predict(X_test)])
            result = np.mean(results, axis=0)
            meta_feature_test.append(result)
        else:
            meta_feature_test.append(model.predict(X_test))
    meta_feature_test = np.column_stack(meta_feature_test)
    pred_orig_ln = np.sum(meta_feature_test * meta_weight, axis=-1)
    np.save('output_{}'.format(i_bag), pred_orig_ln)


n_jobs = gpu_controller.max_size()
with mp.Pool(n_jobs) as pool:
    pool.map(worker, range(conf.N_BAGS))

bags_pred_orig_ln = list()
for i in range(conf.N_BAGS):
    bags_pred_orig_ln.append(np.load('output_{}.npy'.format(i)))

final_pred = test_csv_dsl.reverse_total_price(np.mean(bags_pred_orig_ln, axis=0))

ans = pd.DataFrame()
ans['building_id'] = test_csv_dsl.get_dataset()['building_id']
ans['total_price'] = final_pred
original_ans = ans.copy()
print(ans)


# Post processing
print('Do post processing')


def find_close_hot_prices(x):
    if x < 10000000:
        grid_unit = 50000
        low = (int(x) // grid_unit) * grid_unit
        high = low + 50000
    elif x < 100000000:
        grid_unit = 500000
        low = (int(x) // grid_unit) * grid_unit
        high = low + grid_unit
    else:
        grid_unit = 5000000
        low = (int(x) // grid_unit) * grid_unit
        high = low + grid_unit
    return high, low


def find_hot_prices(low, high):
    results = []
    low -= 0.1
    while True:
        if len(results) == 0:
            val, _ = find_close_hot_prices(low)
        else:
            val, _ = find_close_hot_prices(results[-1] + 0.1)
        if val <= high:
            results.append(val)
        else:
            break
    return results


def calc_prob_pred_over_actual_hot(pred_orig, hot_orig):
    # from Bayes' theorem: P(actal hot | pred) * P(pred) = P(pred | actual hot) * P(actual hot)
    # we want to evaluate P(actal hot | pred) at one prediction result
    # P(pred) is a same value for all evaluated case

    pred_ln = np.log(pred_orig)
    hot_ln = np.log(hot_orig)

    # use normal distribution to discribe P(pred | actual hot)
    std = 0.12
    prob_pred_over_actual_hot = \
        np.exp(-(pred_ln - hot_ln) ** 2 / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
    # P(actual hot)
    prob_actual_hot = np.count_nonzero(orig_vals == np.round(hot_orig)) / len(orig_vals)
    return prob_pred_over_actual_hot * prob_actual_hot


price_converter = ReverseEngineer('total_price')
orig_vals = np.round(price_converter.transform(train_csv_dsl.reverse_total_price(y_train_csv)))

init_pred_confused = np.array(original_ans['total_price'])
init_pred_orig = price_converter.transform(init_pred_confused)
ans['total_price'] = init_pred_confused

for i in range(1):
    pred_confused = np.array(ans['total_price'])
    pred_orig = price_converter.transform(pred_confused)

    eps = 1e-8
    new_pred_confused = []
    n_altered = 0
    for i in range(len(pred_confused)):
        pred_confused_i = pred_confused[i]

        # +/- 10% coverage
        highbound_confused_i = pred_confused_i / 0.9
        lowbound_confused_i = pred_confused_i / 1.1

        highbound_orig_i = price_converter.transform(highbound_confused_i)
        lowbound_orig_i = price_converter.transform(lowbound_confused_i)

        higher_orig_i, _ = find_close_hot_prices(highbound_orig_i)
        _, lower_orig_i = find_close_hot_prices(lowbound_orig_i)

        higher_confused_ln_i = np.log(price_converter.reverse_transform(higher_orig_i))
        lower_confused_ln_i = np.log(price_converter.reverse_transform(lower_orig_i))

        highbound_confused_ln_i = np.log(highbound_confused_i)
        lowbound_confused_ln_i = np.log(lowbound_confused_i)

        higher_dist_confused_ln_i = higher_confused_ln_i - highbound_confused_ln_i
        lower_dist_confused_ln_i = lowbound_confused_ln_i - lower_confused_ln_i

        # when price go up
        higher_gain = calc_prob_pred_over_actual_hot(init_pred_orig[i], higher_orig_i)
        higher_new_lowbound_orig_i = \
            price_converter.transform(np.exp(lowbound_confused_ln_i + higher_dist_confused_ln_i))
        higher_loss = sum([
            calc_prob_pred_over_actual_hot(init_pred_orig[i], hot_orig)
            for hot_orig in find_hot_prices(lowbound_orig_i, higher_new_lowbound_orig_i)
        ])
        higher_net = higher_gain - higher_loss
        higher_feasible_i = (higher_loss < higher_gain)

        # when price go down
        lower_gain = calc_prob_pred_over_actual_hot(init_pred_orig[i], lower_orig_i)
        lower_new_highbound_orig_i = \
            price_converter.transform(np.exp(highbound_confused_ln_i - lower_dist_confused_ln_i))
        lower_loss = sum([
            calc_prob_pred_over_actual_hot(init_pred_orig[i], hot_orig)
            for hot_orig in find_hot_prices(lower_new_highbound_orig_i, highbound_orig_i)
        ])
        lower_net = lower_gain - lower_loss
        lower_feasible_i = (lower_loss < lower_gain)

        # adjust price
        new_pred_confused_ln_i = np.log(pred_confused_i)

        if higher_feasible_i and lower_feasible_i:
            if lower_net > higher_net:
                new_pred_confused_ln_i -= lower_dist_confused_ln_i + eps
            else:
                new_pred_confused_ln_i += higher_dist_confused_ln_i + eps
        elif lower_feasible_i:
            new_pred_confused_ln_i -= lower_dist_confused_ln_i + eps
        elif higher_feasible_i:
            new_pred_confused_ln_i += higher_dist_confused_ln_i + eps
        else:
            new_pred_confused_ln_i = None

        if new_pred_confused_ln_i is not None:
            n_altered += 1
            new_pred_confused_i = np.exp(new_pred_confused_ln_i)
            new_pred_confused.append(new_pred_confused_i)

            if new_pred_confused_i > pred_confused_i:
                assert price_converter.transform(new_pred_confused_i / 0.9) > higher_orig_i
            else:
                assert price_converter.transform(new_pred_confused_i / 1.1) < lower_orig_i
        else:
            new_pred_confused.append(pred_confused_i)

    print('{} prices altered.'.format(n_altered))
    ans['total_price'] = np.array(new_pred_confused)

    if n_altered == 0:
        break

print(ans)


# Save the output csv
csv = ans.to_csv('data.csv', index=False)
print('The output file is saved as data.csv')
