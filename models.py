import tensorflow as tf
import numpy as np

xgboost_param = dict(
    objective='reg:squarederror',
    max_depth=11,
    subsample=0.8005630073716024,
    n_estimators=16440,
    learning_rate=0.00647631488262977,
    gamma=0.9267584767684108,
    colsample_bytree=0.9988015932047529,
    reg_lambda=0.6553015656015893,
    min_child_weight=0.8572504467657722,
    reg_alpha=0.0,
    scale_pos_weight=1,
    tree_method='gpu_hist',
    n_jobs=5,
    verbosity=0,
)

lgb_param = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.012783478569491262,
    'num_leaves': 303,
    'max_depth': 23,
    'min_data_in_leaf': 19,
    'subsample': 0.8499384738625326,
    'colsample_bytree': 0.3838563405503983,
    'reg_alpha': np.exp(-1.6155277167195283),
    'reg_lambda': np.exp(-0.7142681482849973),
    'n_estimators': int(np.exp(9.82895782295262)),
    'n_jobs': 5,
}  # score: 5847.470714125195, loss: 0.02098778825839076


def build_dnn_model(cols, seed=None):
    if seed is not None:
        tf.random.set_random_seed(seed)

    def norm():
        return tf.keras.layers.BatchNormalization()

    enum_col_prefixes = [
        'building_material', 'city', 'town', 'village', 'building_type',
        'building_use', 'parking_way'
    ]
    conjugate_cols_list = []
    non_conjugate_cols = list(range(len(cols)))
    for prefix in enum_col_prefixes:
        conjugate_cols = []
        for idx, label in enumerate(cols):
            if label.startswith(prefix):
                conjugate_cols.append(idx)
                non_conjugate_cols.remove(idx)
        conjugate_cols_list.append(conjugate_cols)

    width = 4096
    depth = 2

    input = tf.keras.layers.Input([len(cols)])
    conjugate_embs = []
    for conjugate_cols in conjugate_cols_list:
        emb_unit = int(np.log(len(conjugate_cols)) / np.log(np.sqrt(2.0 * np.pi * np.e))) + 1
        conjugate_inputs = tf.keras.layers.Lambda(
            lambda x: tf.gather(x, conjugate_cols, axis=-1))(input)
        conjugate_emb = \
            tf.keras.layers.Dense(emb_unit, kernel_initializer='lecun_normal')(conjugate_inputs)
        conjugate_embs.append(conjugate_emb)
    non_conjugate_inputs = \
        tf.keras.layers.Lambda(lambda x: tf.gather(x, non_conjugate_cols, axis=-1))(input)
    input_emb = tf.keras.layers.Concatenate()(conjugate_embs + [non_conjugate_inputs])

    post_input = tf.keras.layers.Dense(width, kernel_initializer='lecun_normal')(input_emb)
    last_layer = post_input

    for layer_id in range(depth):
        bottleneck_width = width
        output_width = width

        x = norm()(last_layer)
        x = tf.keras.layers.Dense(bottleneck_width, kernel_initializer='orthogonal')(x)
        x = norm()(x)
        x = tf.keras.layers.Activation(tf.tanh)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(bottleneck_width, kernel_initializer='orthogonal')(x)
        x = norm()(x)
        x = tf.keras.layers.Activation(tf.tanh)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(output_width)(x)

        shortcut = tf.keras.layers.Dense(output_width, kernel_initializer='orthogonal')(last_layer)

        output = tf.keras.layers.Add()([x, shortcut])
        last_layer = output

    x = norm()(last_layer)
    x = tf.keras.layers.Dense(width, kernel_initializer='orthogonal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(x)
    y = tf.keras.layers.Lambda(lambda x: x * 0.1 + 15.5)(y)
    model = tf.keras.models.Model(inputs=input, outputs=y)

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
    )

    return model
