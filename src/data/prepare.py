# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.sparse import hstack, save_npz


def save_store(store, df, name, val=False):
    print("Storing matrix {}".format(name))
    if val:
        store[name+"_val"] = df
    else:
        store[name] = df


def load_store(store, name, val=False):
    if val:
        name = name+"_val"

    return store[name]


# ## 1st feature parsing
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe:

                `float64` type to `float32`
                `int64`   type to `int32`
    '''

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df


def downcast_all(df_list):
    for df in df_list:
        downcast_dtypes(df)


# ## Group by month/shop/item

# target is total of sales for each shop and item within a month.
# We have to build a dataframe with this shape for our model to work on.
def grp_trans(transactions):
    aggd = dict.fromkeys(transactions.drop(["date"], axis=1).columns, "first")
    aggd["item_cnt_day"] = "sum"
    aggd["item_price"] = "mean"

    grpd = transactions.drop(["date"], axis=1)\
                       .groupby(["date_block_num", "shop_id", "item_id"])\
                       .agg(aggd)
    grpd.rename({"item_cnt_day": "item_cnt"}, axis=1, inplace=True)
    grpd = grpd.drop(["date_block_num", "shop_id", "item_id"], axis=1)\
               .reset_index()
    return grpd


# Add implicit rows (with no transactions)
def add_implicit_rows(grpd):
    grpd.set_index(["date_block_num", "shop_id", "item_id"], inplace=True)
    l1 = np.arange(34)
    shops_idx = grpd.index.get_level_values("shop_id")
    items_idx = grpd.index.get_level_values("item_id")
    l2 = np.arange(shops_idx.min(), shops_idx.max()+1)
    l3 = np.arange(items_idx.min(), items_idx.max()+1)
    idx = pd.MultiIndex.from_product([l1, l2, l3],
                                     names=["date_block_num",
                                            "shop_id", "item_id"])
    grpd = grpd.reindex(idx)
    grpd["item_cnt"] = grpd["item_cnt"].fillna(0)
    return grpd


def make_base_df(transactions):

    base_df = grp_trans(transactions)
    print("Grouping done")

    base_df = add_implicit_rows(base_df)
    print("Implicit rows done")
    return base_df


# ## Validation split

# train/validation split so that validation set is built the same way as the
# test set downloaded from kaggle
def train_test_split(test, base_df, val=False):
    if val:
        # Train with the first 32 months
        val_train = base_df.loc[:32]

        # Keep only the shop/items that will be asked in test by kaggle
        idx = test.set_index(["shop_id", "item_id"]).index
        val_test = base_df.loc[33].reindex(idx).reset_index()
        y_test = val_test["item_cnt"]
        val_test = val_test[["shop_id", "item_id"]]
        return (val_train, val_test, y_test)
    else:
        return (base_df, downcast_dtypes(test), None)


# ## Features Engineering

# Pipelines:
# - 1 pipeline input:
#     - join cat
#     - mean encode
# - 1 pipeline train
#     - Add lag from previous month
# - 1 pipeline test
#     - Add lag from end of train
# - 1 pipeline output
#     - clip

# ### Date

def blocknum_to_year(block_num):
    return block_num // 12 + 13


def blocknum_to_month(block_num):
    return block_num % 12 + 1


class YearMonthAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X.reset_index(inplace=True)
        dbl = X["date_block_num"]
        X["month"] = blocknum_to_month(dbl)
        X["year"] = blocknum_to_year(dbl)
        return X.set_index(["date_block_num", "shop_id", "item_id"])


# Holiday
class HolidayCountAdder(BaseEstimator, TransformerMixin):
    holidays_cnt_mnth = {1: 6,
                         2: 1,
                         3: 2,
                         4: 1,
                         5: 2,
                         6: 2,
                         11: 1}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X["holidays_cnt"] = X["month"].map(HolidayCountAdder.holidays_cnt_mnth)
        return X.fillna(0)


class TrainLastSeenAdder(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        tot_by_item = X.item_cnt.groupby(["date_block_num", self.col]).sum()
        last_seen = pd.Series(np.nan, tot_by_item.index)
        seen = tot_by_item[0] != 0
        max_blck = X.index.get_level_values("date_block_num").max() + 1
        for blck in range(1, max_blck):
            last_seen[blck] = np.where(seen, blck-1, last_seen[blck-1])
            seen = tot_by_item[blck] != 0
        last_seen.fillna(-99, inplace=True)
        X = X.reset_index()\
             .set_index(['date_block_num', self.col])\
             .join(last_seen.to_frame(), on=['date_block_num', self.col])\
             .reset_index()
        return X.rename({0: "last_seen_"+self.col[:-3]}, axis=1)\
                .set_index(["date_block_num", "shop_id", "item_id"])


# ### Categorical

# Categorical features are already encoded as integers, we don't have to
# perform this step. Since we use a tree-based model (xgboost),
# we do not need one-hot encoding.

# Add item_category_id:
class ItemCategoryAdder(BaseEstimator, TransformerMixin):
    def __init__(self, items):
        self.items = items[["item_id", "item_category_id"]]\
            .set_index("item_id")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        return X.join(self.items, on="item_id")


# Mean encoding:
class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, n_folds=5):
        self.n_folds = n_folds
        self.cat_cols = cat_cols
        self.test_encs = {}

    def fit(self, X, y=None):
        if not self.test_encs:
            for col in self.cat_cols:
                self.test_encs[col] = X.groupby(col).item_cnt.mean()
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X.reset_index(inplace=True)
        for col in self.cat_cols:
            X[col+'_target_enc'] = np.nan
        if "item_cnt" in X.columns:
            folds = KFold(self.n_folds)
            for others_idx, cur_idx in folds.split(X):
                cur = X.iloc[cur_idx]
                others = X.iloc[others_idx]
                for col in self.cat_cols:
                    col_target_mean = others.groupby(col).item_cnt.mean()
                    X.loc[cur_idx, col+'_target_enc'] = \
                        cur[col].map(col_target_mean)
            return X.set_index(["date_block_num", "shop_id", "item_id"])
        else:
            for col in self.cat_cols:
                X[col+'_target_enc'] = X[col].map(self.test_encs[col])
            return X


# subcategories
def prepare_subcats(item_categories):
    subcats = item_categories["item_category_name"].str\
                                                   .split(" - ", expand=True)
    item_categories = pd.concat([item_categories, subcats], axis=1)
    item_categories.rename({0: "subcat0", 1: "subcat1"}, axis=1, inplace=True)

    le0 = LabelEncoder()
    item_categories.subcat0 = le0.fit_transform(item_categories.subcat0
                                                .fillna(""))
    le1 = LabelEncoder()
    item_categories.subcat1 = le1.fit_transform(item_categories.subcat1
                                                .fillna(""))
    return item_categories


class ItemSubCategoryAdder(BaseEstimator, TransformerMixin):
    def __init__(self, item_categories):
        columns = ["item_category_id", "subcat0", "subcat1"]
        self.item_categories = \
            item_categories[columns].set_index("item_category_id")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        return X.join(self.item_categories, on="item_category_id")


class CategoriesOHEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.oh = OneHotEncoder()

    def fit(self, X, y=None):
        self.oh.fit(X[:, self.cat_cols])
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        new_vals = self.oh.transform(X[:, self.cat_cols])
        print(new_vals.shape)
        print(X.shape)
        X = hstack((X, new_vals), format="csr")
        return X


# ### Numerical
class NumericalStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols):
        self.num_cols = num_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.vals = X.values[:, self.num_cols]
        self.scaler.fit(self.vals)
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X = X.values
        new_vals = self.scaler.transform(X[:, self.num_cols])
        print(new_vals.shape)
        print(X.shape)
        X = np.concatenate((X, new_vals), axis=1)
        return X


# TODO: remove outliers for non tree based
class CntClipper(BaseEstimator, TransformerMixin):
    def __init__(self, col, min, max):
        self.col = col
        self.min = min
        self.max = max

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        if self.col in X.columns:
            X[self.col] = X[self.col].clip(self.min, self.max)
        return X


class LastSeenDiffer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X["last_seen_item_diff"] = \
            X.index.get_level_values("date_block_num") - X["last_seen_item"]
        X["last_seen_shop_diff"] = \
            X.index.get_level_values("date_block_num") - X["last_seen_shop"]
        return X


# ### Lag Feature

# In order to transform a temporal prediction into a classification problem,
# we simply tranform parameters for the previous month into features for the
# current month.

# There should be a different way of treating the lag feature between the train
# and test dataset, because this feature comes from the same dataset in train
# and from another dataset in test

# #### Lag for train
class TrainLagAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, lags_list):
        self.cols = cols
        self.lags_list = lags_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        nb_groups = X.tail(1).index.item()[0]+1
        shift_val = int(X.shape[0]/nb_groups)
        for lag in self.lags_list:
            for col in self.cols:
                X[col+"-{}".format(lag)] = X[col].shift(lag*shift_val)
        return X


# #### Lag for test

# Test file from kaggle only contains shop_id and item_id for the next month.
# We need to make it comply with the test set by manually filling features
# columns
class TestLastSeenAdder(BaseEstimator, TransformerMixin):
    def __init__(self, col, train):
        self.col = col
        self.new_col = "last_seen_"+self.col[:-3]
        self.prev_dbn = train.index.get_level_values("date_block_num").max()
        tr = train[train.index.get_level_values("date_block_num") ==
                   self.prev_dbn]
        self.tot_by_item = tr.groupby(col).agg({"item_cnt": "sum",
                                                self.new_col: "mean"})

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        seen = self.tot_by_item.item_cnt != 0
        last_seen = pd.Series(np.where(seen, self.prev_dbn,
                                       self.tot_by_item[self.new_col]),
                              self.tot_by_item.index)
        last_seen.fillna(-99, inplace=True)
        X = X.reset_index().set_index(self.col)\
                           .join(last_seen.to_frame(), on=self.col)\
                           .reset_index()
        return X.rename({0: self.new_col}, axis=1)\
                .set_index(["date_block_num", "shop_id", "item_id"])


class TestReindexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        return X.set_index(["shop_id", "item_id"])


class TestLagAdder(BaseEstimator, TransformerMixin):
    def __init__(self, train_df, cols, lags_list):
        self.train_df = train_df
        self.cols = cols
        self.lags_list = lags_list

    def fit(self, X, y=None):
        return self

    def join_test(self, X, off, col):
        date_blck = self.train_df.tail(1).index.item()[0]+1-off
        return X.join(self.train_df.loc[date_blck][col])[col]

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X.reset_index()
        for lag in self.lags_list:
            for col in self.cols:
                X[col+"-{}".format(lag)] = self.join_test(X, lag, col)
        return X


mean_enc_cols = ["item_id", "shop_id", "item_category_id", "year", "month",
                 "subcat0", "subcat1"]
lags_list = [1, 2, 3, 4, 5]
lag_cols = ["item_price", "item_cnt"]
# , "item_cnt_mean_month_item", "item_cnt_mean_month_shop"]


def do_pipelines(train, test, items, item_categories, shops, val=False):

    item_categories = prepare_subcats(item_categories)

    train_ppl_1 = Pipeline([
        ("last_seen_item_adder", TrainLastSeenAdder("item_id")),
        ("last_seen_shop_adder", TrainLastSeenAdder("shop_id")),
    ])

    glob_ppl_1 = Pipeline([
        ("category_adder", ItemCategoryAdder(items)),
        ("subcategory_adder", ItemSubCategoryAdder(item_categories)),
        ("year_month_adder", YearMonthAdder()),
        ("holiday_cnt_adder", HolidayCountAdder()),
        ("item_cnt_clipper", CntClipper("item_cnt", 0, 40)),
        ("last_seen_differ", LastSeenDiffer()),
        ("mean_encoder", MeanEncoder(mean_enc_cols)),
    ])

    train_ppl_2 = Pipeline([
        ("lag_adder", TrainLagAdder(lag_cols, lags_list)),
    ])

    train_ppl = Pipeline([
        ("spec_ppl_1", train_ppl_1),
        ("glob_ppl_1", glob_ppl_1),
        ("spec_ppl_2", train_ppl_2),
    ])

    train = train_ppl.fit_transform(train)
    print("train pipeline done")

    test_ppl_1 = Pipeline([
        ("last_seen_item_adder", TestLastSeenAdder("item_id", train)),
        ("last_seen_shop_adder", TestLastSeenAdder("shop_id", train)),
    ])

    test_ppl_2 = Pipeline([
        ("reindexer", TestReindexer()),
        ("lag_adder", TestLagAdder(train, lag_cols, lags_list)),
    ])

    test_ppl = Pipeline([
        ("spec_ppl_1", test_ppl_1),
        ("glob_ppl_1", glob_ppl_1),
        ("spec_ppl_2", test_ppl_2),
    ])

    if val:
        test["date_block_num"] = 33
    else:
        test["date_block_num"] = 34

    test = test_ppl.fit_transform(test)
    print("test pipeline done")

    return (train, test)


def X_y_split(train, test):
    train = train.loc[lags_list[-1]:].fillna(-99)

    y_train = train["item_cnt"].copy()
    X_train = train.reset_index().drop(["item_cnt", "item_price",
                                        "last_seen_item", "last_seen_shop"],
                                       axis=1)
    X_test = test.reset_index().drop(["last_seen_item", "last_seen_shop"],
                                     axis=1).fillna(-99)

    # print(X_test.head())

    X_test = X_test[X_train.columns]
    print("X/y train matrices done")
    return (X_train, y_train, X_test)


def load_files(input_path):
    transactions = pd.read_csv(os.path.join(input_path, 'sales_train.csv.gz'))
    items = pd.read_csv(os.path.join(input_path, 'items.csv'))
    item_categories = pd.read_csv(os.path.join(input_path,
                                               'item_categories.csv'))
    shops = pd.read_csv(os.path.join(input_path, 'shops.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))

    return (transactions, items, item_categories, shops, test)


def save_processed(output_path, X_train, y_train, X_test, y_test=None):
    print("saving data to output dir:")
    print("y_train")
    np.save(os.path.join(output_path, "y_train.npy"), y_train)

    if y_test is not None:
        print("y_test")
        np.save(os.path.join(output_path, "y_test.npy"), y_test)

    print("X_test")
    save_npz(os.path.join(output_path, "X_test.npz"), X_test)
    print("X_train")
    save_npz(os.path.join(output_path, "X_train.npz"), X_train)


def prepare_all(input_path, output_path, val=False, sample=False, store=None):

    transactions, items, item_categories, shops, test = load_files(input_path)
    if sample:
        transactions = transactions[transactions['shop_id'].isin([26, 27, 28])]

    downcast_all([transactions, items, item_categories, shops, test])

    if store is not None:
        try:
            base_df = load_store(store, "base_df")
            print("base_df loaded from store")
        except KeyError:
            base_df = make_base_df(transactions)
            save_store(store, base_df, "base_df")

        try:
            train_raw = load_store(store, "train_raw", val)
            test_raw = load_store(store, "test_raw", val)
            y_test = load_store(store, "y_test", val)
            print("raw train/test loaded from store")
        except KeyError:
            train_raw, test_raw, y_test = train_test_split(test, base_df, val)
            print("raw train/test successfully built")
            save_store(store, train_raw, "train_raw", val)
            save_store(store, test_raw, "test_raw", val)
            save_store(store, y_test, "y_test", val)

        try:
            train = load_store(store, "train", val)
            test = load_store(store, "test", val)
            print("pipelined train/test recovered from store")
        except KeyError:
            train, test = do_pipelines(train_raw, test_raw, items,
                                       item_categories, shops, val)
            print("pipelined train/test successfully built")
            save_store(store, train, "train", val)
            save_store(store, test, "test", val)

    else:
        base_df = make_base_df(transactions)
        train_raw, test_raw, y_test = train_test_split(test, base_df, val)
        print("raw train/test successfully built")
        train, test = do_pipelines(train_raw, test_raw, items, item_categories,
                                   shops, val)
        print("pipelined train/test successfully built")

    X_train, y_train, X_test = X_y_split(train, test)

    cat_cols = ["item_category_id", "subcat0", "subcat1"]
    num_cols = list(set(X_train.columns.values) - set(cat_cols))

    np_ppl = Pipeline([
        ("standard scaler", NumericalStandardScaler(list(map(X_train.columns
                                                             .get_loc,
                                                             num_cols)))),
        ("oh encoder",
         OneHotEncoder(categorical_features=list(map(X_train.columns.get_loc,
                                                     cat_cols)),
                       dtype=np.float32)),
    ])

    X_train = np_ppl.fit_transform(X_train)
    X_test = np_ppl.transform(X_test)

    save_processed(output_path, X_train, y_train, X_test, y_test)
