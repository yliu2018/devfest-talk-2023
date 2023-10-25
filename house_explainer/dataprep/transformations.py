from typing import Tuple

import numpy as np
import pandas as pd


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def fill_missing(df, cols, val):
    """ Fill with the supplied val """
    for col in cols:
        df[col] = df[col].fillna(val)


def fill_missing_with_mode(df, cols):
    """ Fill with the mode """
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])


def addlogs(res, cols):
    """ Log transform feature list"""
    m = res.shape[1]
    for c in cols:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[c])).values)
        res.columns.values[m] = c + '_log'
        m += 1
    return res


def transformation(train_data: pd.DataFrame, isTrainingPrep=True) -> Tuple[pd.DataFrame, list[str]]:
    train_data['TotalSF'] = train_data['TotalBsmtSF'] + \
        train_data['1stFlrSF'] + train_data['2ndFlrSF']
    loglist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
               'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YearRemodAdd', 'TotalSF']

    train_data = addlogs(train_data, loglist)

    fill_missing(train_data, ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                              "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                              'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                              "MasVnrType", "MSSubClass"], "None")
    fill_missing(train_data, ["GarageYrBlt", "GarageArea", "GarageCars",
                              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                              'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                              "MasVnrArea"], 0)
    fill_missing_with_mode(
        train_data, ["MSZoning", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"])
    fill_missing(train_data, ["Functional"], "Typ")

    train_data.drop(['Utilities'], axis=1, inplace=True)
    train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    if isTrainingPrep:
        train_data.drop(train_data[(train_data['OverallQual'] < 5) & (train_data['SalePrice'] > 200000)].index,
                        inplace=True)
        train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index,
                        inplace=True)
        train_data.reset_index(drop=True, inplace=True)

    train_data['MSSubClass'] = train_data['MSSubClass'].astype('str')
    train_data['YrSold'] = train_data['YrSold'].astype(str)
    train_data['MoSold'] = train_data['MoSold'].astype(str)

    train_data, new_columns = one_hot_encoder(train_data)
    return (train_data, new_columns)
