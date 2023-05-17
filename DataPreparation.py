
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPrep:
    """
    Class to execute the data cleaning and preperations tasks on new data
    """
    def __init__(self):
        pass
    
    def _extract_date(self, df):
        """
        Extract year,month and day columns from date columns
        """
        data = deepcopy(df)
        for col in ['orderDate']:
            data[col] = pd.to_datetime(data[col], format='%Y-%m-%d')
            data[col +"Year"] = data[col].dt.year.astype('Int64', errors='ignore')
            data[col + "Month"] = data[col].dt.month.astype('Int64', errors='ignore')
            data[col + "Day"] = data[col].dt.day.astype('Int64', errors='ignore')
            data[col + "Weekday"] = data[col].dt.weekday.astype('Int64', errors='ignore')
        return data

    def _clean_mixed_data(self, df):
        """
        Clean all ID mixed letter and number
        """
        data = deepcopy(df)
        data['orderID'] = data['orderID'].str.replace('R', '')
        data['itemID'] = data['itemID'].str.replace('A', '')
        data['customerID'] = data['customerID'].str.replace('C', '')
        data['voucherID'] = data['voucherID'].replace('NONE', '0')
        data['voucherID'] = data['voucherID'].str.replace('V', '')

        data['orderID'] = pd.to_numeric(data['orderID'])
        data['itemID'] = pd.to_numeric(data['itemID'])
        data['customerID'] = pd.to_numeric(data['customerID'])
        data['voucherID'] = pd.to_numeric(data['voucherID'])
        return data


    def _get_frequency(self, df):
        """
        Get number of orders
        """
        data = deepcopy(df)
        data1 = data.groupby('customerID', as_index=False).agg({'itemID': ['count']})
        data1.columns = ['customerID', 'frequency']
        data = data.merge(data1, on='customerID', how='left')
        return data
    
    def _calculate_items_per_order(self,df):
        """
        Get sum the number of items in each order Group by 'orderID'
        """
        data = deepcopy(df)
        order_item_counts = data.groupby('orderID')['itemID'].count()
        data = pd.merge(data, order_item_counts, on='orderID', how='left')
        data.rename(columns={'itemID_x': 'itemID', 'itemID_y': 'itemsPerOrderID'}, inplace=True)
        return data
    
    def _calculate_price_per_order(self,df):
        """
        get the total price for each orderID 
        """
        data = deepcopy(df)
        order_total_price = data.groupby('orderID')['price'].sum().rename('pricePerOrder').reset_index()
        data = pd.merge(data, order_total_price, on='orderID', how='left')
        return data

    def _calculate_voucher_rates(self,df):
        """
        Calculate the Discount Percentage base on vocherAmount
        """
        data = deepcopy(df)
        order_prices = data.groupby('orderID')['price'].sum()
        voucher_amounts = data.groupby('orderID')['voucherAmount'].first()
        voucher_rates = (voucher_amounts / order_prices) * 100
        voucher_rates = voucher_rates.round(2)
        data['voucherRate'] = data.groupby('orderID')['orderID'].transform(lambda x: voucher_rates[x.iloc[0]])
        return data

    def _calculate_promotion_rate(self,df):
        """
        calculate the promotion rate based on each itemID
        """
        data = deepcopy(df)
        data['promotionRate'] = round(((data['recommendedPrice'] - data['price']) / data['recommendedPrice']) * 100, 2)
        data.drop(['recommendedPrice'], axis=1, inplace=True) 
        return data

    def _encode_categorical_columns(self,df):
        data = deepcopy(df)
        labelEncoder_columns = ['deviceCode', 'paymentCode']
        label_encoder = LabelEncoder()
        for column in labelEncoder_columns:
            data[column + '_encoded'] = label_encoder.fit_transform(data[column])
        data.drop(columns=labelEncoder_columns, inplace=True)
        return data
    
    def _get_dummy_columns(self,df):
        data = deepcopy(df)
        categorical_columns = ['colorCode','sizeCode','typeCode']
        encoded_columns = pd.DataFrame()
        for column in categorical_columns:
            dummy_encoded = pd.get_dummies(data[column], prefix=column, drop_first=True)
            encoded_columns = pd.concat([encoded_columns, dummy_encoded], axis=1)
        data = pd.concat([data, encoded_columns], axis=1)
        data.drop(columns=categorical_columns, inplace=True)
        return data
    
    def run(self, df):
        data_prep = (
            df
            .pipe(self._extract_date)
            .pipe(self._clean_mixed_data)
            .pipe(self._get_frequency)
            .pipe(self._calculate_items_per_order)
            .pipe(self._calculate_price_per_order)
            .pipe(self._calculate_voucher_rates)
            .pipe(self._calculate_promotion_rate)
            .pipe(self._encode_categorical_columns)
            .pipe(self._get_dummy_columns)
        )

        return data_prep