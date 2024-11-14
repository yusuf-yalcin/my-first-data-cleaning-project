import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
m_data = pd.read_csv("eksikveriler.csv")

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

numeric_datas = m_data.iloc[:,1:4].values

imputer = imputer.fit(numeric_datas)
imputed_m_data = pd.DataFrame(imputer.transform(numeric_datas), columns=m_data.iloc[:,1:4].columns)

encoder_gender = preprocessing.LabelEncoder()
encoded_gender = encoder_gender.fit_transform(m_data.iloc[:,4:5].values)
encoded_gender = pd.DataFrame(data=encoded_gender, columns=["cinsiyet"])

imputed_m_data = imputed_m_data.join(encoded_gender)

"""
country_array = m_data.iloc[:,0:1].values
encoder = preprocessing.LabelEncoder()
country_array[:,0] = encoder.fit_transform(m_data.iloc[:,0])
                                                       
ohe = preprocessing.OneHotEncoder()
country_array = ohe.fit_transform(country_array).toarray()
"""

country_array = m_data.iloc[:,0:1].values

encoder_country = preprocessing.OneHotEncoder(sparse_output=False) #daha yoğun bir matris yapısı elde etmek için sparse_output=False parametresini girdik
country_array = encoder_country.fit_transform(country_array)

encoded_countries = pd.DataFrame(country_array, columns=encoder_country.categories_[0])

encoded_m_data = imputed_m_data.join(encoded_countries)

print(encoded_m_data)