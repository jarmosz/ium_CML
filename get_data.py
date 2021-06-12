import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('ruchi798/movies-on-netflix-prime-video-hulu-and-disney', path='.', unzip=True)

# odczyt danych
film_data = pd.read_csv('MoviesOnStreamingPlatforms_updated.csv')

# Czyszczenie wierszy z pustymi warościami.
film_data.dropna(inplace=True)

# Usunięcie zbędnych kolumn
film_data.drop(film_data.columns[[0, 1]], axis = 1)

# Normalizacja: Lowercase dla danych tekstowych, standaryzacja (0..1) dla wartości float, sortowanie danych w komórce. 

for col_name in ['Title', 'Directors', 'Genres', 'Country', 'Language']:
    film_data[col_name] = film_data[col_name].str.lower()

for col_name in ['Directors', 'Genres', 'Country', 'Language']:
    film_data[col_name] = film_data[col_name].str.split(',').map(lambda x: ','.join(sorted(x)))
        
scaler = preprocessing.MinMaxScaler()
film_data[['IMDb', 'Runtime']] = scaler.fit_transform(film_data[['IMDb', 'Runtime']])

# Podział zbioru na train, dev, test w proporcji 8:1:1
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

film_train, film_test = train_test_split(film_data, test_size=1 - train_ratio)

film_valid, film_test = train_test_split(film_test, test_size=test_ratio/(test_ratio + validation_ratio))

film_train.to_csv('MoviesOnStreamingPlatforms_updated.train')
film_test.to_csv('MoviesOnStreamingPlatforms_updated.test')
film_valid.to_csv('MoviesOnStreamingPlatforms_updated.dev')

# Statystki głównego zbioru i podzbiorów
for i, data_set in enumerate([film_data, film_train, film_valid, film_test]):
    if i == 0:
        print("Główny zbiór danych")
    elif i == 1:
        print("Zbiór trenujący")
    elif i == 2:
        print("Zbiór walidujący")
    if i == 3:
        print("Zbiór testowy")
    print(len(data_set))
    print(data_set.describe().loc[['count','mean', 'max', 'min', 'std', '50%']])
    [print(data_set[name].value_counts()) for idx, name in enumerate(data_set)]

