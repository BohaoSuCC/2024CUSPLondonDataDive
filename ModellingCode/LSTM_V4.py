from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn as sk
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import package to count number of hours after specific time
from datetime import datetime

# Faerier, adjusts for modulation, 7, 31
# Plot residuals to Faerier plot

# Define file paths
combined_dataset = "/home/henry-cao/Desktop/KCL/Extracurriculars/CUSP/Project_Code/Datasets/no_na_dataset.csv"

# Load data
data = pd.read_csv(combined_dataset)

# Print col names
print(data.columns)

# Print number of rows
print("Number of rows before preprocessing: ", data.shape[0])

# Only keep unique rows
data = data.drop_duplicates()

# Drop na values from Measurement, SatMean, and FlowMean
data = data.dropna(subset=['Measurement', 'SatMean', 'FlowMean'])

# When SatMean is less than 0, set value to na
data.loc[data['SatMean'] < 0, 'SatMean'] = np.nan

# Covert time column to datetime
data['time_temp'] = pd.to_datetime(data['MeasurementDateGMT'])

# Find earliest timestamp
earliest_timestamp = data['time_temp'].min()

# Calculated time elapsed
data['time_elapsed'] = data['time_temp'] - earliest_timestamp

# Turn time_elapsed into float32
data['time_elapsed'] = data['time_elapsed'].dt.total_seconds().astype('float32')

# Drop MeasurementDateGMT and time columns
data = data.drop(columns=['time_temp'])

# Convert certain categorical variables to numerical encoded values
data['SiteType'] = data['SiteType'].astype('category').cat.codes

# Store unique values of SpeciesType
unique_species_values = data['SpeciesType'].unique()

# Convert SpeciesType to numerical encoded values, and link up unique_species_values w/ numerical values
data['SpeciesType'] = data['SpeciesType'].astype('category').cat.codes
unique_species_dict = {i: unique_species_values[i] for i in range(len(unique_species_values))}

# Turn time, which represents exact hour of the day, into numerical
data['Hour'] = data['MeasurementDateGMT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').hour).astype(int)

# Extract day, month, and year from MeasurmentDateGMT
# Day is in index 0-1, month is in index 5-7, year is in index 9-12
# Time is in format YYYY-MM-DD HH:MM
data['Day'] = data['MeasurementDateGMT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').day).astype(int)
data['Month'] = data['MeasurementDateGMT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').month).astype(int)
data['Year'] = data['MeasurementDateGMT'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').year).astype(int)

# Dropping irrelevant or redundant columns
data = data.drop(columns=['MeasurementDateGMT', 'LocalAuthorityName', 'SiteCode', 'LocalAuthorityCode', 'SiteName', 'DateOpened', 'DateClosed', 'ID', 'DateTime', 'date',  'Date', 'SatBand', 'DateTimeStr', 'Time'])

# Print number of rows
print("Number of rows after preprocessing: ", data.shape[0])

# Define input size annd output size based on data shape
num_features = data.shape[1] - 1

# Making sure dropped columns are gone
print(f"Columns remaining: {data.columns}")

# Make test and training sets, with Measurements as response variable and
# all other variables as predictors
X = data.drop(columns=['Measurement'])
y = data['Measurement']

# Converting datatypes to float32
X = X.astype('float32')
y = y.astype('float32')

# # Initialise MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))
# X_scaled = scaler.fit_transform(X)

# Separate by SpeciesType
unique_species = data['SpeciesType'].unique()
for species in unique_species:
    # Use unique_species_dict to print species
    species_string = unique_species_dict[species]
    print(f"Pollutant: {species_string}")

    # Filter data by species
    data_species = data[data['SpeciesType'] == species]

    # Make test and training sets, with Measurements as response variable
    X_species = data_species.drop(columns=['Measurement', 'SpeciesType'])
    y_species = data_species['Measurement']
    X_species = X_species.astype('float32')
    y_species = y_species.astype('float32')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_species)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_species, test_size=0.2, random_state=1)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # model = Sequential([
    #     LSTM(units=500, activation='relu', input_shape=(1, X_train.shape[2]), recurrent_initializer='glorot_uniform'),
    #     Dense(1)
    # ])

    model = Sequential([
        Bidirectional(LSTM(units=500, activation='relu', input_shape=(1, X_train.shape[2]), return_sequences = True)),
        Dropout(0.2),
        Bidirectional(LSTM(units=250, activation='relu')),
        Dropout(0.2),
        Dense(100, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    checkpoint_filepath = f"/home/henry-cao/Desktop/KCL/Extracurriculars/CUSP/Project_Code/CUSP_source/LSTM_Models/checkpoint_model_{species_string}.h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Fit model
    history = model.fit(X_train, y_train, epochs=200, batch_size=256, validation_split = 0.2, callbacks=[model_checkpoint_callback])

    # Load best model
    best_model = load_model(checkpoint_filepath)

    best_model.summary()

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Plot Residuals between real and predicted values
    y_residuals = y_test - y_pred.flatten()
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_residuals, color='blue', label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals for {species_string}')
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss for {species_string}')
    plt.legend()
    plt.show()

    # # Calculate and print accuracy of predictions
    MSE = sk.metrics.mean_squared_error(y_test, y_pred)
    print("Mean squared error for {species}: ", MSE)






# # Initialise StandardScaler
# scaler = StandardScaler()

# # Fit X to have mean 0 and variance 1
# X_scaled = scaler.fit_transform(X)

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# model = Sequential([
#     LSTM(units=500, activation='relu', input_shape=(1, X_train.shape[2]), recurrent_initializer='glorot_uniform'),
#     Dense(1)
# ])

# model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=0.5), loss='mean_squared_error')

# model.summary()

# checkpoint_filepath = '/home/henry-cao/Desktop/KCL/Extracurriculars/CUSP/Project_Code/CUSP_source/LSTM_Models/checkpoint_model.h5'
# model_checkpoint_callback = ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True
# )

# # Fit model
# history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split = 0.2, callbacks=[model_checkpoint_callback])

# # Load best model
# best_model = load_model(checkpoint_filepath)

# # Make predictions
# y_pred = model.predict(X_test)

# # Plot Residuals between real and predicted values
# y_residuals = y_test - y_pred.flatten()
# plt.figure(figsize=(10,6))
# plt.scatter(y_test, y_residuals, color='blue', label='Residuals')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Actual Values')
# plt.ylabel('Residuals')
# plt.title('Residuals')
# plt.legend()
# plt.show()

# # Plot accuracy
# plt.figure(figsize=(10,6))
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('Loss')
# plt.legend()
# plt.show()

# # Calculate and print accuracy of predictions
# MSE = sk.metrics.mean_squared_error(y_test, y_pred)
# print("Mean squared error: ", MSE)