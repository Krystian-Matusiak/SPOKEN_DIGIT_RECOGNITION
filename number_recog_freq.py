try:
    # Wavfile libraries
    from scipy.io.wavfile import read
    import wavio as wv
    # ML libraries
    from sklearn.model_selection import train_test_split
    from librosa import get_duration
    from sklearn.metrics import f1_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import os
    import re
    # own libs
    from utils import *
    # from scipy.io import wavfile
    from scipy.fft import fft, ifft, fftfreq
except:
    print("Smth is wrong")



def plot_signal(X, Y, xlabel, ylabel, title):
    plt.figure()
    plt.plot(X, Y)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# ---------------------------------------------------------------------------------
# Read all recordings
records_dir = os.path.dirname(os.path.realpath('__file__')) + '/NUMBER_RECOG/recordings/'
GlobVar = GlobalVariables()

LABEL_NUMBER = 5
save_or_load_csv = 1
if save_or_load_csv == 1:
    for wavfile in os.listdir(records_dir):
        number = int(wavfile[0])
        number_vec = np.zeros(LABEL_NUMBER)
        number_vec[number] = 1
        name, sample_id = re.findall(r'_(.*)_(.*).wav', wavfile)[0]
        signal = np.array(read(records_dir + wavfile)[1], dtype=float)
        f, y_signal = read(records_dir + wavfile)


        if GlobVar.MAX_SAMPLES < signal.shape[0]:
            GlobVar.MAX_SAMPLES = signal.shape[0]
        if GlobVar.MAX_VALUE < np.max(signal):
            GlobVar.MAX_VALUE = np.max(signal)
        if GlobVar.MIN_SAMPLES > signal.shape[0]:
            GlobVar.MIN_SAMPLES = signal.shape[0]

        data = {
            "id": sample_id,
            "name": name,
            "duration": float(get_duration(filename=records_dir + wavfile)),
            "signal": [signal],
            "signal_shape": signal.shape,
            "number_vec": [number_vec.tolist()],
            "number": [number]
        }

        GlobVar.all_records.append(data)
        GlobVar.df_records = pd.concat(
            [GlobVar.df_records, pd.DataFrame(data)], ignore_index=True, axis=0)
        GlobVar.sorted_records[number].append(signal)

    for index, sig in enumerate(GlobVar.df_records["signal"]):
        zero_vec = np.zeros(GlobVar.MAX_SAMPLES)
        zero_vec[0:sig.shape[0]] = sig/GlobVar.MAX_VALUE
        GlobVar.df_records["signal"][index] = zero_vec.tolist()
        GlobVar.df_records["signal_shape"][index] = zero_vec.shape[0]
        GlobVar.all_records[index] = zero_vec.tolist()
        GlobVar.sorted_records[GlobVar.df_records["number"][index]] = zero_vec.tolist()

    # GlobVar.df_records.to_parquet('dataframe.parquet')
if save_or_load_csv == 2:
    GlobVar.df_records = pd.read_parquet('dataframe.parquet')

y_fft_list = []
x_fft_list = []
for signal in GlobVar.df_records["signal"]:
    T = 1.0/f
    N = len(signal)
    N_fft = N//2
    y_fft = fft(signal,2**14)
    y_fft = np.abs(y_fft)
    y_fft = y_fft[0:N_fft]
    y_fft = y_fft/y_fft.max()
    x_fft = fftfreq(N, T)[:N_fft]
    y_fft_list.append(y_fft)
    x_fft_list.append(x_fft)
    if False:    
        plot_signal(x_fft, np.abs(y_fft[:N_fft]), "Frequency [Hz]",
                    "Amplitude", f"Number {number} of person {name}")
        plt.show()

GlobVar.df_records.insert(loc=len(GlobVar.df_records.columns), column="y_fft", value=y_fft_list)
GlobVar.df_records.insert(loc=len(GlobVar.df_records.columns), column="x_fft", value=x_fft_list)

print(GlobVar.df_records)
df_train, df_test = train_test_split(GlobVar.df_records, train_size=0.75)


# ---------------------------------------------------------------------------------
# PCA

pca = PCA(n_components=3)

X_pca_fft = GlobVar.df_records["y_fft"].tolist()
pca.fit(X_pca_fft)

train_pca = pca.transform(X_pca_fft)

train_pca = pd.DataFrame({'First': train_pca[:, 0], 'Second': train_pca[:, 1], 'Third': train_pca[:, 2]})
train_pca = train_pca.join(GlobVar.df_records["number_vec"])
train_pca = train_pca.join(GlobVar.df_records["number"])
colors = {0:'blue', 1:'magenta', 2:'black', 3:'cyan', 4:'orange', 5:'green', 6:'red', 7:'purple', 8:'brown', 9:'pink'}

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.scatter3D(train_pca['First'], train_pca['Second'], train_pca['Third'], c=train_pca['number'].map(colors))

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.show()

# ---------------------------------------------------------------------------------
# Neural Network

df_train, df_test = train_test_split(train_pca, train_size=0.85)
activ = "sigmoid"
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3)),
    tf.keras.layers.Dense(25, activation=activ),
    # tf.keras.layers.Dense(4, activation=activ),
    tf.keras.layers.Dense(LABEL_NUMBER, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=["accuracy"])

X = df_train.iloc[:, [0,1,2]].values.tolist()
Y = df_train["number_vec"].tolist()

X_test = df_test.iloc[:, [0,1,2]].values.tolist()
Y_test = df_test["number_vec"].tolist()

save_or_load = 1
if save_or_load == 1:
    history = model.fit(x=X, y=Y, epochs=100,verbose=2, validation_data=(X_test,Y_test))
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy and loss for train and test')
    plt.ylabel('Value')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy','val_loss', 'val_accuracy'], loc='upper left')
    plt.grid()
    plt.show()
    # model.save("model.h")
if save_or_load == 2:
    # model = tf.keras.models.load_model('model.h')
    pass

print(model.evaluate(X_test, Y_test))

# prediction_exact_compare(df_test, model)
print_crosstab(X_test, Y_test, model)
