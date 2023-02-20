try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import seaborn as sb
except:
    print("Smth is wrong")

# ---------------------------------------------------------------------------------
# Container for variables
class GlobalVariables():
    
    def __init__(self):
        self.df_records = pd.DataFrame()
        self.all_records = []
        self.sorted_records = []
        for i in range(10):
            self.sorted_records.append([])
        
        self.MAX_VALUE = 0
        self.MAX_SAMPLES = 0
        self.MIN_SAMPLES = 10000

# ---------------------------------------------------------------------------------
# Recording
def record(time_dur=4, wavfile_name="recording.wav"):
    freq = 44100
    recording = sd.rec(int(time_dur * freq), samplerate=freq, channels=2)
    sd.wait()
    write(wavfile_name, freq, recording)


# ---------------------------------------------------------------------------------
# Plot record
def plot_record(GlobVar):
    number_of_record = 22
    signal = GlobVar.df_records["signal"][number_of_record]
    duration = GlobVar.df_records["duration"][number_of_record]
    time = np.arange(0, duration, duration/GlobVar.MAX_SAMPLES)
    plt.plot(time, signal)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(f"Sample Wav no. {number_of_record}")
    plt.show()

# ---------------------------------------------------------------------------------
# Print crosstab
def print_crosstab_terminal(X_test, Y_test, model):
    y_true = tf.argmax(Y_test, 1)
    y_pred = tf.argmax(model.predict(X_test), 1)
    results = pd.crosstab(index=y_true, columns=y_pred)
    print(results)

def print_crosstab_seaborn(X_test, Y_test, model):
    labels = Y_test
    preds = model.predict(X_test)
    y_preds = tf.argmax(preds, 1).numpy()

    exact_vec = []
    for y in Y_test:
        exact_vec.append(y)
    predict_vec = []
    for y in y_preds:
        predict_vec.append(y)

    data = {'Exact_values': exact_vec, "Predictions": predict_vec}
    df = pd.DataFrame(data=data)
    results = pd.crosstab(df['Exact_values'],df['Predictions'])
    plt.figure(figsize=(10,7))
    sb.heatmap(results, annot=True, cmap="OrRd", fmt=".0f")
    plt.title("Crosstab for tensorflow")
    # plt.show()

# ---------------------------------------------------------------------------------
# Print example predictions
def prediction_exact_compare(df_test, model, begin=0, end=10):
    val = model.predict(df_test["signal"].tolist()[begin:end])
    print("Predictions: [",end="")
    for vec in val:
        print(f"{np.argmax(vec)}, ",end="")
    print("]")
    print("Exact values: ", end="")
    print(df_test["number"].tolist()[begin:end])    
