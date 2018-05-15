import h5py
import numpy as np

# outputs input sequence and the label sequence
def squareWave(period, duration, length, offset = 0, label_offset=1):
    output = np.zeros(length + label_offset)
    i = 0
    while i < length + label_offset:
        if (i+offset) % period == 0:
            for j in range(duration):
                output[i + j] = 1
            i += duration - 1
        i += 1
    return output[:length], output[-1]

def squareWaveDataset(period, duration, length, offset=0, label_offset=1, size=10):
    X = np.zeros((size, length))
    Y = np.zeros(size)
    for i in range(size):
        example = squareWave(period, duration, length, offset=i, label_offset=label_offset)
        X[i] = example[0]
        Y[i] = example[1]
    return X, Y

if __name__ == "__main__":
    num_time_steps = 6

    execution_duration = 1
    task_period = 3
    trainTextsSeq, labels = squareWaveDataset(task_period, execution_duration, num_time_steps)

    # load a list
    trainTextsSeq_flatten = trainTextsSeq.flatten()
    print(trainTextsSeq_flatten)

    hf = h5py.File("train.hdf5", "w")
    hf.create_dataset('words', data=trainTextsSeq_flatten)
    hf.close()


    # Reshape y_train:
    y_train_tiled = np.tile(labels, (num_time_steps,1))
    y_train_tiled = np.transpose(y_train_tiled)
    y_train_tiled = y_train_tiled.reshape(len(labels), num_time_steps , 1)
    np.save("train_y", y_train_tiled)
    np.save("train_x", trainTextsSeq)
