from keras.models import Model
from keras.layers import Input,LSTM, TimeDistributed, Embedding, Dense
import h5py
import numpy as np

if __name__ == "__main__":
    num_epochs = 50
    num_batch = 1
    max_doc_length = 10
    num_cells = 10
    num_samples = 1000
    num_time_steps = 10


    # load dataset and labels
    trainTextsSeq = np.load('train_x.npy')
    trainTextsSeq = trainTextsSeq.reshape((len(trainTextsSeq),max_doc_length,1))
    print("X.shape: " + str(trainTextsSeq.shape))

    y_train_tiled_old = np.load('train_y.npy')
    y_train_tiled = np.zeros((len(trainTextsSeq), max_doc_length, 2))
    for i in range(len(trainTextsSeq)):
        y_i = y_train_tiled_old[i]
        for j in range(len(y_i)):
            y_train_tiled[i][j][int(y_i[j])] = 1
    print("Y.shape: " + str(y_train_tiled.shape))


    ''' model creation '''
    # max_doc_length vectors of size embedding_size
    myInput = Input(shape=(max_doc_length,1), name='input')
    lstm_out = LSTM(num_cells, return_sequences=True)(myInput)
    predictions = TimeDistributed(Dense(2, activation='softmax'))(lstm_out)
    model = Model(inputs=myInput, outputs=predictions)
    print(model.outputs)
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit({'input': trainTextsSeq}, y_train_tiled, epochs=num_epochs, batch_size=num_batch)
    
    outputs_model = model.predict(trainTextsSeq, batch_size=num_batch)


    ''' model manipulation to get the internal states '''
    model.layers.pop();
    model.summary()
    # Save the states via predict
    inp = model.input
    out = model.layers[-1].output
    model_RetreiveStates = Model(inp, out)
    states_model = model_RetreiveStates.predict(trainTextsSeq, batch_size=num_batch)
    print(states_model.shape)


    ''' saving the model '''
    # Flatten first and second dimension for LSTMVis
    states_model_flatten = states_model.reshape(num_samples * num_time_steps, num_cells)
    outputs_model_flatten = outputs_model.reshape(num_samples * num_time_steps, 2)
    
    print(outputs_model[:10])
    print(outputs_model[:10].reshape((10*num_time_steps, 2)))

    print(states_model_flatten)
    hf = h5py.File("states.hdf5", "w")
    hf.create_dataset('states1', data=states_model_flatten)
    hf.create_dataset('outputs1', data=outputs_model_flatten)
    hf.close()
