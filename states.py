import h5py

f = h5py.File('states.hdf5','r')
print(list(f.keys()))
for r in f['states1']:
    print(min(r))

f = h5py.File('ptrain.hdf5','r')
print(f['words'].value)
