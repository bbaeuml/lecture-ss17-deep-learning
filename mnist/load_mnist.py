import cPickle, gzip, numpy

def load_mnist(file_name):
    # Load the dataset
    f = gzip.open(file_name, 'rb')
    data =  cPickle.load(f)
    f.close()
    return data



