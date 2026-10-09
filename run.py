from data_preprocess.load_to_tensor import load_data_to_tensors_and_transform

if __name__ == "__main__":
    
    res = load_data_to_tensors_and_transform()
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)

