def split_data(data, split_ratio = 0.8):
    df_split_index = int(split_ratio * data.shape[0])
    train = data[:df_split_index]
    test = data[df_split_index:]
    
    return train, test
        
    # raise NotImplementedError