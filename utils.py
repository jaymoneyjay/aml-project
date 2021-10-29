from sklearn import model_selection
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset

def results_to_csv(y_pred, file_name):
    """ Write the results to a csv file in the required format 
    
    Args:
        y_pred (numpy.array): predictions
        file_name (string): name of the csv file
    """
    
    df_res = pd.DataFrame({'id': np.arange(len(y_pred)), 'y': y_pred.ravel()})
    df_res.to_csv(f'data/{file_name}.csv', header=True, index=False)

def gen_dl(data, target=None, batch_size=32):
    """
    Generate DataLoader from numpy array

    Args:
        data (numpy.array): array with input data
        target (numpy.array): array with target data
        batch_size (int optional): size of batch

    Returns:
        dl (DataLoader)
    """
    
    # Cast to numpy array
    if type(data) == pd.DataFrame:
        data = data.values
    if type(target) == pd.DataFrame:
        target = target.values
    
    # Generate DataLoader
    if target is not None:
        ds = TensorDataset(torch.from_numpy(data).type(torch.float32), torch.tensor(target).type(torch.float32))
    else:
        ds = TensorDataset(torch.tensor(data).type(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    return dl

def load_df(base_dir, val_size):
    """
    Load pd.DataFrame from directory and split into train, validation and test set

    Data is expected to have the conform to the following:
        - reside in the csv files X_train.csv, X_test.csv, y_train.csv
        - have column "id"


    Args:
        base_dir: string. Path to the data directory.
        val_size: float. size of validation set

    Returns
        X_train:    pandas.DataFrame
        X_val:      pandas.DataFrame
        X_test:     pandas.DataFrame
        y_train:    pandas.DataFrame
        y_val:      pandas.DataFrame
    """

    # Load data from directory
    X_train = pd.read_csv(f'{base_dir}/X_train.csv')
    X_test = pd.read_csv(f'{base_dir}/X_test.csv')
    y_train = pd.read_csv(f'{base_dir}/y_train.csv')

    # Drop id column
    X_train = X_train.drop(columns='id')
    X_test = X_test.drop(columns='id')
    y_train = y_train.drop(columns='id')

    # Split train and validation set
    if val_size != 0:
        X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=val_size, shuffle=True)
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

    print(f'Loaded data successfully:')
    print(f' - X_train\t{X_train.shape}')
    if val_size != 0:
        print(f' - X_val\t{X_val.shape}')
    print(f' - X_test\t{X_test.shape}')
    print(f' - y_train\t{y_train.shape}')
    if val_size != 0:
        print(f' - y_val\t{y_val.shape}')
    if val_size != 0:
        return X_train, X_val, X_test, y_train, y_val

    return X_train, X_test, y_train

