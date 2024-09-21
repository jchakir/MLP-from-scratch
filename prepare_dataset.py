import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys



DATA_PATH: str = './data/data.csv'
TRAIN_DATA_PATH: str = './data/train-data.csv'
TEST_DATA_PATH: str = './data/test-data.csv'



def __hist_plot(data: pd.DataFrame) -> None:
  nrows, ncols = 8, 4
  _, ax = plt.subplots(nrows, ncols, figsize=(19, 25))
  true_data = data[data['1'] == 1]
  false_data = data[data['1'] == 0]
  for i, ii in enumerate(data.columns):
    if '1' == ii: continue
    row, col = i // ncols, i % ncols
    ax[row, col].hist(true_data[ii], bins=125, alpha=.5, label='1', color='yellow')
    ax[row, col].hist(false_data[ii], bins=25, alpha=.5, label='0', color='red')
    ax[row, col].legend()
    ax[row, col].set_title(ii)
  plt.tight_layout()
  plt.show()
# __hist_plot()


def __scatter_plot(data: pd.DataFrame) -> None:
  x_data = data.drop(columns=['1'])
  nrows, ncols = 11, 6
  _, ax = plt.subplots(nrows, ncols, figsize=(19, 25))
  counter: int = 0
  data_range = range(len(x_data))
  for i, ii in enumerate(x_data.columns[: -1]):
    for jj in x_data.columns[i + 1:]:
      row, col = counter // ncols, counter % ncols
      counter += 1
      ax[row, col].scatter(x_data[ii], data_range, alpha=0.5, color='yellow')
      ax[row, col].scatter(x_data[jj], data_range, alpha=0.5, color='red')
      ax[row, col].set_title(f'{ii}|{jj}')
  plt.tight_layout()
  plt.show()
# __scatter_plot()


def __pair_plot(data: pd.DataFrame) -> None:
  sns.pairplot(data, hue='1', kind='scatter', diag_kind='kde')
# __pair_plot()


def __load_dataset(path: str) -> pd.DataFrame:
  data = pd.read_csv(path, header=None, names=[f'{i}' for i in range(32)])
  data['1'] = data['1'].map({ 'B': 1, 'M': 0 })
  data = data.fillna(data.mean())
  return data


def __prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
  features_to_drop_according_to_hist = [
    '0', '3', '6', '7', '10', '11', '13', '15', '16', '17',
    '18', '19', '20', '21', '23', '26', '27', '30', '31'
  ]
  # features_to_drop_according_to_scatter_plot = []
  features_to_drop_according_to_pair_plot = ['4', '5']
  data = data.drop(columns=features_to_drop_according_to_hist)
  # data = data.drop(columns=features_to_drop_according_to_scatter_plot)
  data = data.drop(columns=features_to_drop_according_to_pair_plot)
  return data


def main() -> None:
  try:
    data = __load_dataset(DATA_PATH)
    train_size = int(data.shape[0] * 0.80)
    data = __prepare_dataset(data)
    train_data = data[ :train_size]
    test_data = data[train_size: ]
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

  except (FileNotFoundError, PermissionError):
    print(f'File Not Found or Permission Error')
  except Exception as error:
    print('Internal Server Error.')
    print(error)


if __name__ == '__main__':
  main()
