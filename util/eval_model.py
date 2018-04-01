import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error


def eval_metrics(model, x_train, y_train, y_predict_train, x_test, y_test, y_predict, eval_dict):
    r2_train = model.score(x_train, y_train)
    r2_test = model.score(x_test, y_test)

    mse_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
    mse_test = np.sqrt(mean_squared_error(y_test, y_predict))

    eval_dict['r2_train'].append(r2_train)
    eval_dict['r2_test'].append(r2_test)
    eval_dict['mse_train'].append(mse_train)
    eval_dict['mse_test'].append(mse_test)

    return eval_dict


def plot_eval_metrics(param_range, eval_dict):
    with sns.axes_style('darkgrid'):
        plt.subplot(121)
        plt.tight_layout()
        plt.plot(param_range, eval_dict['r2_train'], color='blue')
        plt.plot(param_range, eval_dict['r2_test'], color='red')
        plt.title('R^2')

        plt.subplot(122)
        plt.tight_layout()
        plt.plot(param_range, eval_dict['mse_train'], color='blue')
        plt.plot(param_range, eval_dict['mse_test'], color='red')
        plt.title('MSE')
