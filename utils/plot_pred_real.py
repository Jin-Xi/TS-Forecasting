import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def plot_pred_and_real(pred_data, real_data, epoch, save_img=False):
    mse = mean_squared_error(pred_data, real_data)
    # r_square = r2_score(pred_data, real_data)
    mape = mean_absolute_percentage_error(pred_data, real_data)
    print("[testing] [MSE:{}] / [MAE:{}]".format(mse, mape))
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(real_data, "blue", label="real data")
    plt.plot(pred_data, "green", label='pred data')
    plt.title("[MSE:{}] / [MAPE:{}]".format(mse, mape))
    plt.legend()
    if save_img:
        plt.savefig('./animation/DeepAR_beta2/' + str(epoch) + '.jpg')
    plt.show()
    return mse
