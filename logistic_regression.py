# -*- coding:utf-8 -*-
'''
logistic regression
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def inference(sigma0, sigma1, sigma2, x1, x2):
    # h(x)= 1 / (1 + np.e ** (-(sigma0 + sigma1 * x1 + sigma2 * x2)))
    pred_h = 1 / (1 + np.exp(-(sigma0 + sigma1 * x1 + sigma2 * x2)))
    return pred_h

# cost function
def eval_loss(sigma0, sigma1, sigma2, x1_list, x2_list, gt_y_list):
    avg_loss = 0
    for i in range(len(x1_list)):
        # avg_loss += 0.5 * (sigma1 * x_list[i] + sigma0 - gt_y_list[i]) ** 2
        pred_h = inference(sigma0, sigma1, sigma2, x1_list[i], x2_list[i])
        avg_loss += -(gt_y_list[i] * np.log(pred_h) + (1 - gt_y_list[i]) * np.log(1 - pred_h))
    avg_loss /= len(x1_list)

    return avg_loss

# calculate gradient
# 单一样本梯度
def gradient(pred_y, gt_y, x1, x2):
    diff = pred_y - gt_y
    d_sigma0 = diff
    d_sigma1 = diff * x1
    d_sigma2 = diff * x2
    return d_sigma0, d_sigma1, d_sigma2

# 全部样本（batchsize）为sigma0, sigma1, sigma2带来的更新
def cal_step_gradient(batch_x1_list, batch_x2_list, batch_gt_y_list, sigma0, sigma1, sigma2, lr):
    avg_d_sigma0, avg_d_sigma1, avg_d_sigma2 = 0, 0, 0
    batch_size = len(batch_x1_list)
    for i in range(batch_size):
        pred_y = inference(sigma0, sigma1, sigma2, batch_x1_list[i], batch_x2_list[i])
        d_sigma0, d_sigma1, d_sigma2 = gradient(pred_y, batch_gt_y_list[i], batch_x1_list[i], batch_x2_list[i])
        avg_d_sigma0 += d_sigma0
        avg_d_sigma1 += d_sigma1
        avg_d_sigma2 += d_sigma2

    avg_d_sigma0 /= batch_size
    avg_d_sigma1 /= batch_size
    avg_d_sigma2 /= batch_size
    # print('avg_d_sigma0:{}'.format(avg_d_sigma0))
    # print('avg_d_sigma1:{}'.format(avg_d_sigma1))
    # print('avg_d_sigma2:{}'.format(avg_d_sigma2))
    sigma0 -= lr * avg_d_sigma0
    sigma1 -= lr * avg_d_sigma1
    sigma2 -= lr * avg_d_sigma2
    return sigma0, sigma1, sigma2

def draw_logistic_regression(i, sigma0, sigma1, sigma2, fig, ax):


    ax.scatter(positive['Exam_1'], positive['Exam_2'], s=20, c='k', marker='+', label='positive')
    ax.scatter(negitive['Exam_1'], negitive['Exam_2'], s=20, c='r', marker='o', label='negitive')
    ax.legend(loc=1)
    plt.xlabel('Exam 1 score', fontsize=15)
    plt.ylabel('Exam 2 score', fontsize=15)
    x_line = np.linspace(30, 100, 1000)
    y_line = - (sigma0 + sigma1 * x_line) / sigma2
    plt.title(str(i) + ' iterations', fontsize='xx-large')
    ax.plot(x_line, y_line, c='r')
    plt.show()
    # plt.pause(2)
    # ax.cla()


def train(x1_list, x2_list, gt_y_list, batch_size, lr, max_iter):
    print('begin training...')
    sigma0 = 0
    sigma1 = 0
    sigma2 = 0

    best_loss = ''
    best_sigma0 = 0
    best_sigma1 = 0
    best_sigma2 = 0
    # plt.ion()
    fig, ax = plt.subplots()
    for i in range(max_iter):
        batch_idxs = range(batch_size)  # 获取样本前batch_size个样本的所有
        batch_x1 = [x1_list[i] for i in batch_idxs]
        batch_x2 = [x2_list[i] for i in batch_idxs]
        batch_y = [gt_y_list[i] for i in batch_idxs]
        sigma0, sigma1, sigma2 = cal_step_gradient(batch_x1, batch_x2, batch_y, sigma0, sigma1, sigma2, lr)
        loss = eval_loss(sigma0, sigma1, sigma2, batch_x1, batch_x2, batch_y)
        if best_loss == '':
            best_loss = loss
            best_sigma0 = sigma0
            best_sigma1 = sigma1
            best_sigma2 = sigma2
        elif best_loss > loss:
            best_loss = loss
            best_sigma0 = sigma0
            best_sigma1 = sigma1
            best_sigma2 = sigma2
        # draw_logistic_regression(i, best_sigma0, best_sigma1, best_sigma2, fig, ax)
        # print("sigma0:{},sigma1:{},sigma2:{}".format(sigma0, sigma1, sigma2))
        # print("loss is {}".format(loss))
        # plt.pause(1)
    print('End of the training')
    draw_logistic_regression(i, best_sigma0, best_sigma1, best_sigma2, fig, ax)
    return best_sigma0, best_sigma1, best_sigma2, best_loss

def verify_result(best_sigma0, best_sigma1, best_sigma2, x1_list, x2_list, gt_y_list):
    print('begin verify...')
    success_nums = 0
    fail_nums = 0
    for i in range(len(x1_list)):
        time.sleep(2)
        pred_h = inference(best_sigma0, best_sigma1, best_sigma2, x1_list[i], x2_list[i])
        valid = abs(gt_y_list[i] - pred_h)
        # print(valid)
        if valid < 0.5:
            success_nums += 1
        else:
            fail_nums += 1
    print('success_nums:{}'.format(success_nums))
    print('fail_nums:{}'.format(fail_nums))
    print('predict success ratio:{}%'.format(success_nums/(success_nums+fail_nums)*100))



if __name__ == "__main__":
    data_path = 'ex2data1.txt'
    data = pd.read_csv(data_path, names=['Exam_1', 'Exam_2', 'Admission'])
    positive = data[data['Admission'] == 1]  # 得到Admission为1的样本
    negitive = data[data['Admission'] == 0]  # 得到Admission为0的样本

    plt.scatter(positive['Exam_1'], positive['Exam_2'], s=20, c='k', marker='+', label='positive')
    plt.scatter(negitive['Exam_1'], negitive['Exam_2'], s=20, c='r', marker='o', label='negitive')
    plt.legend(loc=1)
    plt.xlabel('Exam 1 score', fontsize=15)
    plt.ylabel('Exam 2 score', fontsize=15)
    plt.show()
    x1_list = data['Exam_1']
    x2_list = data['Exam_2']
    gt_y_list = data['Admission']
    best_sigma0, best_sigma1, best_sigma2, best_loss = train(x1_list, x2_list, gt_y_list, 100, 0.001, 100)
    print("the best loss is {}".format(best_loss))
    print("the best sigma0 is {}".format(best_sigma0))
    print("the best sigma1 is {}".format(best_sigma1))
    print("the best sigma2 is {}".format(best_sigma1))
    verify_result(best_sigma0, best_sigma1, best_sigma2, x1_list, x2_list, gt_y_list)