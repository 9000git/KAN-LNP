# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:50:02 2024

@author: Juntao Wang, Qilie Liu
"""
from bayes_opt import BayesianOptimization
from kan import KAN, SYMBOLIC_LIB
from kan.utils import ex_round
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os
import sympy as sp
import shutil
import optuna
from functools import partial
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# 新增全局变量记录最佳Q²值
best_q2 = -float('inf')
current_best_file = None


def plot_and_save(y_train_true, y_train_pred, y_test_true, y_test_pred,
                  r2_train, mae_train, q2, r2_test, mae_test, trial_number):
    global current_best_file

    plt.figure(figsize=(8, 6), dpi=300)
    # 训练集：蓝色圆圈，测试集：红色圆圈
    plt.scatter(y_train_true, y_train_pred, c='#1f77b4', alpha=0.6,
                edgecolors='w', s=80, marker='o', label='Train')
    plt.scatter(y_test_true, y_test_pred, c='#ff0000', alpha=0.6,
                edgecolors='w', s=80, marker='o', label='Test')

    max_val = max(np.max(y_train_true), np.max(y_test_true)) * 1.1
    min_val = min(np.min(y_train_true), np.min(y_test_true)) * 0.9
    # 红色虚线y=x，线宽改为2
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('Actual Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(f'Best Model (Trial {trial_number})\nQ² = {q2:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    textstr = '\n'.join((
        f'Train:  R² = {r2_train:.4f}, MAE = {mae_train:.4f}',
        f'Test:   R² = {r2_test:.4f}, MAE = {mae_test:.4f}'))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=props, fontsize=10)

    plt.legend(loc='lower right')
    plt.tight_layout()

    # 创建figs文件夹并保存
    if not os.path.exists('figs'):
        os.makedirs('figs')
    filename = os.path.join('figs', f'best_model_trial_{trial_number}_q2_{q2:.4f}.png')
    plt.savefig(filename)
    plt.close()

    # 删除旧的最佳模型图片
    if current_best_file and os.path.exists(current_best_file):
        os.remove(current_best_file)
    current_best_file = filename


def kan_train_valid_test(x_train,
                         y_train,
                         x_test,
                         y_test,
                         hid_dim,
                         grid=5,
                         k=3,
                         seed=0,
                         opt='LBFGS',
                         steps_ori=30,
                         steps_prune=5,
                         lamb=0.001,
                         lamb_entropy=2.0,
                         lib=list(SYMBOLIC_LIB.keys())):
    
    try:
        device = 'cpu'
        if transform_features:
            qt = QuantileTransformer(output_distribution='uniform', random_state=42, n_quantiles=min(1000, len(x_train)))
            x_train_transform = qt.fit_transform(x_train)
            x_test_transform = qt.transform(x_test)
        else:
            x_train_transform = x_train
            x_test_transform = x_test
        if transform_label:
            qt_y = QuantileTransformer(output_distribution='uniform', random_state=42, n_quantiles=min(1000, len(y_train)))
            y_train_transform = qt_y.fit_transform(y_train)
            y_test_transform = qt_y.transform(y_test)
        else:
            y_train_transform = y_train
            y_test_transform = y_test

        # 保存变换后的数据集
        np.savetxt('train_transformed.txt',
                   np.hstack([y_train_transform, x_train_transform]),
                   fmt='%.6f',
                   header='y features')
        np.savetxt('test_transformed.txt',
                   np.hstack([y_test_transform, x_test_transform]),
                   fmt='%.6f',
                   header='y features')


        data = {}
        x_train_torch = torch.from_numpy(x_train_transform).to(device)
        x_test_torch = torch.from_numpy(x_test_transform).to(device)
        y_train_torch = torch.from_numpy(y_train_transform).to(device)
        y_test_torch = torch.from_numpy(y_test_transform).to(device)
        data['train_input'] = x_train_torch
        data['train_label'] = y_train_torch
        data['test_input'] = x_test_torch
        data['test_label'] = y_test_torch

        model = KAN(width=[x_train_torch.size(-1), hid_dim, 1], grid=grid, k=k, seed=seed, device=device)
        model.fit(data, opt=opt, steps=steps_ori, lamb=lamb, lamb_entropy=lamb_entropy)
        model = model.prune(node_th=3e-2)
        model.auto_symbolic(lib=lib, weight_simple=0.0)

        model_r2 = model.copy()
        model_r2.fit(data, opt='LBFGS', steps=steps_prune)
        formula = ex_round(model_r2.symbolic_formula()[0][0], 99)
        if transform_label:
            y_train_pred = model_r2(data['train_input']).detach().cpu().numpy()
            y_train_pred = qt_y.inverse_transform(y_train_pred)
        else:
            y_train_pred = model_r2(data['train_input']).detach().cpu().numpy()
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)

        with open('./kan.pickle', 'wb') as fs:
            pickle.dump(formula, fs)

        result_output = ''
        result_output += 'Formula: \n'
        modified_formula = denormalize_formula(formula)
        result_output += 'y' + ' = ' + modified_formula + '\n'
        result_output += 'R2_train: ' + '%.4f' % r2_train + '\n'
        result_output += 'MAE_train: ' + '%.4f' % mae_train + '\n'


        os.remove('./copy_temp_cache_data')
        os.remove('./copy_temp_config.yml')
        os.remove('./copy_temp_state')
        shutil.rmtree('./model')

        y_test_pred = np.empty([np.size(x_test, 0), 1])
        for i in range(np.size(data['test_input'], 0)):
            # 预测得到二维数组
            pred = kan_predict(data['test_input'][i:(i + 1), :])
            if transform_label:
                # 确保pred是二维结构后进行逆变换
                pred_inv = qt_y.inverse_transform(pred.reshape(1, -1))
                y_test_pred[i, :] = pred_inv.flatten()
            else:
                y_test_pred[i, :] = pred.flatten()

        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        result_output += 'R2_test: ' + '%.4f' % r2_test + '\n'
        result_output += 'MAE_test: ' + '%.4f' % mae_test + '\n'

        # os.remove('./kan.pickle')

        return result_output, r2_train, mae_train, r2_test, mae_test, y_train_pred, y_test_pred

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整堆栈信息
        try:
            os.remove('./copy_temp_cache_data')
            os.remove('./copy_temp_config.yml')
            os.remove('./copy_temp_state')
            shutil.rmtree('./model')
            # os.remove('./kan.pickle')
        except:
            pass
        return "", -1, -1, -1, -1, -1, None, None


def kan_predict(x):
    # 将PyTorch张量转换为numpy数组，并确保二维结构
    x_np = x.cpu().numpy().reshape(1, -1)  # 转换为形状(1, n_features)
    num_features = x_np.shape[1]  # 获取特征数量
    y_pred = np.zeros((1, 1))
    with open('./kan.pickle', 'rb') as fs:
        formula = pickle.load(fs)
    symbol_names = [str(symbol) for symbol in formula.free_symbols]
    subs = {}
    for j in range(num_features):  # 遍历每个特征
        symbol_name = f'x_{j + 1}'
        if symbol_name in symbol_names:
            subs[sp.Symbol(symbol_name)] = x_np[0, j]  # 使用numpy数组中的值
    y_pred[0, 0] = float(formula.evalf(subs=subs))
    return y_pred

def denormalize_formula(formula):
    modified_formula = str(formula)

    return modified_formula



if __name__ == '__main__':
    # 手动指定训练集和测试集文件路径
    train_file = './train.txt'  # 训练集文件路径
    test_file = './test.txt'  # 测试集文件路径
    transform_features = True
    transform_label = False

    # 加载训练集和测试集
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    x_train = train_data[:, 1:]
    y_train = train_data[:, :1]
    x_test = test_data[:, 1:]
    y_test = test_data[:, :1]

    hid_dim = 1
    grid = 5
    k = 3
    steps_ori = 20
    steps_prune = 15
    result_output, r2_train, mae_train, r2_test, mae_test, y_train_pred, y_test_pred = kan_train_valid_test(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        hid_dim=hid_dim, grid=grid, k=k,
        steps_ori=steps_ori, steps_prune=steps_prune)
    with open('./opt_kan.txt', 'a') as fs:
        fs.write(result_output)
