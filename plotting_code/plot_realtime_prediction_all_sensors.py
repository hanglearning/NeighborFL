import argparse
import os
import pickle
from re import L
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import deque
import math

import pandas as pd
from tabulate import tabulate

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

# ''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg plot learning curves")

# arguments for system vars
parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the all_detector_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-pl', '--plot_last_comm_rounds', type=int, default=24, help='The number of the last comm rounds to plot. Will be a backup if starting_comm_round and ending_comm_round are not specified.')
parser.add_argument('-sr', '--starting_comm_round', type=int, default=2, help='round number to start plotting')
parser.add_argument('-er', '--ending_comm_round', type=int, default=None, help='round number to end plotting')
parser.add_argument('-tr', '--time_resolution', type=int, default=5, help='time resolution of the data, default to 5 mins')
parser.add_argument('-sd', '--single_plot_x_axis_density', type=int, default=1, help='label the 1 large plot x-axis by every this number of ticks')
parser.add_argument('-md', '--multi_plot_x_axis_density', type=int, default=4, help='label the multi sub plots x-axis by every this number of ticks')
parser.add_argument('-v', '--version', type=str, default='c', help='c for ccgrid for one step model, or j for journal extension for chained or multi-output models')
parser.add_argument('-r', '--representative', type=str, default=None, help='detector id to be the representative figure. By default, it is kind of random')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in subplots')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in subplots')
parser.add_argument('-attr', '--attribute', type=str, default='Speed', help='prediciting feature')


args = parser.parse_args()
args = args.__dict__

''' Variables Required '''
logs_dirpath = args["logs_dirpath"]
with open(f"{logs_dirpath}/check_point/args.pkl", 'rb') as f:
		args = pickle.load(f)
                
input_length = args["input_length"]
plot_last_comm_rounds = args["plot_last_comm_rounds"] # to plot last plot_last_comm_rounds hours
time_res = args["time_resolution"]
sing_x_density = args["single_plot_x_axis_density"]
mul_x_density = args["multi_plot_x_axis_density"]
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    sys.exit(f"Please specify the number of columns.")
''' Variables Required '''

plot_dir_path = f'{logs_dirpath}/plots/realtime_learning_curves_all_detectors'
os.makedirs(plot_dir_path, exist_ok=True)

def make_plot_data(detector_predicts):
    detector_lists = [detector_file.split('.')[0] for detector_file in detector_predicts.keys()]
    
    plot_data = {}
    for detector_file, models_attr in detector_predicts.items():
      detector_id = detector_file.split('.')[0]
      plot_data[detector_id] = {}

      for model, predicts in models_attr.items():

        if model == 'brute_force':
            continue

        plot_data[detector_id][model] = {}
        plot_data[detector_id][model]['x'] = []
        plot_data[detector_id][model]['y'] = []
        
        processed_rounds = set()
        for predict in predicts:
          round = predict[0]
          if round not in processed_rounds:
            # a simple hack to be backward compatible to the detector_predicts in main.py which may contain duplicate training round due to resuming, to be deleted in final version
            processed_rounds.add(round)
            data = predict[1]
            plot_data[detector_id][model]['x'].extend(range((round - 1) * input_length + 1, round * input_length + 1))
            plot_data[detector_id][model]['y'].extend(data)

    return detector_lists, plot_data
  
def plot_and_save_two_rows(detector_lists, plot_data):
    global COL # Not sure why have to do this here, and not for ROW
    """Plot
    Plot the true data and predicted data.
    """
    
    # draw 1 plot
    # for detector_id in detector_lists:
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.setp(ax, ylim=(0, 100))
    fig.text(0.04, 0.5, args['attribute'], va='center', rotation='vertical')
    ax.set_xlabel('Round Index')
    # detector_id = detector_lists[0]
    detector_id = args['representative']
    
    ax.set_title(detector_id)

    resume_comm_round = args["resume_comm_round"]

    if not args["starting_comm_round"]:
        s_round = int(resume_comm_round - plot_last_comm_rounds)
        e_round = resume_comm_round - 1
    else:
        s_round = args["starting_comm_round"]
        e_round = args["ending_comm_round"]
    
    # no prediction in first round
    start_plot_range = int(60/time_res* (s_round - 1))
    end_plot_range = int(60/time_res * (e_round)) 
        
    plotting_range = int(60/time_res*(e_round - s_round + 1))
    my_xticks = deque(range((sing_x_density - 1) * input_length, plotting_range, input_length * sing_x_density))
    # my_xticks.appendleft(0)
    
    ax.set_xticks(my_xticks)
    # xticklabels = list(range(args["resume_comm_round"] - 1 - plot_last_comm_rounds, args["resume_comm_round"], sing_x_density))
    xticklabels = list(range(s_round, e_round + 1, sing_x_density))
    # xticklabels[0] = 1
    ax.set_xticklabels(xticklabels, fontsize = 9, rotation = 45)
    
    to_plot = plot_data[detector_id]['true']['y'][start_plot_range:end_plot_range]
    ax.plot(range(len(to_plot)), to_plot, label='True Data', color='blue')
    true_curve = mlines.Line2D([], [], color='blue', label="TRUE")

    to_plot = plot_data[detector_id]['naive_fl']['y'][start_plot_range:end_plot_range]
    ax.plot(range(len(to_plot)), to_plot, label='naive_fl', color='lime')

    to_plot = plot_data[detector_id]['stand_alone']['y'][start_plot_range:end_plot_range]
    ax.plot(range(len(to_plot)), to_plot, label='stand_alone', color='orange')
    
    to_plot = plot_data[detector_id]['fav_neighbors_fl']['y'][start_plot_range:end_plot_range]
    ax.plot(range(len(to_plot)), to_plot, label='fav_neighbors_fl', color='red')
    
    # ax.plot(range(len(to_plot)), plot_data[detector_id]['brute_force']['y'][-plotting_range:], label='brute_force', color='violet')

    stand_alone_curve = mlines.Line2D([], [], color='orange', label="BASE")
    naive_fl_curve = mlines.Line2D([], [], color='lime', label="BFRT")
    neighbor_fl_curve = mlines.Line2D([], [], color='red', label="KFRT")
    brute_force_fl_curve = mlines.Line2D([], [], color='violet', label="brute_force")
    
    # ax.legend(handles=[true_curve,stand_alone_curve, naive_fl_curve, neighbor_fl_curve,brute_force_fl_curve], loc='best', prop={'size': 10})
    ax.legend(handles=[true_curve,stand_alone_curve, naive_fl_curve, neighbor_fl_curve], loc='best', prop={'size': 10})
    # ax.legend(handles=[true_curve,brute_force_fl_curve], loc='best', prop={'size': 10})
    fig.set_size_inches(8, 2)
    # plt.savefig(f'{plot_dir_path}/single_figure.png', bbox_inches='tight', dpi=500)
    # plt.show()

    
    # draw subplots
    # default - draw 2 row and 3 col = 6 plots
    # if COL == 1:
    #     fig, axs = plt.subplots(ROW, sharex=True, sharey=True)
    # else:
    if ROW == 1 and COL is None:
        # COL = len(detector_lists) - 1
        COL = len(detector_lists)
    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, 100))
    # axs[0].set_ylabel(args['attribute'])
    # fig.text(0.5, 0.04, 'Round Index', ha='center', size=13)
    fig.text(0.04, 0.5, args['attribute'], va='center', rotation='vertical', size=13)
    if ROW == 1 and COL == 1:
        axs.set_xlabel('Round Index', size=13)
    elif ROW == 1 and COL > 1:
        axs[COL//2].set_xlabel('Round Index', size=13)
    elif ROW > 1 and COL > 1:
        axs[ROW-1][COL//2].set_xlabel('Round Index', size=13)
    
    
    my_xticks = [0, plotting_range//2, plotting_range-2]
    # my_xticklabels = [args["resume_comm_round"] - 1 - plot_last_comm_rounds + 1, args["resume_comm_round"] - 1 - plot_last_comm_rounds//2 + 1, args["resume_comm_round"] - 1]
    my_xticklabels = [s_round, math.ceil((s_round + e_round)/2), e_round]
    
    
    # detector_lists.remove(args['representative'])
    for detector_plot_iter in range(len(detector_lists)):
        # for 6 plots
        # detector_plot_iter 0 ~ 2 -> row 0, col 0 1 2
        # detector_plot_iter 3 ~ 5 ->row 1, col 0 1 2
        # row = detector_plot_iter // 3
        # col = detector_plot_iter % 3
        row = detector_plot_iter // COL
        col = detector_plot_iter % COL

        if ROW == 1 and COL == 1:
          subplots = axs
        elif ROW == 1 and COL > 1:
          subplots = axs[detector_plot_iter]
        elif ROW > 1 and COL > 1:
          subplots = axs[row][col]
        
        detector_id = detector_lists[detector_plot_iter]
        # subplots.set_xlabel('Comm Round')
        subplots.set_title(detector_id)
        
        # plotting_range = int(60/time_res*plot_last_comm_rounds)
        
        subplots.set_xticks(my_xticks)
        subplots.set_xticklabels(my_xticklabels)
        
        subplots.plot(range(len(to_plot)), plot_data[detector_id]['true']['y'][start_plot_range:end_plot_range], label='True Data', color='blue')
        true_curve = mlines.Line2D([], [], color='blue', label="TRUE")
        
        subplots.plot(range(len(to_plot)), plot_data[detector_id]['naive_fl']['y'][start_plot_range:end_plot_range], label='naive_fl', color='lime')

        subplots.plot(range(len(to_plot)), plot_data[detector_id]['stand_alone']['y'][start_plot_range:end_plot_range], label='stand_alone', color='orange')
        
        subplots.plot(range(len(to_plot)), plot_data[detector_id]['fav_neighbors_fl']['y'][start_plot_range:end_plot_range], label='fav_neighbors_fl', color='red')
        
        # subplots.plot(range(len(to_plot)), plot_data[detector_id]['brute_force']['y'][start_plot_range:end_plot_range], label='brute_force', color='violet')
    
        stand_alone_curve = mlines.Line2D([], [], color='orange', label="BASE")
        naive_fl_curve = mlines.Line2D([], [], color='lime', label="BFRT")
        neighbor_fl_curve = mlines.Line2D([], [], color='red', label="KFRT")
        
        # brute_force_fl_curve = mlines.Line2D([], [], color='violet', label="brute_force")

        # subplots.legend(handles=[true_curve,stand_alone_curve, naive_fl_curve, neighbor_fl_curve, brute_force_fl_curve], loc='best', prop={'size': 10})

        subplots.legend(handles=[true_curve,stand_alone_curve, naive_fl_curve, neighbor_fl_curve], loc='best', prop={'size': 10})
        
        # subplots.legend(handles=[true_curve,brute_force_fl_curve], loc='best', prop={'size': 10})
        
            
        
        
    fig.set_size_inches(50, 6)
    plt.savefig(f'{plot_dir_path}/multi_figure.png', bbox_inches='tight', dpi=300)
    # plt.show()

    
def calculate_errors(plot_data):
    error_values = {}
    plotting_range = int(60/time_res*plot_last_comm_rounds)

    for detector_id, prediction_method in plot_data.items():
        error_values[detector_id] = {}
        for model, predicts in prediction_method.items():
            if model != 'true':
            # if model == 'brute_force':
                error_values[detector_id][model] = {}
                error_values[detector_id][model]['MAE'] = get_MAE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[detector_id][model]['MSE'] = get_MSE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[detector_id][model]['RMSE'] = get_RMSE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
                error_values[detector_id][model]['MAPE'] = get_MAPE(prediction_method['true']['y'][-plotting_range:], predicts['y'][-plotting_range:])
    with open(f'{plot_dir_path}/errors.txt', "w") as file:
        for detector_id, model in error_values.items():
            file.write(f'\nfor {detector_id}\n')
            error_values_df = pd.DataFrame.from_dict(model)
            file.write(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))
            file.write('\n')
    
with open(f"{logs_dirpath}/check_point/all_detector_predicts.pkl", 'rb') as f:
    detector_predicts = pickle.load(f)

detector_lists, plot_data = make_plot_data(detector_predicts)
# plot_and_save(detector_lists, plot_data)
plot_and_save_two_rows(detector_lists, plot_data)
calculate_errors(plot_data)