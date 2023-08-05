import argparse
import os
import pickle
from re import L
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import deque
import math

import pandas as pd
import numpy as np
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
parser.add_argument('-lp2', '--logs_dirpath2', type=str, default=None, help='for overwrting fav_neighbor_fl predictions in -lp file due to different kick strategy')

parser.add_argument('-pr', '--plot_rounds', type=int, default=24, help='The number of comm rounds to plot. If starting_comm_round are not specified, plots the last these number of rounds.')
parser.add_argument('-sr', '--starting_comm_round', type=int, default=None, help='round number to start plotting')
# parser.add_argument('-er', '--ending_comm_round', type=int, default=None, help='round number to end plotting, inclusive')
parser.add_argument('-tr', '--time_resolution', type=int, default=5, help='time resolution of the data, default to 5 mins')
parser.add_argument('-nsx', '--num_sing_xticks', type=int, default=12, help='how many x-axes (xticks) on single detector plot, usually should set to same as plot_rounds')
parser.add_argument('-nmx', '--num_mul_xticks', type=int, default=4, help='how many x-axes (xticks) on multiple detectors plot')
parser.add_argument('-r', '--representative', type=str, default=None, help='detector id to be the representative figure. If not speified, no single figure will be generated')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in subplots')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in subplots')
parser.add_argument('-attr', '--attribute', type=str, default='Speed', help='prediciting feature')

args = parser.parse_args()
args = args.__dict__

neighbor_fl_config = args["logs_dirpath2"].split("/")[-1] if args["logs_dirpath2"] else args["logs_dirpath"].split("/")[-1]

COLORS = {'stand_alone': 'orange', 'naive_fl': 'green', 'radius_naive_fl': 'grey', 'fav_neighbors_fl': "red", 'true': 'blue'}
NAMES = {'stand_alone': 'Central', 'naive_fl': 'NaiveFL', 'radius_naive_fl': 'r-NaiveFL', 'fav_neighbors_fl': f"NeighborFL {neighbor_fl_config}", 'true': 'TRUE'}

''' Variables Required - Below '''
logs_dirpath = args["logs_dirpath"]
with open(f"{logs_dirpath}/check_point/config_vars.pkl", 'rb') as f:
		config_vars = pickle.load(f)
                
with open(f"{logs_dirpath}/check_point/all_detector_predicts.pkl", 'rb') as f:
    detector_predicts = pickle.load(f)


try:
    with open(f"{args['logs_dirpath2']}/check_point/all_detector_predicts.pkl", 'rb') as f:
        fav_predicts = pickle.load(f)
        for detector in detector_predicts:
           detector_predicts[detector]['fav_neighbors_fl'] = fav_predicts[detector]['fav_neighbors_fl']
        logs_dirpath = args["logs_dirpath2"]
except:
    print("-lp2 for overwriting fav_neighbor_fl in -lp1 not provided or not valid")
                
input_length = config_vars["input_length"]
plot_rounds = args["plot_rounds"] # to plot last plot_rounds hours
time_res = args["time_resolution"]
num_sing_xticks = args["num_sing_xticks"]
num_mul_xticks = args["num_mul_xticks"]

last_comm_round = detector_predicts[list(detector_predicts.keys())[0]]['true'][-1][0]
if not args["starting_comm_round"]:
    s_round = last_comm_round - plot_rounds + 1
    e_round = last_comm_round
else:
    s_round = args["starting_comm_round"]
    e_round = s_round + plot_rounds - 1
    
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    COL = math.ceil(len(detector_predicts) / ROW)
    if args["representative"]:
        COL = math.ceil((len(detector_predicts) - 1) / ROW)

''' Variables Required - Above'''

plot_dir_path = f'{logs_dirpath}/plots/realtime_learning_curves_all_detectors'
os.makedirs(plot_dir_path, exist_ok=True)

def make_plot_data(detector_predicts, skip_models):
    detector_lists = [detector_file.split('.')[0] for detector_file in detector_predicts.keys()]
    
    plot_data = {}
    for detector_file, models_attr in detector_predicts.items():
      detector_id = detector_file.split('.')[0]
      plot_data[detector_id] = {}

      for model, predicts in models_attr.items():

        # models to skip
        if model in skip_models or not predicts:
            continue

        plot_data[detector_id][model] = []
        
        for predict in predicts:
            round = predict[0]
            if round in range(s_round, e_round + 1):
                plot_data[detector_id][model].extend(predict[1])

    return detector_lists, plot_data
  
def plot_and_save_two_rows(detector_lists, plot_data):
    global COL # Not sure why have to do this here, and not for ROW
    """Plot
    Plot the true data and predicted data.
    """
    
    # draw 1 representative plot
    rep_sensor_id = args['representative']
    if rep_sensor_id:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        plt.setp(ax, ylim=(15, 75))
        fig.text(0.04, 0.5, args['attribute'], va='center', rotation='vertical')
        ax.set_xlabel('Round Index')
        
        ax.set_title(rep_sensor_id)
        
        plotting_range = int(60/time_res * plot_rounds)
        
        my_xticks = deque([i for i in range(0, plotting_range, num_sing_xticks)])   
        ax.set_xticks(my_xticks)
        
        xticklabels = list(range(s_round, s_round + plot_rounds))
        if xticklabels[-1] != s_round + plot_rounds - 1:
            xticklabels.append(s_round + plot_rounds - 1) # e_round
        ax.set_xticklabels(xticklabels, fontsize = 9)

        # plot data
        legend_handles = []
        for model in plot_data[rep_sensor_id]:
            if plot_data[rep_sensor_id][model]:
                to_plot = plot_data[rep_sensor_id][model]
                ax.plot(range(len(to_plot)), to_plot, label=NAMES[model], color=COLORS[model])
                legend_handles.append(mlines.Line2D([], [], color=COLORS[model], label=NAMES[model]))
        
        ax.legend(handles=legend_handles, loc='best', prop={'size': 10})
        
        fig.set_size_inches(10, 2) # width, height
        plt.savefig(f'{plot_dir_path}/single_figure_{rep_sensor_id}_.png', bbox_inches='tight', dpi=500)
        # plt.show()

    # draw subplots
    if ROW == 1 and COL is None:
        # COL = len(detector_lists) - 1
        COL = len(detector_lists)
    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, 75))
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
    # my_xticklabels = [config_vars["last_comm_round"] - 1 - plot_rounds + 1, config_vars["last_comm_round"] - 1 - plot_rounds//2 + 1, config_vars["last_comm_round"] - 1]
    my_xticklabels = [s_round, math.ceil((s_round + e_round)/2), e_round]
    
    if rep_sensor_id:
        detector_lists.remove(rep_sensor_id)

    for detector_plot_iter in range(len(detector_lists)):
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
        # subplots.set_title(detector_id, fontsize=30)
        subplots.set_title(detector_id)
        
        
        subplots.set_xticks(my_xticks)
        subplots.set_xticklabels(my_xticklabels)

        # plot data
        legend_handles = []
        for model in plot_data[detector_id]:
            if plot_data[detector_id][model]:
                to_plot = plot_data[detector_id][model]
                subplots.plot(range(len(to_plot)), to_plot, label=NAMES[model], color=COLORS[model])
                legend_handles.append(mlines.Line2D([], [], color=COLORS[model], label=NAMES[model]))
        
        # subplots.legend(handles=legend_handles, loc='best', prop={'size': 10})
    
    # fig.subplots_adjust(hspace=1)
    fig.set_size_inches(10, 10)
    plt.savefig(f'{plot_dir_path}/multi_figure.png', bbox_inches='tight', dpi=300)
    # plt.show()

def calculate_errors_and_output_table(plot_data, err_type):
    # plotting_range = int(60/time_res*plot_rounds)
    detetor_id_to_model_errors = {}
    for detector_id, prediction_method in plot_data.items():
        model_to_error = {}
        for model, predicts in prediction_method.items():
            if model != 'true' and predicts:
                model_to_error[model] = globals()[f'get_{err_type}'](prediction_method['true'], predicts)
        detetor_id_to_model_errors[detector_id] = model_to_error
    
    avg_model_error_by_detector = detetor_id_to_model_errors
    
    columns = list(detetor_id_to_model_errors[detector_id].keys())
    err_df = pd.DataFrame([detetor_id_to_model_errors[detector_id] for detector_id in plot_data], columns=columns, index=list(detetor_id_to_model_errors.keys()))
    err_df.rename(columns={old:new for old, new in zip(columns, [NAMES[n] for n in columns])}, inplace=True)

    # ChatGPT generated

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f'{plot_dir_path}/{err_type}_errors_rounds_{s_round}_{e_round}.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    err_df.to_excel(writer, sheet_name='Sheet1')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']


    # Apply conditional formatting to highlight the minimum value in each row
    format_rule = workbook.add_format({'bg_color': '#90EE90'})
    worksheet.conditional_format('B2:E27', {'type': 'formula', 'criteria': '=B2=MIN($B2:$E2)', 'format': format_rule})

    # Save the workbook
    writer.save()

    # err_df.to_csv(f'{plot_dir_path}/{err_type}_errors_rounds_{s_round}_{e_round}.csv')

    return avg_model_error_by_detector

def calc_average_prediction_error(avg_model_error_by_detector, error_type):
    model_to_errors_accum = {}
    for detector_id, model_errors in avg_model_error_by_detector.items():
        for model_name, error_value in model_errors.items():
            if model_name not in model_to_errors_accum:
                model_to_errors_accum[model_name] = []
            model_to_errors_accum[model_name].append(error_value)
    model_to_avg_err = {}
    for model, errors in model_to_errors_accum.items():
       model_to_avg_err[model] = round(np.average(errors, axis=0), 2)
       print(f"Avg {NAMES[model]} {error_type}: {model_to_avg_err[model]}")
    return model_to_avg_err
    

detector_lists, plot_data = make_plot_data(detector_predicts, ['offline_fl'])
plot_and_save_two_rows(detector_lists, plot_data)
avg_model_error_by_detector = calculate_errors_and_output_table(plot_data, "MSE")
calc_average_prediction_error(avg_model_error_by_detector, "MSE")