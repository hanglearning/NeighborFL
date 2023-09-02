import argparse
import os
import pickle
from re import L
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import deque
import math
import random

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
parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the all_device_predicts.pkl, e.g., /content/drive/MyDrive/NeighborFL/logs/09212021_142926_lstm')
parser.add_argument('-lp2', '--logs_dirpath2', type=str, default=None, help='for overwrting neighbor_fl predictions in -lp file due to different kick strategy')

parser.add_argument('-pr', '--plot_rounds', type=int, default=24, help='The number of comm rounds to plot. If starting_comm_round are not specified, plots the last these number of rounds.')
parser.add_argument('-sr', '--starting_comm_round', type=int, default=None, help='round number to start plotting')
# parser.add_argument('-er', '--ending_comm_round', type=int, default=None, help='round number to end plotting, inclusive')
parser.add_argument('-tr', '--time_resolution', type=int, default=5, help='time resolution of the data, default to 5 mins')
parser.add_argument('-nsx', '--num_sing_xticks', type=int, default=12, help='how many x-axes (xticks) on single device plot, usually should set to same as plot_rounds')
parser.add_argument('-nmx', '--num_mul_xticks', type=int, default=4, help='how many x-axes (xticks) on multiple devices plot')
parser.add_argument('-r', '--representative', type=str, default=None, help='device id to be the representative figure. If not speified, no single figure will be generated')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in subplots')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in subplots')
parser.add_argument('-f', '--feature', type=str, default='Speed', help='prediciting feature')
parser.add_argument('-Oseqs', '--output_sequences', type=str, default='1', help='For predictions having output_length > 1, specify the output sequences to plot. For instance, for a prediction having output_length as 5, e.g., [50.72 , 51.77 , 49.75, 66.02 , 66.92], speicying "24" will plot one for the 2nd prediction taking 51.77, and another plot for the 4th prediction taking 66.02. Default to 1.')


args = parser.parse_args()
args = args.__dict__

neighbor_fl_config = args["logs_dirpath2"].split("/")[-1] if args["logs_dirpath2"] else args["logs_dirpath"].split("/")[-1]

COLORS = {'central': 'orange', 'naive_fl': 'green', 'radius_naive_fl': 'grey', 'neighbor_fl': "red", 'true': 'blue'}
NAMES = {'central': 'Central', 'naive_fl': 'NaiveFL', 'radius_naive_fl': 'r-NaiveFL', 'neighbor_fl': f"NeighborFL {neighbor_fl_config}", 'true': 'TRUE'}

''' Variables Required - Below '''
logs_dirpath = args["logs_dirpath"]
with open(f"{logs_dirpath}/check_point/config_vars.pkl", 'rb') as f:
		config_vars = pickle.load(f)
                
with open(f"{logs_dirpath}/check_point/all_device_predicts.pkl", 'rb') as f:
    device_predicts = pickle.load(f)


try:
    with open(f"{args['logs_dirpath2']}/check_point/all_device_predicts.pkl", 'rb') as f:
        fav_predicts = pickle.load(f)
        for device in device_predicts:
           device_predicts[device]['neighbor_fl'] = fav_predicts[device]['neighbor_fl']
        logs_dirpath = args["logs_dirpath2"]
except:
    print("-lp2 for overwriting neighbor_fl in -lp1 not provided or not valid")
                
input_length = config_vars["input_length"]
plot_rounds = args["plot_rounds"] # to plot last plot_rounds hours
time_res = args["time_resolution"]
num_sing_xticks = args["num_sing_xticks"]
num_mul_xticks = args["num_mul_xticks"]

last_comm_round = device_predicts[list(device_predicts.keys())[0]]['true'][-1][0]
if not args["starting_comm_round"]:
    s_round = last_comm_round - plot_rounds + 1
    e_round = last_comm_round
else:
    s_round = args["starting_comm_round"]
    e_round = s_round + plot_rounds - 1

s_round = max(1, s_round)
    
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    COL = math.ceil(len(device_predicts) / ROW)
    if args["representative"]:
        COL = math.ceil((len(device_predicts) - 1) / ROW)

''' Variables Required - Above'''

plot_dir_path = f'{logs_dirpath}/plots/realtime_learning_curves_all_devices'
os.makedirs(plot_dir_path, exist_ok=True)

# get output length
OUTPUT_LENGTH = len(random.choice(list(device_predicts.values()))['true'][-1][-1][-1])

# get plotting output sequences
output_sequences = [int(x) for x in args["output_sequences"]]

def make_plot_data(device_predicts, skip_models, output_seq):
    device_lists = [device_file.split('.')[0] for device_file in device_predicts.keys()]
    
    plot_data = {}
    for device_file, models_feature in device_predicts.items():
      device_id = device_file.split('.')[0]
      plot_data[device_id] = {}

      for model, predicts in models_feature.items():

        # models to skip
        if model in skip_models or not predicts:
            continue

        plot_data[device_id][model] = []
        
        for i, predict in enumerate(predicts):
            round = predict[0]
            if model == 'true':
                # special dealing with ground truth
                if round > 1:
                    # drop earlist for ground truth
                    values = predict[1][OUTPUT_LENGTH - 1:]
                else:
                    values = predict[1]
                # extend O-1 more truth instances from the next record
                if i + 1 < len(predicts):
                    values = np.concatenate((values, predicts[i + 1][1][:OUTPUT_LENGTH - 1]))
                values = values[:,output_seq-1:output_seq].flatten()
            else:
                values = predict[1][:,output_seq-1:output_seq].flatten()
            
            if round in range(s_round, e_round + 1):
                plot_data[device_id][model].extend(values)

    return device_lists, plot_data
  
def plot_and_save_two_rows(device_lists, plot_data, output_seq):
    global COL # Not sure why have to do this here, and not for ROW
    """Plot
    Plot the true data and predicted data.
    """
    
    # draw 1 representative plot
    rep_sensor_id = args['representative']
    if rep_sensor_id:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        plt.setp(ax, ylim=(15, 75))
        fig.text(0.04, 0.5, args['feature'], va='center', rotation='vertical')
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
        plt.savefig(f'{plot_dir_path}/single_figure_{rep_sensor_id}_Oseq_{output_seq}.png', bbox_inches='tight', dpi=500)
        # plt.show()

    # draw subplots
    if ROW == 1 and COL is None:
        # COL = len(device_lists) - 1
        COL = len(device_lists)
    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, 75))
    # axs[0].set_ylabel(args['feature'])
    # fig.text(0.5, 0.04, 'Round Index', ha='center', size=13)
    fig.text(0.04, 0.5, args['feature'], va='center', rotation='vertical', size=13)
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
        device_lists.remove(rep_sensor_id)

    for device_plot_iter in range(len(device_lists)):
        row = device_plot_iter // COL
        col = device_plot_iter % COL

        if ROW == 1 and COL == 1:
          subplots = axs
        elif ROW == 1 and COL > 1:
          subplots = axs[device_plot_iter]
        elif ROW > 1 and COL > 1:
          subplots = axs[row][col]
        
        device_id = device_lists[device_plot_iter]
        # subplots.set_xlabel('Comm Round')
        # subplots.set_title(device_id, fontsize=30)
        subplots.set_title(device_id)
        
        
        subplots.set_xticks(my_xticks)
        subplots.set_xticklabels(my_xticklabels)

        # plot data
        legend_handles = []
        for model in plot_data[device_id]:
            if plot_data[device_id][model]:
                to_plot = plot_data[device_id][model]
                subplots.plot(range(len(to_plot)), to_plot, label=NAMES[model], color=COLORS[model])
                legend_handles.append(mlines.Line2D([], [], color=COLORS[model], label=NAMES[model]))
        
        # subplots.legend(handles=legend_handles, loc='best', prop={'size': 10})
    
    # fig.subplots_adjust(hspace=1)
    fig.set_size_inches(10, 10)
    plt.savefig(f'{plot_dir_path}/multi_figure_Oseq_{output_seq}.png', bbox_inches='tight', dpi=300)
    # plt.show()

def calculate_errors_and_output_table(plot_data, err_type, output_seq):
    # plotting_range = int(60/time_res*plot_rounds)
    device_id_to_model_errors = {}
    for device_id, prediction_method in plot_data.items():
        model_to_error = {}
        for model, predicts in prediction_method.items():
            if model != 'true' and predicts:
                len_truth = len(prediction_method['true']) # offset for the last O-1 predictions
                model_to_error[model] = globals()[f'get_{err_type}'](prediction_method['true'], predicts[:len_truth])
        device_id_to_model_errors[device_id] = model_to_error
    
    avg_model_error_by_device = device_id_to_model_errors
    
    columns = list(device_id_to_model_errors[device_id].keys())
    err_df = pd.DataFrame([device_id_to_model_errors[device_id] for device_id in plot_data], columns=columns, index=list(device_id_to_model_errors.keys()))
    err_df.rename(columns={old:new for old, new in zip(columns, [NAMES[n] for n in columns])}, inplace=True)

    # ChatGPT generated

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f'{plot_dir_path}/{err_type}_errors_rounds_{s_round}_{e_round}_Oseq_{output_seq}.xlsx', engine='xlsxwriter')

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

    return avg_model_error_by_device

def calc_average_prediction_error(avg_model_error_by_device, error_type, output_seq):
    model_to_errors_accum = {}
    for device_id, model_errors in avg_model_error_by_device.items():
        for model_name, error_value in model_errors.items():
            if model_name not in model_to_errors_accum:
                model_to_errors_accum[model_name] = []
            model_to_errors_accum[model_name].append(error_value)
    model_to_avg_err = {}
    for model, errors in model_to_errors_accum.items():
       model_to_avg_err[model] = round(np.average(errors, axis=0), 2)
       print(f"Oseq{output_seq}, Avg {NAMES[model]} {error_type}: {model_to_avg_err[model]}")
    return model_to_avg_err
    
for output_seq in output_sequences:
    print(f"For output sequence {output_seq}")
    if output_seq > OUTPUT_LENGTH:
        print(f"output_seq {output_seq} not available.")
        continue
    print(f"Output sequence {output_seq} out of {output_sequences}...")
    device_lists, plot_data = make_plot_data(device_predicts, [], output_seq)
    plot_and_save_two_rows(device_lists, plot_data, output_seq)
    avg_model_error_by_device = calculate_errors_and_output_table(plot_data, "MSE", output_seq)
    calc_average_prediction_error(avg_model_error_by_device, "MSE", output_seq)