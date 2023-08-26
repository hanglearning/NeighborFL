import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import math
import pandas as pd

import sys
import inspect

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

from tabulate import tabulate

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="inferencing code")

parser.add_argument('-lp', '--logs_dirpath', type=str, default=None, help='the log path where resides the realtime_predicts.pkl, e.g., /content/drive/MyDrive/09212021_142926_lstm')
parser.add_argument('-lp2', '--logs_dirpath2', type=str, default=None, help='for overwrting neighbor_fl predictions in -lp file due to different kick strategy')
parser.add_argument('-et', '--error_type', type=str, default="MSE", help='error type to calculate, namely MAE, MSE, MAPE, RMSE.')
parser.add_argument('-ei', '--error_interval', type=int, default=24, help='unit is comm rounds, used in showing error table. will plot (-ei - 1) rounds')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in the plot')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in the plot')
parser.add_argument('-sr', '--start_round', type=int, default=1, help='provide the starting communication round, by default the 1st round of the simulation')
parser.add_argument('-er', '--end_round', type=int, default=None, help='provide the ending communication round, by default the last round of the simulation')
parser.add_argument('-r', '--representative', type=str, default=None, help='device id to be the representative figure. If not speified, no single figure will be generated')

args = parser.parse_args()
args = args.__dict__

# plot legends
neighbor_fl_config = args["logs_dirpath2"].split("/")[-1] if args["logs_dirpath2"] else args["logs_dirpath"].split("/")[-1]

COLORS = {'central': 'orange', 'naive_fl': 'green', 'radius_naive_fl': 'grey', 'neighbor_fl': "red", 'true': 'blue'}
NAMES = {'central': 'Central', 'naive_fl': 'NaiveFL', 'radius_naive_fl': 'r-NaiveFL', 'neighbor_fl': f"NeighborFL {neighbor_fl_config}", 'true': 'TRUE'}


''' load vars '''
logs_dirpath = args['logs_dirpath']
with open(f"{logs_dirpath}/check_point/config_vars.pkl", 'rb') as f:
    config_vars = pickle.load(f)
# all_device_files = config_vars["all_device_files"]
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

all_device_files = [device_file.split('.')[0] for device_file in device_predicts.keys()]
    
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    COL = math.ceil(len(device_predicts) / ROW)
    if args["representative"]:
        COL = math.ceil((len(device_predicts) - 1) / ROW)

''' load vars '''
start_round = args["start_round"]
end_round = args["end_round"]
if not end_round:
    end_round = config_vars["resume_comm_round"] - 1

plot_dir_path = f'{logs_dirpath}/plots/realtime_errors_interval'
os.makedirs(plot_dir_path, exist_ok=True)

def construct_realtime_error_table(realtime_predicts):
    realtime_error_table_normalized = {}
    for sensor_file, models_attr in realtime_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      realtime_error_table_normalized[sensor_id] = {}
      xticklabels = [[1]]
      for model, predicts in models_attr.items():
        if predicts and model != 'true' and model != 'offline_fl':
          realtime_error_table_normalized[sensor_id][model] = []

          data_list = []
          true_data_list = []
          first_round_skip = False
          for predict_iter in range(len(predicts)):
            predict = predicts[predict_iter] 
            # see plot_realtime_learning_curves.py, to be deleted in final version
            round = predict[0]
            if round > end_round:
                break
            data = predict[1]
            true_data = models_attr['true'][predict_iter][1]
            data_list.extend(data)
            true_data_list.extend(true_data)
            if (not first_round_skip and (round + 1) % args["error_interval"] == 0) or (first_round_skip and round != args["error_interval"] and round % args["error_interval"] == 0):
                # conclude the errors
                realtime_error_table_normalized[sensor_id][model].append(globals()[f"get_{args['error_type']}"](true_data_list, data_list))
                data_list = []
                true_data_list = []
                first_round_skip = True
                # deal with xticklabels
                if model == 'neighbor_fl':
                    # only need one iteration to form this list (by specifying one model)
                    xticklabels[-1].append(round)
                    xticklabels.append([round + 1])
          # if there's leftover
          if data_list and true_data_list:
            realtime_error_table_normalized[sensor_id][model].append(globals()[f"get_{args['error_type']}"](true_data_list, data_list))

    
    xticklabels[-1].append(end_round)
    if args["error_interval"] == 1:
       xticklabels = [[r] for r in range(1, len(xticklabels))]
    return realtime_error_table_normalized, xticklabels

def compare_l1_smaller_equal_percent(l1, l2):
  if len(l1) != len(l2):
    return "Error"
  l1_smaller_or_equal_values_count = 0
  for _ in range(len(l1)):
    if l1[_] <= l2[_]:
      l1_smaller_or_equal_values_count += 1
  percentage = l1_smaller_or_equal_values_count/len(l1)
  percent_string = f"{percentage:.2%}"
  return percentage, percent_string

def compare_l1_smallest_equal_percent(l1, l2, l3):
    if len(l1) != len(l2) or len(l1) != len(l3):
        return "Error"
    l1_smallest_or_equal_values_count = 0
    for _ in range(len(l1)):
        if l1[_] < l2[_] and l1[_] < l3[_]:
            l1_smallest_or_equal_values_count += 1
    percentage = l1_smallest_or_equal_values_count/len(l1)
    percent_string = f"{percentage:.2%}"
    return percentage, percent_string

def plot_realtime_errors(realtime_error_table, to_compare_model, COL, xticklabels, width, length, hspace):

    error_to_plot = args["error_type"]

    sensor_lists = [sensor_file.split('.')[0] for sensor_file in all_device_files]

    # draw single rep plot
    rep_sensor_id = args['representative']
    if rep_sensor_id:
        ylim = 15
        fig, ax = plt.subplots(1, 1)
        plt.setp(ax, ylim=(0, ylim))
        
        ax.set_xlabel('Round Range')
        ax.set_ylabel(f'{error_to_plot} Value')

        model_err_normalized = realtime_error_table[rep_sensor_id]
        
        num_of_plot_points = len(model_err_normalized[to_compare_model])
        
        # prepare for xticks
        ax.set_xticks(range(num_of_plot_points))
        ax.set_xticklabels([f'{r[0]}-{r[-1]}' for r in xticklabels])

        # start plotting with annotation
        model_iter = len(model_err_normalized)
        
        legend_handles = []
        for model_name in model_err_normalized:
            ax.plot(range(num_of_plot_points), model_err_normalized[model_name], label=model_name, color=COLORS[model_name])

            # compare this model vs. to_compare_model (e.g. neighbor_fl) and show smaller-error-value percentage, normalized. Smaller is better since we compare error, and we put to_compare_model as the first argument to compare_l1_smaller_equal_percent.

            if model_name != to_compare_model:
                to_compare_better_percent_val, to_compare_better_percent_string = compare_l1_smaller_equal_percent(model_err_normalized[to_compare_model], model_err_normalized[model_name])
                
                annotation_color = 'black'
                if to_compare_better_percent_val >= 0.5:
                    annotation_color = 'red'
                    if model_name == 'naive_fl':
                       ax.text(0.05, 0.95, '★', transform=ax.transAxes, fontsize=25, verticalalignment='top', horizontalalignment='left')
                # annotate fav vs other models
                ax.annotate(f"{NAMES[model_name]}:{to_compare_better_percent_string}", xy=(num_of_plot_points * 0.45, ylim * 0.55 + model_iter * 1.2), size=10, color=annotation_color)
                model_iter -= 1 

            # record legends
            vars()[f'{model_name}_curve'] = mlines.Line2D([], [], color=COLORS[model_name], label=NAMES[model_name])
            legend_handles.append(vars()[f'{model_name}_curve'])               
                
        ax.set_title(rep_sensor_id)

        # show legend
        ax.legend(handles=legend_handles, loc='upper right', prop={'size': 10})        
        
        fig.set_size_inches(10, 2) # width, height
        plt.savefig(f'{plot_dir_path}/real_time_errors_{rep_sensor_id}_{error_to_plot}.png', bbox_inches='tight', dpi=500)

    # draw subplots
    if rep_sensor_id:
       sensor_lists.remove(rep_sensor_id)
    # determine y_top (ylim)
    ylim = 0
    for realtime_errors in realtime_error_table.values():
        for model_errors in realtime_errors.values():
            max_err = max(model_errors)
            ylim = max_err if max_err > ylim else ylim
    ylim = min(20, ylim) # top threshold otherwise the curves in some plots are too flat

    if ROW == 1 and COL is None:
        COL = len(sensor_lists)
    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.setp(axs, ylim=(0, ylim))
    if ROW == 1 and COL == 1:
        axs.set_xlabel('Round Range', size=13)
        axs.set_ylabel(f'{error_to_plot} Value', size=13)
    elif ROW == 1 and COL > 1:
        axs[COL//2].set_xlabel('Round Range', size=13)
        axs[0].set_ylabel(f'{error_to_plot} Value', size=13)
    elif ROW > 1 and COL > 1:
        axs[ROW-1][COL//2].set_xlabel('Round Range', size=13)
        axs[ROW//2][0].set_ylabel(f'{error_to_plot} Value', size=13)

    model_to_red_label_counts = {key: 0 for key in realtime_error_table[list(realtime_error_table.keys())[0]].keys()}
    
    for sensor_plot_iter in range(len(sensor_lists)):
      
        row = sensor_plot_iter // COL
        col = sensor_plot_iter % COL

        if ROW == 1 and COL == 1:
          subplot = axs
        elif ROW == 1 and COL > 1:
          subplot = axs[sensor_plot_iter]
        elif ROW > 1 and COL > 1:
          subplot = axs[row][col]
          
        sensor_id = sensor_lists[sensor_plot_iter]
        model_err_normalized = realtime_error_table[sensor_id]
        
        num_of_plot_points = len(model_err_normalized[to_compare_model])
        
        # prepare for xticks
        subplot.set_xticks(range(num_of_plot_points))
        subplot.set_xticklabels([f'{r[0]}-{r[-1]}' for r in xticklabels], rotation=90)
        
        # start plotting with annotation
        model_iter = len(model_err_normalized)
        
        for model_name in model_err_normalized:
            subplot.plot(range(num_of_plot_points), model_err_normalized[model_name], label=model_name, color=COLORS[model_name])

            # compare this model vs. to_compare_model (e.g. neighbor_fl) and show smaller-error-value percentage, normalized. Smaller is better since we compare error, and we put to_compare_model as the first argument to compare_l1_smaller_equal_percent.

            if model_name != to_compare_model:
                to_compare_better_percent_val, to_compare_better_percent_string = compare_l1_smaller_equal_percent(model_err_normalized[to_compare_model], model_err_normalized[model_name])
                
                annotation_color = 'black'
                if to_compare_better_percent_val >= 0.5:
                    annotation_color = 'red'
                    model_to_red_label_counts[model_name] += 1
                    if model_name == 'naive_fl':
                       subplot.text(0.01, 0.95, '★', transform=subplot.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')

                # annotate annotate fav vs other models
                subplot.annotate(f"{NAMES[model_name]}:{to_compare_better_percent_string}", xy=(num_of_plot_points * 0.12, ylim * 0.58 + model_iter * 1.4), size=8, color=annotation_color)
                model_iter -= 1

        subplot.set_title(sensor_id)
    
    fig.subplots_adjust(hspace=hspace)
    fig.set_size_inches(width, length)
    plt.savefig(f'{plot_dir_path}/real_time_errors_all_sensors_{error_to_plot}.png', bbox_inches='tight', dpi=300)
    # plt.show()

    # print the comparison stat
    for model_name in model_to_red_label_counts:
       print(f"fav_beats_{model_name}: {model_to_red_label_counts[model_name]}")


def save_error_df(realtime_error_table, xticklabels, models_in_df):
    error_df = pd.DataFrame(columns=["ID", "Model"] + [f'R{r[0]}-R{r[-1]}' for r in xticklabels] + ["Surpass Percentage"])
    for device_id, model_errors in realtime_error_table.items():
        for model_name, error_values in model_errors.items():
            if model_name in models_in_df:
                to_compare_better_percent_val, _ = compare_l1_smaller_equal_percent(model_errors[models_in_df[0]], model_errors[model_name])
                new_row = [device_id, NAMES[model_name]] + error_values
                if model_name != models_in_df[0]:
                    new_row += [f"{to_compare_better_percent_val:.2%}"]
                else:
                   new_row += ["N/A"]
                error_df = error_df.append(pd.Series(new_row, index=error_df.columns), ignore_index=True)

    # highlight error values if beaten by model to compare

    error_df.to_csv(f'{plot_dir_path}/real_time_errors_all_sensors_{args["error_type"]}.csv')

def calc_average_prediction_error(realtime_error_table):
    model_to_errors_accum = {}
    for device_id, model_errors in realtime_error_table.items():
        for model_name, error_values in model_errors.items():
            if model_name not in model_to_errors_accum:
                model_to_errors_accum[model_name] = []
            model_to_errors_accum[model_name].append(np.average(error_values, axis=0))
    
    model_to_avg_err = {}
    for model, errors in model_to_errors_accum.items():
       model_to_avg_err[model] = round(np.average(errors, axis=0), 2)
       print(f"Avg {NAMES[model]} {args['error_type']}: {model_to_avg_err[model]}")
    return model_to_avg_err
    
realtime_error_table, xticklabels = construct_realtime_error_table(device_predicts)

# show plots
to_compare_model = 'neighbor_fl'
print(f"Plotting {args['error_type']}...")
plot_realtime_errors(realtime_error_table, to_compare_model, COL, xticklabels, 11, 10, 0.2)

models_in_df = ["neighbor_fl", "naive_fl", "central", "radius_naive_fl"] # index 0 model to compare
save_error_df(realtime_error_table, xticklabels, models_in_df)
# calc_average_prediction_error(realtime_error_table)