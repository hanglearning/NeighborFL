import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

import pandas as pd

import sys
import inspect

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
parser.add_argument('-ei', '--error_interval', type=int, default=100, help='unit is comm rounds, used in showing error table')
parser.add_argument('-row', '--row', type=int, default=1, help='number of rows in the plot')
parser.add_argument('-col', '--column', type=int, default=None, help='number of columns in the plot')
parser.add_argument('-er', '--end_round', type=int, default=None, help='provide the ending communication round, by default the last round of the simulation')

args = parser.parse_args()
args = args.__dict__

'''
Note that if there was X rounds in FL, the real time predictions were (X-1) rounds rather than X rounds. The first round's prediction will predict for 2nd round, and so.
'''

''' load vars '''
logs_dirpath = args['logs_dirpath']
with open(f"{logs_dirpath}/check_point/config_vars.pkl", 'rb') as f:
    config_vars = pickle.load(f)
# all_detector_files = config_vars["all_detector_files"]
with open(f"{logs_dirpath}/check_point/all_detector_predicts.pkl", 'rb') as f:
    detector_predicts = pickle.load(f)

all_detector_files = [detector_file.split('.')[0] for detector_file in detector_predicts.keys()]
    
ROW = args["row"]
COL = args["column"]
if ROW != 1 and COL is None:
    sys.exit(f"Please specify the number of columns.")
''' load vars '''

end_round = args["end_round"]
if not end_round:
    end_round = config_vars["resume_comm_round"] - 1

plot_dir_path = f'{logs_dirpath}/plots/realtime_errors_interval'
os.makedirs(plot_dir_path, exist_ok=True)

def calculate_errors(realtime_predicts):
  prediction_errors = {} # prediction_errors[sensor][model][error_type] = [error values based on comm round]
  for sensor_file, models_attr in realtime_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      prediction_errors[sensor_id] = {}

      for model, predicts in models_attr.items():
        if model != 'true':
          prediction_errors[sensor_id][model] = {}
          prediction_errors[sensor_id][model]['MAE'] = []
          prediction_errors[sensor_id][model]['MSE'] = []
          prediction_errors[sensor_id][model]['RMSE'] = []
          prediction_errors[sensor_id][model]['MAPE'] = []

          processed_rounds = set() # see plot_realtime_learning_curves.py, to be deleted in final version
          for predict in predicts:
            round = predict[0]
            if round > end_round:
                break
            if round not in processed_rounds:
              processed_rounds.add(round)
              data = predict[1]
              true_data = models_attr['true'][round - 1][1]
              prediction_errors[sensor_id][model]['MAE'].append(get_MAE(true_data, data))
              prediction_errors[sensor_id][model]['MSE'].append(get_MSE(true_data, data))
              prediction_errors[sensor_id][model]['RMSE'].append(get_RMSE(true_data, data))
              prediction_errors[sensor_id][model]['MAPE'].append(get_MAPE(true_data, data))
  return prediction_errors

def construct_realtime_error_table(realtime_predicts):
    realtime_error_table_normalized = {}
    for sensor_file, models_attr in realtime_predicts.items():
      sensor_id = sensor_file.split('.')[0]
      realtime_error_table_normalized[sensor_id] = {}
      
      for model, predicts in models_attr.items():
        if model != 'true':
          realtime_error_table_normalized[sensor_id][model] = {}
          realtime_error_table_normalized[sensor_id][model]['MAE'] = []
          realtime_error_table_normalized[sensor_id][model]['MSE'] = []
          realtime_error_table_normalized[sensor_id][model]['RMSE'] = []
          realtime_error_table_normalized[sensor_id][model]['MAPE'] = []

          processed_rounds = set() # see plot_realtime_learning_curves.py, to be deleted in final version
          data_list = []
          true_data_list = []
          for predict_iter in range(len(predicts)):
            predict = predicts[predict_iter] 
            # see plot_realtime_learning_curves.py, to be deleted in final version
            round = predict[0]
            if round > end_round:
                break
            if round not in processed_rounds:
              processed_rounds.add(round)
              data = predict[1]
              true_data = models_attr['true'][predict_iter][1]
              data_list.extend(data)
              true_data_list.extend(true_data)
              if round != 1 and (round - 1) % args["error_interval"] == 0:
                # conclude the errors
                realtime_error_table_normalized[sensor_id][model]['MAE'].append(get_MAE(true_data_list, data_list))
                realtime_error_table_normalized[sensor_id][model]['MSE'].append(get_MSE(true_data_list, data_list))
                realtime_error_table_normalized[sensor_id][model]['RMSE'].append(get_RMSE(true_data_list, data_list))
                realtime_error_table_normalized[sensor_id][model]['MAPE'].append(get_MAPE(true_data_list, data_list))
                data_list = []
                true_data_list = []
          # if there's leftover
          if data_list and true_data_list:
            realtime_error_table_normalized[sensor_id][model]['MAE'].append(get_MAE(true_data_list, data_list))
            realtime_error_table_normalized[sensor_id][model]['MSE'].append(get_MSE(true_data_list, data_list))
            realtime_error_table_normalized[sensor_id][model]['RMSE'].append(get_RMSE(true_data_list, data_list))
            realtime_error_table_normalized[sensor_id][model]['MAPE'].append(get_MAPE(true_data_list, data_list))
    return realtime_error_table_normalized
  

def plot_realtime_errors_one_by_one(prediction_errors, error_to_plot):
    for sensor_id, model_error in prediction_errors.items():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(f"Plotting {error_to_plot} error during real time FL simulation for {sensor_id} with interval {args['error_interval']}.")
        
        ax.plot(range(1, len(model_error['stand_alone'][error_to_plot]) + 1), model_error['stand_alone'][error_to_plot], label='stand_alone', color='#ffb839')
        
        ax.plot(range(1, len(model_error['naive_fl'][error_to_plot]) + 1), model_error['naive_fl'][error_to_plot], label='naive_fl', color='#5a773a')
        
        ax.plot(range(1, len(model_error['fav_neighbors_fl'][error_to_plot]) + 1), model_error['fav_neighbors_fl'][error_to_plot], label='fav_neighbors_fl', color='#5a773a')
        
        ax.plot(range(1, len(model_error['brute_force'][error_to_plot]) + 1), model_error['brute_force'][error_to_plot], label='brute_force', color='#5a773a')
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid(True)
        plt.xlabel(f"Comm Round with Interval {args['error_interval']}")
        plt.ylabel('Error')
        plt.title(f'{sensor_id} - Real Rime {error_to_plot} Error')
        fig = plt.gcf()
        # fig.set_size_inches(228.5, 10.5)
        fig.set_size_inches(10.5, 3.5)
        print()
        plt.savefig(f"{plot_dir_path}/{sensor_id}_{error_to_plot}_interval_{args['error_interval']}.png", bbox_inches='tight', dpi=100)
        plt.show()

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

def plot_realtime_errors_all_sensors(realtime_error_table, error_to_plot, to_compare_model, COL):

    sensor_lists = [sensor_file.split('.')[0] for sensor_file in all_detector_files]
    
    # determine y_top (ylim)
    ylim = 0
    for realtime_errors in realtime_error_table.values():
        for model_errors in realtime_errors.values():
            max_err = max(model_errors[error_to_plot])
            ylim = max_err if max_err > ylim else ylim

    # if ROW == 1:
    #   COL = len(sensor_lists)
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
    
      
    # axs[0].set_ylabel(f'{error_to_plot} Error', size=13)
    # fig.text(0.5, 0.04, 'Comm Round Index', ha='center', size=13)
    # axs[COL//2].set_xlabel(f'Round Range', size=13)

    # count red labels
    # naive = 0
    # standalone = 0
    # same_dir = 0
    # same_vs_naive = 0
    model_to_red_label_counts = []
    
    for sensor_plot_iter in range(len(sensor_lists)):
      
        row = sensor_plot_iter // COL
        col = sensor_plot_iter % COL

        if ROW == 1 and COL == 1:
          subplots = axs
        elif ROW == 1 and COL > 1:
          subplots = axs[sensor_plot_iter]
        elif ROW > 1 and COL > 1:
          subplots = axs[row][col]
          
        sensor_id = sensor_lists[sensor_plot_iter]
        model_error_normalized = realtime_error_table[sensor_id]
        
        subplots.set_title(sensor_id)
        
        num_of_plot_points = len(model_error_normalized['stand_alone'][error_to_plot])
        e_interval = args["error_interval"]
        
        
        subplots.set_xticks([0, num_of_plot_points // 2, num_of_plot_points - 1])
        subplots.set_xticklabels([f'  2-\n{(num_of_plot_points // 2 - 1) * e_interval}', f'{(num_of_plot_points // 2 - 1) * e_interval + 1}-\n{(num_of_plot_points - 1) * e_interval}', f'{(num_of_plot_points - 1) * e_interval + 1}-\n{end_round}'], fontsize=8)
        
        
        # start plotting with annotation
        for model_err in model_error_normalized:
            subplots.plot(range(0, num_of_plot_points), model_error_normalized[model_err][error_to_plot], label=model_err, color='#ffb839')

        
        # compare naive_fl vs. fav_neighbors_fl and show smaller-error-value percentage, normalized. Smaller is better since we compare error, and we put what we expect to surpass as the first argument.


        fav_better_percent_val_fav_vs_naive, fav_better_percent_string_fav_vs_naive = compare_l1_smaller_equal_percent(model_error_normalized['fav_neighbors_fl'][error_to_plot], model_error_normalized['naive_fl'][error_to_plot])
        # same for standalone vs. fav_neighbors_fl
        fav_better_percent_val_fav_vs_standalone, fav_better_percent_string_fav_vs_standalone = compare_l1_smaller_equal_percent(model_error_normalized['fav_neighbors_fl'][error_to_plot], model_error_normalized['stand_alone'][error_to_plot])
        # same for same_dir_fl vs. fav_neighbors_fl
        try:
            fav_better_percent_val_fav_vs_same_dir, fav_better_percent_string_fav_vs_same_dir = compare_l1_smaller_equal_percent(model_error_normalized['fav_neighbors_fl'][error_to_plot], model_error_normalized['same_dir_fl'][error_to_plot])
            # compare same_dir_fl vs. native_fl
            same_dir_percent_val_same_dir_vs_naive, same_dir_percent_string_same_dir_vs_naive = compare_l1_smaller_equal_percent(model_error_normalized['same_dir_fl'][error_to_plot], model_error_normalized['naive_fl'][error_to_plot])
        except:
           pass
           # print("same_dir_fl skipped")
           

        if fav_better_percent_val_fav_vs_naive >= 0.5:
          annotation_color_fav_vs_naive = 'red'
          naive += 1
        else:
          annotation_color_fav_vs_naive = 'black'

        if fav_better_percent_val_fav_vs_standalone >= 0.5:
          annotation_color_fav_vs_standalone = 'red'
          standalone += 1
        else:
          annotation_color_fav_vs_standalone = 'black'

        try:
            if fav_better_percent_val_fav_vs_same_dir >= 0.5:
                annotation_color_fav_vs_same_dir = 'red'
                same_dir += 1
            else:
                annotation_color_fav_vs_same_dir = 'black'

            if same_dir_percent_val_same_dir_vs_naive >= 0.5:
                annotation_color_same_dir_vs_naive = 'red'
                same_vs_naive += 1
            else:
                annotation_color_same_dir_vs_naive = 'black'
        except:
           pass
           # print("same_dir_fl skipped")

        # annotate fav_vs_naive
        subplots.annotate(f">=BFRT:{fav_better_percent_string_fav_vs_naive}", xy=(num_of_plot_points * 0.4, ylim * 0.65), size=8, color=annotation_color_fav_vs_naive)
        # annotate fav_vs_standalone
        subplots.annotate(f">=BASE:{fav_better_percent_string_fav_vs_standalone}", xy=(num_of_plot_points * 0.4, ylim * 0.6), size=8, color=annotation_color_fav_vs_standalone)
        
        try:
            subplots.annotate(f">=SAME:{fav_better_percent_string_fav_vs_same_dir}", xy=(num_of_plot_points * 0.4, ylim * 0.55), size=8, color=annotation_color_fav_vs_same_dir)
            subplots.annotate(f"SD>=FL:{same_dir_percent_string_same_dir_vs_naive}", xy=(num_of_plot_points * 0.4, ylim * 0.5), size=8, color=annotation_color_same_dir_vs_naive)
        except:
           pass
           # print("same_dir_fl skipped")


        '''
        # compare fav_neighbors_fl with stand_alone AND naive_fl
        fav_best_percent_val, fav_best_percent_string = compare_l1_smallest_equal_percent(model_error_normalized['fav_neighbors_fl'][error_to_plot], model_error_normalized['stand_alone'][error_to_plot], model_error_normalized['naive_fl'][error_to_plot])
        if fav_best_percent_val >= 0.5:
          annotation_color = 'red'
        else:
          annotation_color = 'black'
        
        subplots.annotate(fav_best_percent_string, xy=(num_of_plot_points - 5, model_error_normalized['stand_alone'][error_to_plot][-5] + 10), size=8, color=annotation_color)
        '''
        
        subplots.plot(range(len(model_error_normalized['naive_fl'][error_to_plot])), model_error_normalized['naive_fl'][error_to_plot], label='naive_fl', color='#5a773a')

        try:
            subplots.plot(range(len(model_error_normalized['same_dir_fl'][error_to_plot])), model_error_normalized['same_dir_fl'][error_to_plot], label='same_dir_fl', color='pink')
        except:
            pass
            # print("same_dir_fl skipped")

        subplots.plot(range(len(model_error_normalized['fav_neighbors_fl'][error_to_plot])), model_error_normalized['fav_neighbors_fl'][error_to_plot], label='fav_neighbors_fl', color='blue')

        
        
        if sensor_plot_iter == 0:   
          baseline_curve = mlines.Line2D([], [], color='#ffb839', label="BASE")
          global_curve = mlines.Line2D([], [], color='#5a773a', label="BFRT")
          if 'same_dir_fl' in model_error_normalized.keys():
            same_dir_curve = mlines.Line2D([], [], color='pink', label="SAME_DIR")
          fav_neighbors_fl_curve = mlines.Line2D([], [], color='blue', label="KFRT")
        
        if 'same_dir_fl' in model_error_normalized.keys():
          subplots.legend(handles=[baseline_curve, global_curve, same_dir_curve, fav_neighbors_fl_curve], loc='best', prop={'size': 10})
        else:
           subplots.legend(handles=[baseline_curve, global_curve, fav_neighbors_fl_curve], loc='best', prop={'size': 10})
    # show legend on each plot
    # baseline_curve = mlines.Line2D([], [], color='#ffb839', label="BASE")
    # global_curve = mlines.Line2D([], [], color='#5a773a', label="FED")    
    # fig.legend(handles=[baseline_curve, global_curve], loc='upper center')
    fig.set_size_inches(2 * len(sensor_lists), 3.8)
    plt.savefig(f'{plot_dir_path}/real_time_errors_all_sensors_{error_to_plot}.png', bbox_inches='tight', dpi=300)
    # plt.show()

    if 'same_dir_fl' in model_error_normalized.keys():
        return [f"fav_beats_naive: {naive}", f"fav_beats_standalone: {standalone}", f"fav_beats_same_dir: {same_dir}", f"same_beats_naive: {same_vs_naive}"]
    return [f"fav_beats_naive_FL: {naive}", f"fav_beats_standalone: {standalone}"]
    
realtime_error_table = construct_realtime_error_table(detector_predicts)

# show table
if False:
    with open(f'{plot_dir_path}/errors.txt', "w") as file:
        for sensor_id, model in realtime_error_table.items():  
            file.write(f'\nfor {sensor_id}')
            error_values_df = pd.DataFrame.from_dict(model)
            file.write(tabulate(error_values_df.round(2), headers='keys', tablefmt='psql'))
            file.write('\n')
if False:
    with open(f'{plot_dir_path}/errors_2nd.txt', "w") as file:
        for sensor_id, model in realtime_error_table.items():  
            file.write(f'\nfor {sensor_id}')
            error_values_df = pd.DataFrame.from_dict(model)
            file.write(f"{sensor_id} stand_alone: {error_values_df['stand_alone']['MAE'][0]}, fav: {error_values_df['fav_neighbors_fl']['MAE'][0]}, {error_values_df['stand_alone']['MAE'][0] == error_values_df['fav_neighbors_fl']['MAE'][0]}")
            file.write('\n')
    
# show plots
all_prediction_errors = calculate_errors(detector_predicts) # calculate global model outperform percentage

# err_type_to_y_top = {"MAE": }
to_compare_model = 'fav_neighbors_fl'
for err_type in ["MAE", "MSE", "MAPE", "RMSE"]:
    print(f"Plotting {err_type}...")
    red_label_counts = plot_realtime_errors_all_sensors(realtime_error_table, err_type, to_compare_model, COL)
    print(red_label_counts)
# error_ylim = dict(MAE = 200, MSE = 6000, RMSE = 80, MAPE = 0.6)
# for error_type in error_ylim.keys():
#   plot_realtime_errors_all_sensors(realtime_error_table, error_type, error_ylim[error_type])