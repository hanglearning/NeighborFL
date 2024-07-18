# NeighborFL (Simulation)

### Please note the paper of this work is still under review and not available for public yet.

## Introduction

<b>NeighborFL</b> is an individualized real-time federated learning scheme for traffic prediction that incorporates haversine distance as a heuristic and  error-driven  local  models  grouping  from  the  aspect  of  each  individual traffic device. This approach allows NeighborFL to create location-aware and tailored prediction models for each client while fostering collaborative learning. It can be seen as an extension of our prior work, <b>BFRT</b> ([paper](https://drive.google.com/file/d/1WtmQeVPWmMUTlkD0xYskHNRQ6GRrYiyM/view)), which uses a conventional FL aggregation method and equates to <b>NaiveFL</b> in this codebase.

Please refer to [the NeighborFL paper](https://arxiv.org/pdf/2407.12226) for detailed explanations of the framework.

## Run the Simulation

### <ins>Suggested</ins> Running Environment

We recommend using Google Colab to run this codebase, as it simplifies the setup process compared to running it locally. With the built-in resume functionality ([explained later](#resume-a-simulation)), the free version of Colab can handle simulations with a large number of specified communication rounds.

We have provided a [sample Colab file](https://colab.research.google.com/drive/1Rhy3YCM3MIXUBYr3YriESiuvcu203eyh) that demonstrates the steps for running the code. Detailed instructions for all the functionalities explained later in this readme file can be found in the [Colab file](https://colab.research.google.com/drive/1Rhy3YCM3MIXUBYr3YriESiuvcu203eyh).

If you prefer to set up and run the code on your local machine, we assume you are already familiar with setting up libraries such as numpy, pandas, and TensorFlow for your OS. In that case, you can use the same commands provided in the following sections to run the code.

### 1. Steps to run the simulation

The gateway to running simulations is [main.py](https://github.com/hanglearning/NeighborFL/blob/main/main.py). It supports simulating four methods: Central, NaiveFL, r-NaiveFL, and NeighborFL, which can be manually enabled or disabled using the `-lm` argument (explained later). The program scans a folder containing your traffic data (csv) files, identifies each file as a traffic device, and processes the chosen feature (e.g., volume, speed, occupancy) as input for the selected model (e.g., LSTM, GRU, or your customized model). The data processing logic can be found in [process_data.py](https://github.com/hanglearning/NeighborFL/blob/main/process_data.py). After loading the data files, the program allows each device to assign its candidate neighbors within a specified radius if NeighborFL is enabled, and then the simulation begins. Unless the number of communication rounds is set via the command line (using `-c`), the code calculates the maximum possible rounds based on the number of data points in the provided files and stops after the last possible round. To assess NeighborFL's performance against the baseline methods, the code includes [two plotting scripts](#evaluate-performance-by-plots-and-error-tables) for generating average prediction error tables, prediction and error curve plots as shown in Section IV of the paper.


### (1) Prepare the dataset

main.py will look for the traffic data (csv) files inside of the `data` directory residing at the root of the downloaded repo. By default, we have included the data files we used for our experiments shown in the paper, with each file named after its sensor ID. Additionally, there is another csv file containing the GPS locations of these 26 sensors.

If you have your own dataset, you may either replace the files in the `data` with your data files or specify the absolute path to your dataset folder using the `-dp` argument (as explained [later](#i-common-arguments-for-both-pretrainpy-and-mainpy)). **Please make sure that you include a csv file containing the GPS locations of the traffic devices, if NeighborFL and/or r-NaiveFL will be simulated.** This file should have "location" in its file name, as indicated by our example file. The other csv files containing traffic data should be named after the corresponding traffic device's ID and must contain the learning/prediction feature (e.g., speed, volume, occupancy) specified using the `-f` command line argument.

For dataset formatting, you can refer to [the datasets we derived](https://drive.google.com/drive/folders/1q_IPvLyWw7zJKSyQ89N8fGWVLAfRXaN8?usp=drive_link) from the publicly available [PEMS-BAY dataset](https://zenodo.org/record/5146275) used in our paper. [Our datasets](https://drive.google.com/drive/folders/1q_IPvLyWw7zJKSyQ89N8fGWVLAfRXaN8?usp=drive_link) contain speed data from our selected 26 sensors, ranging from January 2017 to June 2017.
  

### (2) Run a new simulation

#### ① Without pretrained models
Sample running command -

    $ python NeighborFL/main.py -c 250 -lm 1111 -m LSTM -ms 72 -r 1 -f Speed -et MSE -ks 3 -nu 1 -e 5 -b 1 -I 12 -O 1 -seed 40

Arguments explanation: 

1. `-c 250`: Runs the simulation for 250 communication rounds.
2. `-lm 1111`: Enables the simulation for (1) Central, (2) NaiveFL, (3) r-NaiveFL, and (4) NeighborFL.
3. `-m LSTM`: Configures devices to use an LSTM model for centralized/federated learning.
4. `-ms 72`: Sets the maximum data size to 72 while learning, denoted as _MaxDataSize_ in the paper.
5. `-r 1`: Specifies a radius of 1 for all devices to search for their candidate favorite neighbors.
6. `-f Speed`: Selected feature used for training and inferencing set to Speed
7. `-et MSE`: Specifies the error type used to evaluate candidates in NeighborFL, in this case, mean squared error. This corresponds to the _CalcError_ method in Algorithm 6 of the paper.
8. `-rs 3`: Sets the remove strategy to "remove the last added one," referring to Algorithm 8 in the paper.
9. `-nu 1`: Defines the $\nu$ value as 1, meaning a device removes the last added favorite neighbor whenever the real-time prediction error increases.
10. `-e 5`: Local learning epochs set to 5.
11. `-b 1`: Local learning batch size set to 1.
12. `-I 12`: Specifies the input length of the model, indicating the number of input neurons for the chosen LSTM model.
13. `-O 1`: Sets the output length of the model, which also represents the prediction horizon.
14. `-sd 40`: Set up random seed to a fixed integer, in this case 40, for reproduction and comparison.

In a nutshell, this command configures each device (identified by its individual csv data file) to create its candidate favorite neighbors set within a 1-mile radius, centered around itself. It employs a randomly initialized LSTM model (currently identical across all devices) with 12 input neurons and 1 output neuron for learning, with 5 local epochs, a batch size of 1, and a maximum data size set to 72. The simulation will run for 250 communication rounds for Central, NaiveFL, r-NaiveFL, and NeighborFL methods. In the case of NeighborFL, the error metric used to assess candidates is the mean squared error, and the favorite neighbor removal mechanism follows the "remove the last added one" approach whenever the real-time prediction error increases. This command corresponds to the **Central, NaiveFL, r-NaiveFL, and NeighborFL L1 methods without pretrained models** outlined in the paper.

#### ② With pretrained models

To provide pretrained or customized models as initial models for Central and/or federated methods, please store these pretrained model files (in .h5 format) in a designated folder and specify the folder's path using the `-pp` argument to `main.py`. Each device should have its own pretrained model file with a name identical to its corresponding dataset file. You can refer to the [pretrained model files we used in our experiment](https://drive.google.com/drive/folders/1WDqxGWr8iHX1uxZmA8gRjPN5MI8ul7_c?usp=drive_link) for an example. Our pretrained models were generated using the provided [pretrain.py](https://github.com/hanglearning/NeighborFL/blob/main/pretrain.py).

Sample run of `pretrain.py` -

    $ python NeighborFL/pretrain.py -dp /path/to/datasets -si 0 -ei 2016 -e 5 -f Speed -m LSTM  -b 1 -I 12 -O 1 -sp /pretrained/models/save/path

Explanation of the arguments:

1. `-dp`: Specifies the path to the dataset csv files. If not provided, the program will search for datasets in the `data` directory at the root of the downloaded repository.
2. `-si 0`: Sets the start data index for pretraining to 0, which corresponds to the first data point in the csv file.
3. `-ei 2016`: Defines the end data index for pretraining as 2015 (inclusive). Note that since datasets are zero-based, the data point at index 2016 is not included. `-si 0` and `-ei 2016` together select the first 2016 data points in the csv file(s) as the pretraining dataset.
4. `-e 5`: Sets the number of local training epochs to 5. Each local epoch covers a device's entire dataset once.
5. `-f Speed -m LSTM  -b 1 -I 12 -O 1`: Functions similarly to `main.py`.
6. `-sp`: Specifies the save path for pretrained models. If not provided, the models will be saved in the `pretrained_models` directory located at the root of the downloaded repository.

The program will save the generated pretrained models in the folder specified using the `-sp` argument once it completes execution. The program will also save the arguments used to run the program to recall the generated model structure conveniently.

To make these models available to `main.py`, please specify this path using the `-pp` argument to `main.py`. Additionally, it's advisable to set the `-si` argument to `main.py` to indicate the starting data index from which devices should begin collecting data for real-time learning and inference. If you've used the aforementioned arguments with `pretrain.py`, a straightforward way to seamlessly transition from pretraining to real-time training in `main.py` is to:

1. Keep the `-sp` argument in `pretrain.py` and `main.py` the same, ensuring both programs use the same folder for storing and accessing pretrained models.
2. Keep the `-ei` argument in `pretrain.py` and the `-si` argument in `main.py` the same. This way, devices can start real-time training immediately after the pretraining period ends.

Here's a sample command for running `main.py` with the pretrained models generated by the previously used command for `pretrain.py`. This command initiates the simulation with the data index immediately following the pretraining end index -

    $ python NeighborFL/main.py -pp /pretrained/models/save/path -si 2016 -c 250 -lm 1111 -m LSTM -ms 72 -r 1 -et MSE -ks 3 -nu 1 -e 5 -b 1 -I 12 -O 1 -sd 40

This command corresponds to the **Central, NaiveFL, r-NaiveFL, and NeighborFL L1 methods with pretrained models** outlined in the paper.

**⚠️ Please note that the pretrained models must match the model structure used for simulations in `main.py`, including aspects such as input and output length, the number of hidden layers, neurons, and so on.**

### (3) Simulation logs

Each time a new simulation is executed, a log folder is created under the path specified by `-lb` to `main.py`. If `-lb` is not specified, the folder is created within the `logs` directory located at the root of the downloaded repository. The log folder is named with the simulation execution date and time as the prefix, accompanied by customizable argument indicators configured in `main.py`. For instance, a log folder named `09022023_202131_lstm_mds_72_I_12_O_3` indicates that the program was run on September 2nd, 2023, at 20:21:31, based on the local time zone of the running environment, with an LSTM network featuring 12 input neurons and 3 output neurons, with a max data size of 72 for the simulation methods.

A specific log folder has a text file recording the command line and default arguments while executing the program the 1st time. The log folder also has a subdolder named `check_point`, which includes a file (`config_vars.pkl`) recording these arguments in order to resume a simulation later, and another file (`all_device_predicts.pkl`) recording the prediction values categorized by communication rounds by all the devices for the enabled simulation methods. This file is useful when plotting the prediction curves and analyzing model performances. Another useful file (`fav_neighbors_by_round.pkl`) records the favorite neighbors set of each device categorized by communication rounds, which may be useful in our future work. 

**Note that the program will print out the path to its log folder upon execution for your convenience.**

### (4) Resume a simulation

The program supports resuming the simulation process in case the simulation is interrupted due to time/resource limitation of colab or your running environment. As mentioned above, each new simulation will create a log folder. To resume a simulation, the **only** argument you need is `-rp`. A sample command is

      $ python NeighborFL/main.py -rp NeighborFL/logs/09022023_202131_lstm_mds_72_I_12_O_3

`rp` stands for "resume path". This is the path to the log folder of the interrupted simulation. The program will automatically resume from the interrupted communication round, utilizing the latest models stored in the specified log folder and continuing with the remaining communication rounds. This command can be resued again and again until all the specified communication rounds are exhausted.

You may also overwrite some arguments while resuming, such as providing a higher value for `-c` to extend the simulation with more communication rounds than initially configured. The overwritten arguments will also be updated in `config_vars.pkl` for resuming next time. However, please note that some arguments may not support being overwritten, like altering the network model structure by providing `-I` and/or `-O` values different from the models stored in the log folder. Doing so will result in runtime errors, unless you replace the model files in the log folder with the structure of the models that align with the new argument values. **The `-dp` argument for the dataset path does not support overwriting, as the processed dataset is stored in the log folder for convenient resumption.**
  

### All available arguments to [pretrain.py](https://github.com/hanglearning/NeighborFL/blob/main/pretrain.py) and [main.py](https://github.com/hanglearning/NeighborFL/blob/main/main.py)

#### I. Common Arguments for both `pretrain.py` and `main.py`

| Argument | Type | Default Value | Description |
| --- | --- | --- | --- |
| `-dp`, `--dataset_path` | str | 'NeighborFL/data/' | Path to the traffic datasets. |
| `-sd`, `--seed` | int | 40 | Random seed value specified for reproducibility. |
| `-m`, `--model` | str | 'LSTM' | Choose our predefined LSTM or GRU model for Central/Federated simulations. You may also write your own model in `create_model.py`. |
| `-I`, `--input_length` | int | 12 | Input length for the chosen model, i.e., our predefined GRU/LSTM or your customized model. |
| `-O`, `--output_length` | int | 1 | Output length for the chosen model, i.e., our predefined GRU/LSTM or your customized model. |
| `-hn`, `--hidden_neurons` | int | 128 | Number of neurons in one of each 2 hidden layers in our predefined GRU/LSTM network. |
| `-op`, `--optimizer` | str | 'rmsprop' | Optimizer for training. |
| `-b`, `--batch` | int | 1 | Batch number for training. |
| `-lo`, `--loss` | str | 'mse' | Loss function used for training, refer to [TensorFlow Losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss). |
| `-f`, `--feature` | str | 'Speed' | Feature used for training (and prediction in `main.py`), depending on your traffic dataset, usually speed, volume, occupancy. |

#### II. Arguments specifically for `pretrain.py`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-me`, `--metric` | str | 'mse' | Metric used for evaluation by the model. Refer to [TensorFlow Model Compilation](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) and [TensorFlow Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric) for more details. |
| `-e`, `--epochs` | int | 5 | Local epoch number. One complete epoch goes through the entire pretraining dataset once. |
| `-pp`, `--pretrain_percent` | float | 1.0 | The percentage of the provided dataset files allocated for pretraining. |
| `-si`, `--pretrain_start_index` | int | 0 | The starting data index (row in pandas dataframe) for pretraining |
| `-ei`, `--pretrain_end_index` | int | None | End data index (not included) for pretraining. If this is provided, overwrite -pp |
| `-sp`, `--model_save_path` | str | 'NeighborFL/pretrained_models/' | The path to save the pretrained models |

#### III. Arguments specifically for `main.py`

#### (a) Arguments for System Variables


| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-lb`, `--logs_base_folder` | str | 'NeighborFL/logs/' | Base folder path to store running logs and h5 files for plotting and resuming |
| `-pm`, `--preserve_historical_models` | int | 1 | Whether to preserve models from old communication rounds. This is useful when you run the program on colab and have limited Google Drive storage. Set to `1` to only preserve the latest models that are enough for resumption of the simulation. `0` to store all historical models. |
  
#### (b) Arguments for Resume Learning

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-rp`, `--resume_path` | str | None | Provide the interrupted log folder path to continue the simulation. Note that the program will print out this folder path upon executing |

#### (c\) Arguments for Central/Federated Real-time Learning


| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-c`, `--comm_rounds` | int | None | Number of communication rounds, default aims to run until all data is exhausted. |
| `-ms`, `--max_data_size` | int | 72 | Maximum data length for training in each communication round, simulating the memory size the devices have. |
| `-tau1`, `--tau1` | int | 24 | Number of data point collections and prediction amount in round 1, rule of thumb: 2 * `--tau2p`. |
| `-tau2p`, `--tau2p` | int | 12 | Number of data points collected, and also the number of predictions made in round 2 and beyond, rule of thumb - same as `-I`. |
| `-lm`, `--learning_methods` | str | 1111 | Enable learning methods for simulation. Use `1` to enable and `0` to disable. The indices correspond to: 1. Baseline centralized, 2. Pure FedAvg NaiveFL, 3. RadiusFL, 4. NeighborFL. |


#### (d) Arguments for Learning

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-me`, `--metric` | str | 'mse' | Evaluation metric by the local models during training and testing. This is essentially the same as the `-me` used in pretrain.py, but it should not be confused with `-et`. |
| `-e`, `--epochs` | int | 5 | Number of epochs per communication round for the enabled simulation methods. Each epoch processes the latest _MaxDataSize_ data points once. |
| `-dlp`, `--data_load_percent` | float | 1.0 | The percentage of data to load. Note that the program calculates the maximum number of communication rounds based on the number of data points provided. If the specified `-c` is larger than the calculated maximum communication rounds, `-c` will be overwritten. |
| `-si`, `--start_train_index` | int | 0 | Start the simulation at this data index, usually to accommodate pretrained models. |
| `-pp`, `--pretrained_models_path` | str | None | Pretrained models path. If not provided, the program will create an initial model for all devices and train from scratch. |

#### (e) Arguments for NeighborFL

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-r`, `--radius` | float | 1 | Specifies the radius within which participants are treated as candidate favorite neighbors. Unit - mile. |
| `-et`, `--error_type` | str | MSE | Defines the error type used to evaluate candidate neighbors. Not to be confused with `-me`. |
| `-nt`, `--num_neighbors_try` | int | 1 | Determines how many new neighbors to evaluate in each communication round. |
| `-ah`, `--add_heuristic` | int | 1 | Sets the heuristic for adding favorite neighbors: 1 - by distance from close to far, 2 - randomly. |
| `-rt`, `--remove_trigger` | int | 2 | Specifies the removal trigger: 0 - never remove; 1 - trigger by probability (set by -ep); 2 - trigger by consecutive rounds of error increase (set by -nu). |
| `-ep`, `--epsilon` | float | 0.2 | Used when -rt is set to 1; devices have a probability to remove their worst neighbors to explore new ones. |
| `-nu`, `--remove_rounds` | int | 3 | Triggers removal if the error of a device's NeighborFL aggregated model has increased for `-nu` rounds. If `-nu` is set to 1, removal occurs as soon as the error increases. |
| `-rn`, `--remove_num` | int | 1 | Specifies how many favorite neighbors to remove based on their reputation. |
| `-rs`, `--remove_strategy` | int | 1 | Removal strategy: 1 - remove by worst reputation; 2 - remove randomly; 3 - always remove the last added one. |
  

## Evaluate Performance by Plots and Error Tables
  
The code for plotting the experimental results and generating the tables in the paper are provided in the <i>plotting_code</i> folder. [plot_prediction_curves_save_err_table.py](https://github.com/hanglearning/NeighborFL/blob/main/plotting_code/plot_prediction_curves_save_err_table.py) was used to plot Figure 3 and create Table III, IV and V.  [plot_error_curves.py](https://github.com/hanglearning/NeighborFL/blob/main/plotting_code/plot_error_curves.py) was used to plot Figure 4 in the paper. The code will look for `config_vars.pkl` and `all_device_predicts.pkl` files in the `check_point` folder under the specified log folder for these tasks.

### (1) Plot the real-time prediction curves and generate error table

Sample command:

    $ python NeighborFL/plotting_code/plot_prediction_curves_save_err_table.py -lp /NeighborFL/logs/08282023_233406_lstm_I_12_O_3_lm_1111 lp2 /NeighborFL/logs/09022023_202131_lstm_I_12_O_3_lm_0001 -NFLConfig L1 -r 400760_N -et MSE -pr 24 -Oseqs 13 -row 5

Arguments explanation:

 1. `-lp`: the path of the desired log folder for plotting.
 2. `-lp2`: this is an optional second log path that allows you to overwrite NeighborFL predictions in `-lp`. It comes in handy when you're running multiple simulations of NeighborFL and want to compare with the same baseline methods. For example, in simulation A, you might run all three baseline methods and one NeighborFL configuration, while in simulation B, you want to compare a different NeighborFL configuration with the baseline methods from simulation A. For instance, let's say the simulation run on 08282023_233406 (provided in `-lp` in the command) includes all three baseline methods and a NeighborFL configuration (indicated by `-lm` set to 1111, as referenced in the arguments in `main.py`). On the other hand, the simulation run on 09022023_202131 (provided in `-lp2`) only runs a NeighborFL simulation (indicated by `-lm` set to 0001). As a result, the NeighborFL predictions in 09022023_202131 will overwrite the NeighborFL predictions in 08282023_233406 for plots, tables, and various types of analysis while comparing with the predictions from the baseline methods in 08282023_233406. Notice that if `-lp2` is provided, the plots and tables will be saved under the plot folder within the path indicated in `-lp2` rather than `-lp`.
 3. `-NFLConfig`: the NeighborFL configuration name to show on plots and tables, such as R1, R3, L1, L3.
 4. `-r` denotes "representative", allows you to specify a particular traffic device to be showcased in a large figure, i.e., spans the top row of the figures, such as in Figure 3 and 4 of the paper.
 5. `-et`: Specify the error type for analysis to make the error tables within the specified round range.
 6. `-pr` determines the number of the latest rounds of the simulation to include in the plots. Using `-pr 24` will generate plots for the most recent 24 rounds of predictions, sourced from the `all_device_predicts.pkl` file.
 7. `-Oseqs` specifies which output sequences to plot for the prediction instances with a prediction horizon greater than 1. For example, when the `output_length` is 4, and you have two consecutive prediction instances in two continuous communication rounds A and B, like [50.72, 51.77, 49.75, 66.02] and [66.92, 49.24, 45.05, 55.12], setting `-Oseqs 13` will generate figures for the 1st predictions, taking 50.72 from round A and 66.92 from round B, and another figure for the 3rd predictions, taking 49.75 from round A and 45.05 from round B. This is useful for evaluating prediction performance in scenarios where long-term forecasting is required, such as predicting the impacts of non-recurrent events.
 8. `-row` specifies how many rows to arrange the plots for traffic devices except the representative. The column number will be automatically determined. For instance, if the log has 26 devices and -row set to 5, each row will have 5 plots based on ⌈(26 - 1) / 5⌉ = 5, then the subplots will have 5 columns.

Suppose we have 26 devices running for 250 communication rounds, this sample command will plot the most recent 24 rounds (round 227 to round 250, both inclusive) of real-time prediction values for all three baseline methods, as recorded in the `08282023_233406` simulation, and the NeighborFL method, as recorded in the `09022023_202131` simulation, for output horizon 1 and 3. The program will plot for all 26 devices, with the traffic device 400760_N as the representative, and render the rest of the devices into 5*5 subplots, just like Figure 3 in the paper. If you have a different number of devices, you can adjust the `-row` value as needed. The resulting figures will be saved in a folder named `prediction_curves` within the `plots` directory located under the specified log folder. If `-lp2` is provided, the figures are saved in the `plots` directory within the path of `-lp2`, otherwise `-lp`.

Additionally, the program will generate a table named `error.csv` in the `plots` folder, which calculates individual device MSE values over the specified round range consistent to the plotting range, akin to Table IV in the paper. Furthermore, the program will print (in the console) the average device MSE values across all devices for the specified round range, which we used to create Tables III and V in the paper.


### (2) Plot the real-time learning error curves

Sample command:

    $ python NeighborFL/plotting_code/plot_error_curves.py -lp /NeighborFL/logs/08282023_233406_lstm_I_12_O_3_lm_1111 lp2 /NeighborFL/logs/09022023_202131_lstm_I_12_O_3_lm_0001 -NFLConfig L1 -r 400760_N -et MSE -Oseqs 13 -row 5 -ei 24

  

Arguments explanation:

1. `-ei` represents error interval, unit is communication rounds. This is used to create smoother error curves, especially when dealing with a large number of communication rounds, as seen in Figure 4 of the paper. In this example, `-ei 24` is used to achieve a smoothed representation of errors across the entire 250 rounds while maintaining the overall error trend.

2. All remaining arguments operate in the same manner as they do for `plot_prediction_curves_save_err_table.py`.
  
In a nutshell, this sample command will plot the MSE value trends across the entire simulation for the real-time learning process using the same data (`all_device_predicts.pkl`) and plot style as demonstrated in the previous example for plotting prediction curves. The resulting plot is akin to Figure 4 in the paper. The code also calculates and shows the surpass percentage between NeighborFL and available baselines on the figure and labels a star if NeighborFL surpasses NaiveFL for 50% or more. The generated figures will be saved in the `plots/error_curves` directory, which can be found under either the `plot` folder within the `-lp2` specified log folder or under the `plot` folder within the `-lp` specified log folder if `-lp2` is not provided.

### All available arguments to [plot_prediction_curves_save_err_table.py](https://github.com/hanglearning/NeighborFL/blob/main/plotting_code/plot_prediction_curves_save_err_table.py) and [plot_error_curves.py](https://github.com/hanglearning/NeighborFL/blob/main/plotting_code/plot_error_curves.py)

#### I. Common Arguments for both `plot_prediction_curves_save_err_table.py` and `plot_error_curves.py`


| Argument | Type | Default Value | Description |
| --- | --- | --- | --- |
| `-lp`, `--logs_dirpath` | str | None | The path to the log directory where `all_device_predicts.pkl` resides. |
| `-lp2`, `--logs_dirpath2` | str | None | An optional log path for overwriting NeighborFL predictions in the `-lp` file. A use case is running NeighborFL simulations with different remove strategies. |
| `-et`, `--error_type` | str | 'MSE' | The type of error metric to use to generate error tables, which can be MAE, MSE, RMSE, or MAPE. |
| `-r`, `--representative` | str | None | The device ID to be featured as the representative figure. If not specified, no representative figure will be generated. |
| `-row`, `--row` | int | 1 | The number of rows in the subplots. Columns will be automatically determined based on the number of available devices.|
| `-NFLConfig`, `--NFLConfig` | str | '' | The NeighborFL configuration name to display on plots and tables, such as R1, R3, L1, L3. |
| `-Oseqs`, `--output_sequences` | str | '1' | For predictions with an `output_length` greater than 1, specify the output sequences to plot.|

  
#### II. Arguments specifically for `plot_prediction_curves_save_err_table.py`

| Argument | Type | Default Value | Description |
| --- | --- | --- | --- |
| `-sr`, `--starting_comm_round` | int | None | The round number to start plotting. |
| `-pr`, `--plot_rounds` | int | 24 | The number of the communication rounds to plot. If `starting_comm_round` is not specified, it plots the most recent `plot_rounds` rounds in the simulations. |
| `-tr`, `--time_resolution` | int | 5 | The time resolution of the data, commonly 5 minutes. |
| `-xi`, `--xticks_interval` | int | 12 | Specify the number of points to plot within a communication round to ensure accurate round labeling on the x-axis for the representative plots, typically set to the same value as `-tau2p`. The program is hardcoded to create three xticks for subplots. |
| `-f`, `--feature` | str | 'Speed' | The feature name of the predictions used to set up the y-axis. |


#### III. Arguments specifically for `plot_error_curves.py`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-sr, --start_round` | int | 1 | Starting round to plot and calculate error. Default to the 1st round of the simulation. |
| `-er, --end_round` | int | None | Ending round to plot and calculate error. Default to to the last round of the simulation. |
| `-ei, --error_interval` | int | 24 | The unit is communication rounds. Used to smooth out error curves, especially in simulations with a large number of rounds to improve plot clarity. |

## Reproduce our Experimental Results

To replicate our experimental results and plots, you could refer to the provided [Colab file](%28https://colab.research.google.com/drive/1Rhy3YCM3MIXUBYr3YriESiuvcu203eyh%29.) **section 6 Reproduce our experiments and plots**, which contains direct code for recreating our plots, tables and simulation results.

We've refactored the code for improved clarity in variable names and some writing styles of the algorithms after obtaining those running logs. Therefore, to reproduce our experiments, please check out to the commit `41a6735` by doing this first

```shell
$ git clone https://github.com/hanglearning/NeighborFL.git
$ cd NeighborFL
$ git checkout 41a6735
```

### Recreate Our Plots and Error Tables

You can find the logs for the eight simulations used in our paper in [this folder](https://drive.google.com/drive/folders/1b7GjnddtKXcgfgjJ2dGoklxjC6osIlqS?usp=drive_link). 

#### Figure 3, Table III, IV, V

To recreate Figure 3, Table IV, and the MSE values of the three baselines and NeighborFL L1 under the Pretrain column in Table III, execute the following commands:

```shell
$ git checkout 41a6735
$ pip install xlsxwriter
$ python NeighborFL/plotting_code/plot_realtime_prediction_curves_and_save_err_table.py -lp "logs_on_paper/pretrain/R3" -lp2 "logs_on_paper/pretrain/L1" -sr 227 -pr 24 -r 400760_N -row 5
```

The resulting plots and Table IV will be saved under `logs_on_paper/pretrain/L1/plots/realtime_learning_curves_all_detectors`, and the Average Device MSE values of the last 24 rounds for the three baselines and NeighborFL L1 under the Pretrain setting in Table III will be displayed in the console.

To get all the values in Table III and Table V, execute the following command:

```shell
$ git checkout 41a6735
$ python NeighborFL/plotting_code/plot_realtime_prediction_curves_and_save_err_table.py -r 400760_N -lp <see_below> -sr <see_below> -pr <see_below>
```

Replace `-lp <see_below>` with the appropriate value based on the configuration you want to reproduce. Refer to the table below for the correct value:

| Avg MSE value of Config     | `-lp` value                                    |
| ----------------------------| ---------------------------------------------- |
| Pretrain NeighborFL L1      | `"logs_on_paper/pretrain/L1"`                 |
| Pretrain NeighborFL L3      | `"logs_on_paper/pretrain/L3"`                 |
| Pretrain NeighborFL R1      | `"logs_on_paper/pretrain/R1"`                 |
| Pretrain All Baselines + R3 | `"logs_on_paper/pretrain/R3"`                 |
| Non-Pretrain NeighborFL L1  | `"logs_on_paper/no_pretrain/L1"`             |
| Non-Pretrain NeighborFL L3  | `"logs_on_paper/no_pretrain/L3"`             |
| Non-Pretrain NeighborFL R1  | `"logs_on_paper/no_pretrain/R1"`             |
| Non-Pretrain All Baselines + R3 | `"logs_on_paper/no_pretrain/R3"`          |

For Table III, use `-sr 227` and `-pr 24`. For Table V, use `-sr 1` and `-pr 250`.


[comment]: <> (Please note that any time the `plot_realtime_prediction_curves_and_save_err_table.py` is executed, the associated prediction curves plots of the corresponding config will be overwritten. )


Please note the `-r` argument is required but can have any random device ID due to a bug in the code.

#### Figure 4

To recreate Figure 4, execute the following command:

```shell
$ git checkout 41a6735
$ python NeighborFL/plotting_code/plot_realtime_errors_and_save_err_interval_table.py -lp "NeighborFL/logs_on_paper/pretrain/R3" -lp2 "NeighborFL/logs_on_paper/pretrain/L1" -ei 24 -row 5 -r 400922_S
```

The resulting plots will be saved under `logs_on_paper/pretrain/L1/plots/realtime_errors_interval`. Note that there is also a csv file saved and some console output, but they are not used in the paper.

### Reproduce Our Experiments

#### Non-Pretrain Experiments

Execute the following command to run the non-pretrain experiments:

```shell
$ git checkout 7019541
$ python NeighborFL/main.py -dp <path/to/data> -lb "NeighborFL/logs" -c 250 -lm <lm> -ms 72 -r 1 -et MSE -ks <ks> -kr <kr> -si 2016
```

Replace `<path/to/data>` with the absolute path to the `data` folder where you've placed the downloaded [dataset](https://drive.google.com/drive/folders/1q_IPvLyWw7zJKSyQ89N8fGWVLAfRXaN8?usp=drive_link). Modify `-lm <lm> -ks <ks> -kr <kr>` according to the following table:

| Running Config     | `-lm`   | `-ks` | `-kr` |
| -------------------| --------| ------| ------|
| Central            | 100000  |       |       |
| NaiveFL            | 010000  |       |       |
| r-NaiveFL          | 000100  |       |       |
| NeighborFL L1      | 000001  |   3   |   1   |
| NeighborFL L3      | 000001  |   3   |   3   |
| NeighborFL R1      | 000001  |   1   |   1   |
| NeighborFL R3      | 000001  |   1   |   3   |

#### Pretrain Experiments

**1. Reproduce the Pretrained Models**

~~Execute the following command to generate our pretrained models:~~

```shell
$ # git checkout 7019541
$ # python NeighborFL/pretrain.py -dp <path/to/data> -ei 2016 -sp <path/to/save/models>
```

~~Replace `<path/to/data>` with the absolute path to the `data` folder where you've placed the downloaded [dataset](https://drive.google.com/drive/folders/1q_IPvLyWw7zJKSyQ89N8fGWVLAfRXaN8?usp=drive_link). For `-sp`, specify any desired path, such as `NeighborFL/pretrained_models`.~~

It appears that running the above command on Google Colab may not produce the exact same pretrained models as those created on our local machine. We are concerned that this command may not reproduce the same models in other environments either. To reproduce our experiments precisely, please download the [pretrained models](https://drive.google.com/drive/folders/1WDqxGWr8iHX1uxZmA8gRjPN5MI8ul7_c?usp=drive_link) we used for our experiments.

**2. Reproduce Experiments with the Pretrained Models**

Execute the following command to run experiments with the pretrained models:

```shell
$ git checkout 7019541
$ python NeighborFL/main.py -dp <path/to/data> -lb "NeighborFL/logs" -c 250 -lm <lm> -ms 72 -r 1 -et MSE -ks <ks> -kr <kr> -si 2016 -pp <path/to/pretrained_models>
```

The only difference between this command and the one in the [non-pretrain experiments](#non-pretrain-experiments) is the extra `-pp` argument. Replace `<path/to/pretrained_models>` with the path specified in [`-sp` in the previous step](#pretrain-experiments) or the path containing our provided pretrained models.

To run different configurations, refer to the `-lm <lm> -ks <ks> -kr <kr>` table provided in the [non-pretrain experiments section](#non-pretrain-experiments).

**3. Resuming a Simulation**

If you need to resume a simulation in commit `41a6735`, use the log path obtained from the console output when the simulation was first executed through `main.py`. Direct the `-rp` argument to this path as follows:

```shell
$ git checkout 7019541
$ python NeighborFL/main.py -rp <resume/log/path>
```

Please note that this old commit doesn't support argument override.
  

## Acknowledgments

  

(1) Our federated learning code is extended from the LSTM and GRU centralized learning methods found in this [xiaochus's TrafficFlowPrediction repo](https://github.com/xiaochus/TrafficFlowPrediction). Thank you!

  

(2) This document was composed for free at [StackEdit](https://stackedit.io/).

---
### Please raise any issues and concerns you found. Thank you!
