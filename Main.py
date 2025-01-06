import tkinter as tk, pandas as pd, numpy as np
import Python.Multistage_Stochastic as Multistage_Stochastic
import os, time
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import ttk
from itertools import product
from sklearn.model_selection import KFold

def open_directory( directory_path ):
    """
    Opens the specified directory in the default file explorer.
    
    Args:
        directory_path: The path to the directory to open.
    """
    try:
        os.startfile( directory_path )  # Windows
    except AttributeError:
        try:
            os.system( 'open "{}"'.format( directory_path ) )  # macOS
        except OSError:
            try:
                os.system( 'xdg-open "{}"'.format( directory_path ) )  # Linux
            except OSError:
                print( f"Could not open directory: {directory_path}" )

def loadFile_Historical():
    if loadFile_historical.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_historical.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_historical.delete( 0, 'end' )
        loadFile_historical.insert( 0, file_path )

def loadFile_Future():
    if loadFile_future.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_future.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_future.delete( 0, 'end' )
        loadFile_future.insert( 0, file_path )

def Save_Path():
    if input_save.get() is None:
        file_path = filedialog.askdirectory()
        input_save.insert( 0, file_path )
    else:
        file_path = filedialog.askdirectory()
        input_save.delete( 0, 'end' )
        input_save.insert( 0, file_path )

def loadFile_Intercept():
    if loadFile_intercept.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_intercept.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_intercept.delete( 0, 'end' )
        loadFile_intercept.insert( 0, file_path )

def loadFile_Uncertainty():
    if loadFile_uncertainty.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_uncertainty.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_uncertainty.delete( 0, 'end' )
        loadFile_uncertainty.insert( 0, file_path )

def loadFile_Production():
    if loadFile_production.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_production.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_production.delete( 0, 'end' )
        loadFile_production.insert( 0, file_path )

def loadFile_Holding():
    if loadFile_holding.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_holding.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_holding.delete( 0, 'end' )
        loadFile_holding.insert( 0, file_path )

def loadFile_Backorder():
    if loadFile_backorder.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_backorder.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_backorder.delete( 0, 'end' )
        loadFile_backorder.insert( 0, file_path )

def loadFile_Capacity():
    if loadFile_capacity.get() is None:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_capacity.insert( 0, file_path )
    else:
        file_path = filedialog.askopenfilename( filetypes = ( ( "csv files","*.csv" ),( "excel files","*.xlsx" ) ) )
        loadFile_capacity.delete( 0, 'end' )
        loadFile_capacity.insert( 0, file_path )

def Production_Cost( Num_of_Stages ):
    if production_radioVar.get() == 1:
        if input_production_single.get() is not None:
            try:
                c_p = float( input_production_single.get() )
                c_p = np.array( [c_p for _ in range( Num_of_Stages )] )
                return c_p
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid production cost!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please input the production cost!" )
    elif production_radioVar.get() == 2:
        if loadFile_production.get() != "":
            try:
                c_p = pd.read_csv( loadFile_production.get() ).to_numpy().reshape( -1 )
                return c_p
            except:
                tk.messagebox.showinfo( "Warning", "Please select a valid production cost file!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please select the production cost file!" )
    else:
        tk.messagebox.showinfo( "Warning", "Please select the production cost!" )

def Holding_Cost( Num_of_Stages ):
    if holding_radioVar.get() == 1:
        if input_holding_single.get() is not None:
            try:
                c_h = float( input_holding_single.get() )
                c_h = np.array( [c_h for _ in range( Num_of_Stages )] )
                return c_h
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid holding cost!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please input the holding cost!" )
    elif holding_radioVar.get() == 2:
        if loadFile_holding.get() != "":
            try:
                c_h = pd.read_csv( loadFile_holding.get() ).to_numpy().reshape( -1 )
                return c_h
            except:
                tk.messagebox.showinfo( "Warning", "Please select a valid holding cost file!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please select the holding cost file!" )
    else:
        tk.messagebox.showinfo( "Warning", "Please select the holding cost!" )

def Backorder_Cost( Num_of_Stages ):
    if backorder_radioVar.get() == 1:
        if input_backorder_single.get() is not None:
            try:
                c_b = float( input_backorder_single.get() )
                c_b = np.array( [c_b for _ in range( Num_of_Stages )] )
                return c_b
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid backorder cost!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please input the backorder cost!" )
    elif backorder_radioVar.get() == 2:
        if loadFile_backorder.get() != "":
            try:
                c_b = pd.read_csv( loadFile_backorder.get() ).to_numpy().reshape( -1 )
                return c_b
            except:
                tk.messagebox.showinfo( "Warning", "Please select a valid backorder cost file!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please select the backorder cost file!" )
    else:
        tk.messagebox.showinfo( "Warning", "Please select the backorder cost!" )

def Capacity( Num_of_Stages ):
    if capacity_radioVar.get() == 1:
        if input_capacity_single.get() is not None:
            try:
                x_bar = float( input_capacity_single.get() )
                x_bar = np.array( [x_bar for _ in range( Num_of_Stages )] )
                return x_bar
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid capacity!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please input the capacity!" )
    elif capacity_radioVar.get() == 2:
        if loadFile_capacity.get() != "":
            try:
                x_bar = pd.read_csv( loadFile_capacity.get() ).to_numpy().reshape( -1 )
                return x_bar
            except:
                tk.messagebox.showinfo( "Warning", "Please select a valid capacity file!" )
        else:
            tk.messagebox.showinfo( "Warning", "Please select the capacity file!" )
    else:
        tk.messagebox.showinfo( "Warning", "Please select the capacity!" )

def Check_Button_State_Hyperparameter():
    if cv_checkbutton.get() == 1:
        specific_checkbutton1.config( state = tk.DISABLED )
        kfold_5_radio.config( state = tk.NORMAL )
        kfold_5_radio.select()
        kfold_10_radio.config( state = tk.NORMAL )
        kfold_other_radio.config( state = tk.NORMAL )
        input_kfold.config( state = tk.NORMAL )
        input_epsilon_lb.config( state = tk.NORMAL )
        logspace_epsilon_N_radio.config( state = tk.NORMAL )
        other_epsilon_N_radio.config( state = tk.NORMAL )
        input_epsilon_other.config( state = tk.NORMAL )
        arange_omega_radio.config( state = tk.NORMAL )
        other_omega_radio.config( state = tk.NORMAL )
        input_omega_other.config( state = tk.NORMAL )
        arange_n_estimators_radio.config( state = tk.NORMAL )
        other_n_estimators_radio.config( state = tk.NORMAL )
        input_n_estimators_other.config( state = tk.NORMAL )
        arange_max_depth_radio.config( state = tk.NORMAL )
        other_max_depth_radio.config( state = tk.NORMAL )
        input_max_depth_other.config( state = tk.NORMAL )
        if input_epsilon_lb.get() == "":
            input_epsilon_lb.insert( 0, "0.1" )
        input_epsilon_ub.config( state = tk.NORMAL )
        if input_epsilon_ub.get() == "":
            input_epsilon_ub.insert( 0, "1" )
        input_epsilon_num.config( state = tk.NORMAL )
        if input_epsilon_num.get() == "":
            input_epsilon_num.insert( 0, "10" )
        input_omega_lb.config( state = tk.NORMAL )
        if input_omega_lb.get() == "":
            input_omega_lb.insert( 0, "0.1" )
        input_omega_ub.config( state = tk.NORMAL )
        if input_omega_ub.get() == "":
            input_omega_ub.insert( 0, "0.3" )
        input_omega_increment.config( state = tk.NORMAL )
        if input_omega_increment.get() == "":
            input_omega_increment.insert( 0, "0.1" )
        input_n_estimators_lb.config( state = tk.NORMAL )
        if input_n_estimators_lb.get() == "":
            input_n_estimators_lb.insert( 0, "50" )
        input_n_estimators_ub.config( state = tk.NORMAL )
        if input_n_estimators_ub.get() == "":
            input_n_estimators_ub.insert( 0, "150" )
        input_n_estimators_increment.config( state = tk.NORMAL )
        if input_n_estimators_increment.get() == "":
            input_n_estimators_increment.insert( 0, "50" )
        input_max_depth_lb.config( state = tk.NORMAL )
        if input_max_depth_lb.get() == "":
            input_max_depth_lb.insert( 0, "5" )
        input_max_depth_ub.config( state = tk.NORMAL )
        if input_max_depth_ub.get() == "":
            input_max_depth_ub.insert( 0, "15" )
        input_max_depth_increment.config( state = tk.NORMAL )
        if input_max_depth_increment.get() == "":
            input_max_depth_increment.insert( 0, "5" )
    
    elif cv_checkbutton.get() == 0:
        specific_checkbutton1.config( state = tk.NORMAL )
        kfold_5_radio.config( state = tk.DISABLED )
        kfold_10_radio.config( state = tk.DISABLED )
        kfold_other_radio.config( state = tk.DISABLED )
        input_kfold.config( state = tk.DISABLED )
        input_epsilon_lb.config( state = tk.DISABLED )
        input_epsilon_ub.config( state = tk.DISABLED )
        input_epsilon_num.config( state = tk.DISABLED )
        input_omega_lb.config( state = tk.DISABLED )
        input_omega_ub.config( state = tk.DISABLED )
        input_omega_increment.config( state = tk.DISABLED )
        input_n_estimators_lb.config( state = tk.DISABLED )
        input_n_estimators_ub.config( state = tk.DISABLED )
        input_n_estimators_increment.config( state = tk.DISABLED )
        input_max_depth_lb.config( state = tk.DISABLED )
        input_max_depth_ub.config( state = tk.DISABLED )
        input_max_depth_increment.config( state = tk.DISABLED )
        logspace_epsilon_N_radio.config( state = tk.DISABLED )
        other_epsilon_N_radio.config( state = tk.DISABLED )
        input_epsilon_other.config( state = tk.DISABLED )
        arange_omega_radio.config( state = tk.DISABLED )
        other_omega_radio.config( state = tk.DISABLED )
        input_omega_other.config( state = tk.DISABLED )
        arange_n_estimators_radio.config( state = tk.DISABLED )
        other_n_estimators_radio.config( state = tk.DISABLED )
        input_n_estimators_other.config( state = tk.DISABLED )
        arange_max_depth_radio.config( state = tk.DISABLED )
        other_max_depth_radio.config( state = tk.DISABLED )
        input_max_depth_other.config( state = tk.DISABLED )
    
    if specific_checkbutton.get() == 1:
        cv_checkbutton1.config( state = tk.DISABLED )
        input_epsilon_N.config( state = tk.NORMAL )
        input_omega.config( state = tk.NORMAL )
        input_n_estimators.config( state = tk.NORMAL )
        input_max_depth.config( state = tk.NORMAL )
    
    elif specific_checkbutton.get() == 0:
        cv_checkbutton1.config( state = tk.NORMAL )
        input_epsilon_N.config( state = tk.DISABLED )
        input_omega.config( state = tk.DISABLED )
        input_n_estimators.config( state = tk.DISABLED )
        input_max_depth.config( state = tk.DISABLED )

def Check_Button_State_Save():
    if save_checkbutton_True.get() == 1:
        save_checkbutton_False1.config( state = tk.DISABLED )
        input_save.config( state = tk.NORMAL )
        input_save_btn.config( state = tk.NORMAL )
    elif save_checkbutton_True.get() == 0:
        save_checkbutton_False1.config( state = tk.NORMAL )
        input_save.config( state = tk.DISABLED )
        input_save_btn.config( state = tk.DISABLED )
    
    if save_checkbutton_False.get() == 1:
        save_checkbutton_True1.config( state = tk.DISABLED )
        input_save.config( state = tk.DISABLED )
        input_save_btn.config( state = tk.DISABLED )
    elif save_checkbutton_False.get() == 0:
        save_checkbutton_True1.config( state = tk.NORMAL )
        input_save.config( state = tk.NORMAL )
        input_save_btn.config( state = tk.NORMAL )

def Check_Validity():
    if save_checkbutton_True.get() == 0 and save_checkbutton_False.get() == 0:
        tk.messagebox.showinfo( "Warning", "Please select whether to save the results or not!" )
        return False
    
    if save_checkbutton_True.get() == 1:
        if input_save.get() == "":
            tk.messagebox.showinfo( "Warning", "Please choose a folder to save the results!" )
            return False
    
    if function_radioVar.get() == 1:
        if loadFile_historical.get() == "":
            tk.messagebox.showinfo( "Warning", "Please select the historical demand data file!" )
            return False
        else:
            historical_data = pd.read_csv( loadFile_historical.get() ).to_numpy()
            Num_of_Stages = historical_data.shape[1]
            if Production_Cost( Num_of_Stages = Num_of_Stages ) is None:
                return False
            elif Holding_Cost( Num_of_Stages = Num_of_Stages ) is None:
                return False
            elif Backorder_Cost( Num_of_Stages = Num_of_Stages ) is None:
                return False
            elif Capacity( Num_of_Stages = Num_of_Stages ) is None:
                return False
            else:
                c_p = Production_Cost( Num_of_Stages = Num_of_Stages )
                c_h = Holding_Cost( Num_of_Stages = Num_of_Stages )
                c_b = Backorder_Cost( Num_of_Stages = Num_of_Stages )
                x_bar = Capacity( Num_of_Stages = Num_of_Stages )
                if len( c_p ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of production costs is not the same as the number of stages!" )
                    return False
                if len( c_h ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of holding costs is not the same as the number of stages!" )
                    return False
                if len( c_b ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of backorder costs is not the same as the number of stages!" )
                    return False
                if len( x_bar ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of capacities is not the same as the number of stages!" )
                    return False
    
    if function_radioVar.get() == 2:
        if loadFile_intercept.get() == "":
            tk.messagebox.showinfo( "Warning", "Please select the intercept data file!" )
            return False
        if loadFile_uncertainty.get() == "":
            tk.messagebox.showinfo( "Warning", "Please select the uncertainty data file!" )
            return False
        if loadFile_intercept.get() != "" and loadFile_uncertainty.get() != "":
            # Check if the number of stages are matched
            intercept = pd.read_csv( loadFile_intercept.get() ).to_numpy().reshape( -1 )
            uncertainty = pd.read_csv( loadFile_uncertainty.get() ).to_numpy()
            if len( intercept ) != uncertainty.shape[0]:
                tk.messagebox.showinfo( "Warning", "The number of stages for intercept and uncertainty data is not the same!" )
                return False
            
            Num_of_Stages = len( intercept )
            if Capacity( Num_of_Stages = Num_of_Stages ) is None:
                return False
            else:
                x_bar = Capacity( Num_of_Stages = Num_of_Stages )
                if len( x_bar ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of capacities is not the same as the number of stages!" )
                    return False
    
    if function_radioVar.get() == 3:
        if loadFile_historical.get() == "":
            tk.messagebox.showinfo( "Warning", "Please select the historical demand data file!" )
            return False
        if loadFile_future.get() == "":
            tk.messagebox.showinfo( "Warning", "Please select the future demand data file!" )
            return False
        if loadFile_historical.get() != "" and loadFile_future.get() != "" :
            # Check if the number of stages for each data is the same
            historical_data = pd.read_csv( loadFile_historical.get() ).to_numpy()
            future_data = pd.read_csv( loadFile_future.get() ).to_numpy()
            Num_of_Stages = historical_data.shape[1]
            if Production_Cost( Num_of_Stages = Num_of_Stages ) is None:
                return False
            elif Holding_Cost( Num_of_Stages = Num_of_Stages ) is None:
                return False
            elif Backorder_Cost( Num_of_Stages = Num_of_Stages ) is None:
                return False
            elif Capacity( Num_of_Stages = Num_of_Stages ) is None:
                return False
            else:
                c_p = Production_Cost( Num_of_Stages = Num_of_Stages )
                c_h = Holding_Cost( Num_of_Stages = Num_of_Stages )
                c_b = Backorder_Cost( Num_of_Stages = Num_of_Stages )
                x_bar = Capacity( Num_of_Stages = Num_of_Stages )
                if len( c_p ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of production costs is not the same as the number of stages!" )
                    return False
                if len( c_h ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of holding costs is not the same as the number of stages!" )
                    return False
                if len( c_b ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of backorder costs is not the same as the number of stages!" )
                    return False
                if len( x_bar ) != Num_of_Stages:
                    tk.messagebox.showinfo( "Warning", "The number of capacities is not the same as the number of stages!" )
                    return False
    
    # Hyperparameters part
    if cv_checkbutton.get() == 1: # Cross-Validation
        if kfold_radioVar.get() == 0: # Other
            if input_kfold.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the number of folds!" )
                return False
            else:
                try:
                    kfold = int( input_kfold.get() )
                    if kfold <= 0:
                        tk.messagebox.showinfo( "Warning", "Please input a valid number of folds!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid number of folds!" )
                    return False
        
        if epsilon_N_radioVar.get() == 0: # Logspace
            if input_epsilon_lb.get() == "" or input_epsilon_ub.get() == "" or input_epsilon_num.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the epsilon range!" )
                return False
            else:
                try:
                    epsilon_lb = float( input_epsilon_lb.get() )
                    epsilon_ub = float( input_epsilon_ub.get() )
                    epsilon_num = int( input_epsilon_num.get() )
                    if epsilon_num <= 1:
                        tk.messagebox.showinfo( "Warning", "The number of epsilon_N must at least be 2!" )
                        return False
                    if epsilon_lb <= 0 or epsilon_ub <= 0:
                        tk.messagebox.showinfo( "Warning", "Please input a valid epsilon range!" )
                        return False
                    if epsilon_lb >= epsilon_ub:
                        tk.messagebox.showinfo( "Warning", "The lower bound must be smaller than the upper bound!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid epsilon range!" )
                    return False
        else: # Other
            if input_epsilon_other.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the epsilon range!" )
                return False
            else:
                try:
                    epsilon_space = input_epsilon_other.get().split( "," )
                    epsilon_space = [float( epsilon ) for epsilon in epsilon_space]
                    if len( epsilon_space ) < 1:
                        tk.messagebox.showinfo( "Warning", "The number of epsilon_N must at least be 1!" )
                        return False
                    if min( epsilon_space ) < 0:
                        tk.messagebox.showinfo( "Warning", "The values must be non-negative!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid epsilon range! (Use comma to separate the values)" )
                    return False
        
        if omega_radioVar.get() == 0: # arange
            if input_omega_lb.get() == "" or input_omega_ub.get() == "" or input_omega_increment.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the omega range!" )
                return False
            else:
                try:
                    omega_lb = float( input_omega_lb.get() )
                    omega_ub = float( input_omega_ub.get() )
                    omega_increment = float( input_omega_increment.get() )
                    if omega_lb <= 0 or omega_ub <= 0 or omega_increment <= 0:
                        tk.messagebox.showinfo( "Warning", "Please input a valid omega range!" )
                        return False
                    if omega_ub >= 1:
                        tk.messagebox.showinfo( "Warning", "The upper bound must be smaller than 1!" )
                        return False
                    if omega_lb >= omega_ub:
                        tk.messagebox.showinfo( "Warning", "The lower bound must be smaller than the upper bound!" )
                        return False
                    if ( omega_ub - omega_lb ) < omega_increment:
                        tk.messagebox.showinfo( "Warning", "The increment must be smaller than the difference between the upper bound and the lower bound!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid omega range!" )
                    return False
        else: # Other
            if input_omega_other.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the omega range!" )
                return False
            else:
                try:
                    omega_space = input_omega_other.get().split( "," )
                    omega_space = [float( omega ) for omega in omega_space]
                    if len( omega_space ) < 1:
                        tk.messagebox.showinfo( "Warning", "The number of omega must at least be 1!" )
                        return False
                    if min( omega_space ) < 0 or max( omega_space ) > 1:
                        tk.messagebox.showinfo( "Warning", "The omega values must be between 0 and 1!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid omega range! (Use comma to separate the values)" )
                    return False
        
        if n_estimators_radioVar.get() == 0: # arange
            if input_n_estimators_lb.get() == "" or input_n_estimators_ub.get() == "" or input_n_estimators_increment.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the n_estimators range!" )
                return False
            else:
                try:
                    n_estimators_lb = int( input_n_estimators_lb.get() )
                    n_estimators_ub = int( input_n_estimators_ub.get() )
                    n_estimators_increment = int( input_n_estimators_increment.get() )
                    if n_estimators_lb <= 0 or n_estimators_ub <= 0 or n_estimators_increment <= 0:
                        tk.messagebox.showinfo( "Warning", "Please input a valid n_estimators range!" )
                        return False
                    if n_estimators_lb >= n_estimators_ub:
                        tk.messagebox.showinfo( "Warning", "The lower bound must be smaller than the upper bound!" )
                        return False
                    if ( n_estimators_ub - n_estimators_lb ) < n_estimators_increment:
                        tk.messagebox.showinfo( "Warning", "The increment must be smaller than the difference between the upper bound and the lower bound!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid n_estimators range! (Integer)" )
                    return False
        else: # Other
            if input_n_estimators_other.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the n_estimators range!" )
                return False
            else:
                try:
                    n_estimators_space = input_n_estimators_other.get().split( "," )
                    n_estimators_space = [int( n_estimators ) for n_estimators in n_estimators_space]
                    if len( n_estimators_space ) < 1:
                        tk.messagebox.showinfo( "Warning", "The number of n_estimators must at least be 1!" )
                        return False
                    if min( n_estimators_space ) < 0:
                        tk.messagebox.showinfo( "Warning", "The n_estimators values must be non-negative!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid n_estimators range! (Use comma to separate the values)" )
                    return False
        
        if max_depth_radioVar.get() == 0: # arange
            if input_max_depth_lb.get() == "" or input_max_depth_ub.get() == "" or input_max_depth_increment.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the max_depth range!" )
                return False
            else:
                try:
                    max_depth_lb = int( input_max_depth_lb.get() )
                    max_depth_ub = int( input_max_depth_ub.get() )
                    max_depth_increment = int( input_max_depth_increment.get() )
                    if max_depth_lb <= 0 or max_depth_ub <= 0 or max_depth_increment <= 0:
                        tk.messagebox.showinfo( "Warning", "Please input a valid max_depth range!" )
                        return False
                    if max_depth_lb >= max_depth_ub:
                        tk.messagebox.showinfo( "Warning", "The lower bound must be smaller than the upper bound!" )
                        return False
                    if ( max_depth_ub - max_depth_lb ) < max_depth_increment:
                        tk.messagebox.showinfo( "Warning", "The increment must be smaller than the difference between the upper bound and the lower bound!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid max_depth range! (Integer)" )
                    return False
        else: # Other
            if input_max_depth_other.get() == "":
                tk.messagebox.showinfo( "Warning", "Please input the max_depth range!" )
                return False
            else:
                try:
                    max_depth_space = input_max_depth_other.get().split( "," )
                    max_depth_space = [int( max_depth ) for max_depth in max_depth_space]
                    if len( max_depth_space ) < 1:
                        tk.messagebox.showinfo( "Warning", "The number of max_depth must at least be 1!" )
                        return False
                    if min( max_depth_space ) < 0:
                        tk.messagebox.showinfo( "Warning", "The max_depth values must be non-negative!" )
                        return False
                except:
                    tk.messagebox.showinfo( "Warning", "Please input a valid max_depth range! (Use comma to separate the values)" )
                    return False
    
    if specific_checkbutton.get() == 1: # Specific
        if input_epsilon_N.get() == "":
            tk.messagebox.showinfo( "Warning", "Please input the epsilon_N!" )
            return False
        else:
            try:
                epsilon_N = float( input_epsilon_N.get() )
                if epsilon_N <= 0:
                    tk.messagebox.showinfo( "Warning", "Please input a valid epsilon_N!" )
                    return False
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid epsilon_N!" )
                return False
        
        if input_omega.get() == "":
            tk.messagebox.showinfo( "Warning", "Please input the omega!" )
            return False
        else:
            try:
                omega = float( input_omega.get() )
                if omega <= 0 or omega >= 1:
                    tk.messagebox.showinfo( "Warning", "Please input a valid omega!" )
                    return False
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid omega!" )
                return False
        
        if input_n_estimators.get() == "":
            tk.messagebox.showinfo( "Warning", "Please input the n_estimators!" )
            return False
        else:
            try:
                n_estimators = int( input_n_estimators.get() )
                if n_estimators <= 0:
                    tk.messagebox.showinfo( "Warning", "Please input a valid n_estimators!" )
                    return False
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid n_estimators!" )
                return False
        
        if input_max_depth.get() == "":
            tk.messagebox.showinfo( "Warning", "Please input the max_depth!" )
            return False
        else:
            try:
                max_depth = int( input_max_depth.get() )
                if max_depth <= 0:
                    tk.messagebox.showinfo( "Warning", "Please input a valid max_depth!" )
                    return False
            except:
                tk.messagebox.showinfo( "Warning", "Please input a valid max_depth!" )
                return False
    
    if function_radioVar.get() == 1 or function_radioVar.get() == 3:
        if cv_checkbutton.get() == 0 and specific_checkbutton.get() == 0:
            tk.messagebox.showinfo( "Warning", "Please select the type of hyperparameters!" )
            return False
    return True

def Check_State_Function():
    if function_radioVar.get() == 1: # Train new model
        loadFile_historical.config( state = tk.NORMAL )
        loadFile_historical_btn.config( state = tk.NORMAL )
        production_single_radio.config( state = tk.NORMAL )
        input_production_single.config( state = tk.NORMAL )
        production_multi_radio.config( state = tk.NORMAL )
        loadFile_production.config( state = tk.NORMAL )
        loadFile_production_btn.config( state = tk.NORMAL )
        holding_single_radio.config( state = tk.NORMAL )
        input_holding_single.config( state = tk.NORMAL )
        holding_multi_radio.config( state = tk.NORMAL )
        loadFile_holding.config( state = tk.NORMAL )
        loadFile_holding_btn.config( state = tk.NORMAL )
        backorder_single_radio.config( state = tk.NORMAL )
        input_backorder_single.config( state = tk.NORMAL )
        backorder_multi_radio.config( state = tk.NORMAL )
        loadFile_backorder.config( state = tk.NORMAL )
        loadFile_backorder_btn.config( state = tk.NORMAL )
        capacity_single_radio.config( state = tk.NORMAL )
        input_capacity_single.config( state = tk.NORMAL )
        capacity_multi_radio.config( state = tk.NORMAL )
        loadFile_capacity.config( state = tk.NORMAL )
        loadFile_capacity_btn.config( state = tk.NORMAL )
        cv_checkbutton1.config( state = tk.NORMAL )
        specific_checkbutton1.config( state = tk.NORMAL )
        
        loadFile_future.config( state = tk.DISABLED )
        loadFile_future_btn.config( state = tk.DISABLED )
        loadFile_intercept.config( state = tk.DISABLED )
        loadFile_intercept_btn.config( state = tk.DISABLED )
        loadFile_uncertainty.config( state = tk.DISABLED )
        loadFile_uncertainty_btn.config( state = tk.DISABLED )
    
    elif function_radioVar.get() == 2: # Load in pre-trained model
        loadFile_historical.config( state = tk.DISABLED )
        loadFile_historical_btn.config( state = tk.DISABLED )
        production_single_radio.config( state = tk.DISABLED )
        input_production_single.config( state = tk.DISABLED )
        production_multi_radio.config( state = tk.DISABLED )
        loadFile_production.config( state = tk.DISABLED )
        loadFile_production_btn.config( state = tk.DISABLED )
        holding_single_radio.config( state = tk.DISABLED )
        input_holding_single.config( state = tk.DISABLED )
        holding_multi_radio.config( state = tk.DISABLED )
        loadFile_holding.config( state = tk.DISABLED )
        loadFile_holding_btn.config( state = tk.DISABLED )
        backorder_single_radio.config( state = tk.DISABLED )
        input_backorder_single.config( state = tk.DISABLED )
        backorder_multi_radio.config( state = tk.DISABLED )
        loadFile_backorder.config( state = tk.DISABLED )
        loadFile_backorder_btn.config( state = tk.DISABLED )
        cv_checkbutton1.config( state = tk.DISABLED )
        specific_checkbutton1.config( state = tk.DISABLED )
        
        loadFile_future.config( state = tk.NORMAL )
        loadFile_future_btn.config( state = tk.NORMAL )
        loadFile_intercept.config( state = tk.NORMAL )
        loadFile_intercept_btn.config( state = tk.NORMAL )
        loadFile_uncertainty.config( state = tk.NORMAL )
        loadFile_uncertainty_btn.config( state = tk.NORMAL )
        capacity_single_radio.config( state = tk.NORMAL )
        input_capacity_single.config( state = tk.NORMAL )
        capacity_multi_radio.config( state = tk.NORMAL )
        loadFile_capacity.config( state = tk.NORMAL )
        loadFile_capacity_btn.config( state = tk.NORMAL )
    
    elif function_radioVar.get() == 3: # Train and make future decisions
        loadFile_historical.config( state = tk.NORMAL )
        loadFile_historical_btn.config( state = tk.NORMAL )
        loadFile_future.config( state = tk.NORMAL )
        loadFile_future_btn.config( state = tk.NORMAL )
        production_single_radio.config( state = tk.NORMAL )
        input_production_single.config( state = tk.NORMAL )
        production_multi_radio.config( state = tk.NORMAL )
        loadFile_production.config( state = tk.NORMAL )
        loadFile_production_btn.config( state = tk.NORMAL )
        holding_single_radio.config( state = tk.NORMAL )
        input_holding_single.config( state = tk.NORMAL )
        holding_multi_radio.config( state = tk.NORMAL )
        loadFile_holding.config( state = tk.NORMAL )
        loadFile_holding_btn.config( state = tk.NORMAL )
        backorder_single_radio.config( state = tk.NORMAL )
        input_backorder_single.config( state = tk.NORMAL )
        backorder_multi_radio.config( state = tk.NORMAL )
        loadFile_backorder.config( state = tk.NORMAL )
        loadFile_backorder_btn.config( state = tk.NORMAL )
        capacity_single_radio.config( state = tk.NORMAL )
        input_capacity_single.config( state = tk.NORMAL )
        capacity_multi_radio.config( state = tk.NORMAL )
        loadFile_capacity.config( state = tk.NORMAL )
        loadFile_capacity_btn.config( state = tk.NORMAL )
        cv_checkbutton1.config( state = tk.NORMAL )
        specific_checkbutton1.config( state = tk.NORMAL )
        
        loadFile_intercept.config( state = tk.DISABLED )
        loadFile_intercept_btn.config( state = tk.DISABLED )
        loadFile_uncertainty.config( state = tk.DISABLED )
        loadFile_uncertainty_btn.config( state = tk.DISABLED )

def Run():
    validity = Check_Validity()
    if validity:
        if function_radioVar.get() == 1: # Train new model
            # Load in user input
            historical_data = pd.read_csv( loadFile_historical.get() ).to_numpy()
            Num_of_Stages = historical_data.shape[1]
            c = Production_Cost( Num_of_Stages = Num_of_Stages )
            h = Holding_Cost( Num_of_Stages = Num_of_Stages )
            b = Backorder_Cost( Num_of_Stages = Num_of_Stages )
            x_bar = Capacity( Num_of_Stages = Num_of_Stages )
            if cv_checkbutton.get() == 1: # Cross-Validation
                if epsilon_N_radioVar.get() == 0: # Logspace
                    epsilon_lb, epsilon_ub, epsilon_num = float( input_epsilon_lb.get() ), float( input_epsilon_ub.get() ), int( input_epsilon_num.get() )
                    epsilon_space = np.logspace( start = np.log10( epsilon_lb ), stop = np.log10( epsilon_ub ), num = epsilon_num, endpoint = True ).tolist()
                else: # Other
                    epsilon_space = input_epsilon_other.get().split( "," )
                    epsilon_space = [float( epsilon ) for epsilon in epsilon_space]
                
                if omega_radioVar.get() == 0: # arange
                    omega_lb, omega_ub, omega_increment = float( input_omega_lb.get() ), float( input_omega_ub.get() ), float( input_omega_increment.get() )
                    omega_space = np.arange( start = omega_lb, stop = omega_ub, step = omega_increment ).tolist() + [omega_ub]
                else:
                    omega_space = input_omega_other.get().split( "," )
                    omega_space = [float( omega ) for omega in omega_space]
                
                if n_estimators_radioVar.get() == 0: # arange
                    n_estimators_lb, n_estimators_ub, n_estimators_increment = int( input_n_estimators_lb.get() ), int( input_n_estimators_ub.get() ), int( input_n_estimators_increment.get() )
                    n_estimators_space = np.arange( start = n_estimators_lb, stop = n_estimators_ub, step = n_estimators_increment ).tolist() + [n_estimators_ub]
                else:
                    n_estimators_space = input_n_estimators_other.get().split( "," )
                    n_estimators_space = [int( n_estimators ) for n_estimators in n_estimators_space]
                
                if max_depth_radioVar.get() == 0: # arange
                    max_depth_lb, max_depth_ub, max_depth_increment = int( input_max_depth_lb.get() ), int( input_max_depth_ub.get() ), int( input_max_depth_increment.get() )
                    max_depth_space = np.arange( start = max_depth_lb, stop = max_depth_ub, step = max_depth_increment ).tolist() + [max_depth_ub]
                else:
                    max_depth_space = input_max_depth_other.get().split( "," )
                    max_depth_space = [int( max_depth ) for max_depth in max_depth_space]
                
                # Set the maximum of the progress bar
                progress_bar["maximum"] = len( epsilon_space ) * len( omega_space ) * len( n_estimators_space ) * len( max_depth_space )
                
                if kfold_radioVar.get() == 5: # 5-fold CV
                    x_opt, X_opt = Cross_Validation( historical_data, c, h, b, x_bar, 5, epsilon_space, omega_space, n_estimators_space, max_depth_space )
                elif kfold_radioVar.get() == 10: # 10-fold CV
                    x_opt, X_opt = Cross_Validation( historical_data, c, h, b, x_bar, 10, epsilon_space, omega_space, n_estimators_space, max_depth_space )
                else:
                    x_opt, X_opt = Cross_Validation( historical_data, c, h, b, x_bar, int( input_kfold.get() ), epsilon_space, omega_space, n_estimators_space, max_depth_space )
            else: # Specific
                epsilon_N = float( input_epsilon_N.get() )
                omega = float( input_omega.get() )
                n_estimators = int( input_n_estimators.get() )
                max_depth = int( input_max_depth.get() )
                x_opt, X_opt = Multistage_Stochastic.Specific( historical_data, c, h, b, x_bar, epsilon_N, omega, n_estimators, max_depth )
            
            # Save results
            if save_checkbutton_True.get() == 1:
                save_folder = input_save.get()
                if not os.path.exists( save_folder ):
                    os.makedirs( save_folder, exist_ok = True )
                
                intercept_df = pd.DataFrame( x_opt )
                intercept_df.to_csv( os.path.join( save_folder, "intercept.csv" ), index = False )
                uncertainty_df = pd.DataFrame( X_opt )
                uncertainty_df.to_csv( os.path.join( save_folder, "uncertainty.csv" ), index = False )
                
                tk.messagebox.showinfo( "Information", "The pre-trained model has been saved in the folder: " + save_folder )
                
                open_directory( save_folder )
        
        elif function_radioVar.get() == 2: # Load in pre-trained model
            # Load in user input
            future_data = pd.read_csv( loadFile_future.get() ).to_numpy()
            intercept = pd.read_csv( loadFile_intercept.get() ).to_numpy().reshape( -1 )
            uncertainty = pd.read_csv( loadFile_uncertainty.get() ).to_numpy()
            Num_of_Stages = len( intercept )
            x_bar = Capacity( Num_of_Stages = Num_of_Stages )
            
            # Make decisions
            if future_data.shape[1] == Num_of_Stages:
                decision_x = Multistage_Stochastic.Generate_Decision( N = future_data.shape[0], T = Num_of_Stages, demand = future_data, x_opt = intercept, X_opt = uncertainty, x_bar = x_bar )
            else:
                decision_x = Multistage_Stochastic.Next_Stage_Decision( future_data, intercept, uncertainty, x_bar )
            
            if save_checkbutton_True.get() == 1:
                save_folder = input_save.get()
                if not os.path.exists( save_folder ):
                    os.makedirs( save_folder, exist_ok = True )
                
                if len( decision_x.shape ) == 1:
                    future_df = pd.DataFrame( decision_x, columns = [f'Stage {t + 1}' for t in range( len( decision_x ) )] )
                else:
                    future_df = pd.DataFrame( decision_x, columns = [f'Stage {t + 1}' for t in range( decision_x.shape[1] )] )
                future_df.to_csv( os.path.join( save_folder, "future.csv" ), index = False )
                
                tk.messagebox.showinfo( "Information", "The future decisions have been saved in the folder: " + save_folder )
                
                open_directory( save_folder )
        
        elif function_radioVar.get() == 3: # Train and make future decisions
            # Load in user input
            historical_data = pd.read_csv( loadFile_historical.get() ).to_numpy()
            future_data = pd.read_csv( loadFile_future.get() ).to_numpy()
            Num_of_Stages = historical_data.shape[1]
            c = Production_Cost( Num_of_Stages = Num_of_Stages )
            h = Holding_Cost( Num_of_Stages = Num_of_Stages )
            b = Backorder_Cost( Num_of_Stages = Num_of_Stages )
            x_bar = Capacity( Num_of_Stages = Num_of_Stages )
            if cv_checkbutton.get() == 1: # Cross-Validation
                if epsilon_N_radioVar.get() == 0: # Logspace
                    epsilon_lb, epsilon_ub, epsilon_num = float( input_epsilon_lb.get() ), float( input_epsilon_ub.get() ), int( input_epsilon_num.get() )
                    epsilon_space = np.logspace( start = np.log10( epsilon_lb ), stop = np.log10( epsilon_ub ), num = epsilon_num, endpoint = True ).tolist()
                else: # Other
                    epsilon_space = input_epsilon_other.get().split( "," )
                    epsilon_space = [float( epsilon ) for epsilon in epsilon_space]
                
                if omega_radioVar.get() == 0: # arange
                    omega_lb, omega_ub, omega_increment = float( input_omega_lb.get() ), float( input_omega_ub.get() ), float( input_omega_increment.get() )
                    omega_space = np.arange( start = omega_lb, stop = omega_ub, step = omega_increment ).tolist() + [omega_ub]
                else:
                    omega_space = input_omega_other.get().split( "," )
                    omega_space = [float( omega ) for omega in omega_space]
                
                if n_estimators_radioVar.get() == 0: # arange
                    n_estimators_lb, n_estimators_ub, n_estimators_increment = int( input_n_estimators_lb.get() ), int( input_n_estimators_ub.get() ), int( input_n_estimators_increment.get() )
                    n_estimators_space = np.arange( start = n_estimators_lb, stop = n_estimators_ub, step = n_estimators_increment ).tolist() + [n_estimators_ub]
                else:
                    n_estimators_space = input_n_estimators_other.get().split( "," )
                    n_estimators_space = [int( n_estimators ) for n_estimators in n_estimators_space]
                
                if max_depth_radioVar.get() == 0: # arange
                    max_depth_lb, max_depth_ub, max_depth_increment = int( input_max_depth_lb.get() ), int( input_max_depth_ub.get() ), int( input_max_depth_increment.get() )
                    max_depth_space = np.arange( start = max_depth_lb, stop = max_depth_ub, step = max_depth_increment ).tolist() + [max_depth_ub]
                else:
                    max_depth_space = input_max_depth_other.get().split( "," )
                    max_depth_space = [int( max_depth ) for max_depth in max_depth_space]
                
                # Set the maximum of the progress bar
                progress_bar["maximum"] = len( epsilon_space ) * len( omega_space ) * len( n_estimators_space ) * len( max_depth_space )
                
                if kfold_radioVar.get() == 5: # 5-fold CV
                    x_opt, X_opt = Cross_Validation( historical_data, c, h, b, x_bar, 5, epsilon_space, omega_space, n_estimators_space, max_depth_space )
                elif kfold_radioVar.get() == 10: # 10-fold CV
                    x_opt, X_opt = Cross_Validation( historical_data, c, h, b, x_bar, 10, epsilon_space, omega_space, n_estimators_space, max_depth_space )
                else:
                    x_opt, X_opt = Cross_Validation( historical_data, c, h, b, x_bar, int( input_kfold.get() ), epsilon_space, omega_space, n_estimators_space, max_depth_space )
            else: # Specific
                epsilon_N = float( input_epsilon_N.get() )
                omega = float( input_omega.get() )
                n_estimators = int( input_n_estimators.get() )
                max_depth = int( input_max_depth.get() )
                x_opt, X_opt = Multistage_Stochastic.Specific( historical_data, c, h, b, x_bar, epsilon_N, omega, n_estimators, max_depth )
            
            # Make decisions
            if future_data.shape[1] == Num_of_Stages:
                decision_x = Multistage_Stochastic.Generate_Decision( N = future_data.shape[0], T = Num_of_Stages, demand = future_data, x_opt = x_opt, X_opt = X_opt, x_bar = x_bar )
            else:
                decision_x = Multistage_Stochastic.Next_Stage_Decision( future_data, x_opt, X_opt, x_bar )
            
            if save_checkbutton_True.get() == 1:
                save_folder = input_save.get()
                if not os.path.exists( save_folder ):
                    os.makedirs( save_folder, exist_ok = True )
                
                if len( decision_x.shape ) == 1:
                    future_df = pd.DataFrame( decision_x, columns = [f'Stage {t + 1}' for t in range( len( decision_x ) )] )
                else:
                    future_df = pd.DataFrame( decision_x, columns = [f'Stage {t + 1}' for t in range( decision_x.shape[1] )] )
                future_df.to_csv( os.path.join( save_folder, "future.csv" ), index = False )
                
                intercept_df = pd.DataFrame( x_opt )
                intercept_df.to_csv( os.path.join( save_folder, "intercept.csv" ), index = False )
                uncertainty_df = pd.DataFrame( X_opt )
                uncertainty_df.to_csv( os.path.join( save_folder, "uncertainty.csv" ), index = False )
                
                tk.messagebox.showinfo( "Information", "The results have been saved in the folder: " + save_folder )
                
                open_directory( save_folder )
        
        window.destroy()

def Cross_Validation( historical, c, h, b, x_bar, n_splits, epsilon_space, omega_space, n_estimators_space, max_depth_space, random_seed = 320426 ):
    tk.messagebox.showinfo( "Reminder", "Please don't click any buttons during the cross validation!" )
    # Using K-fold cross validation find the optimal hyperparameters
    best_dict = {
        'valid_cost': np.inf,
        'valid_service_level': np.inf,
        'epsilon_N': None,
        'omega': None,
        'n_estimators': None,
        'max_depth': None
    }
    kf = KFold( n_splits = n_splits, shuffle = True, random_state = random_seed )
    combination_idx = 1
    for epsilon_N, omega, n_estimators, max_depth in product( epsilon_space, omega_space, n_estimators_space, max_depth_space ):
        valid_cost = np.zeros( n_splits )
        valid_service_level = np.zeros( n_splits )
        for i, ( train_index, valid_index ) in enumerate( kf.split( historical ) ):
            train_historical = historical[train_index, :]
            valid_historical = historical[valid_index, :]
            
            x_opt, X_opt = Multistage_Stochastic.Specific( train_historical, c, h, b, x_bar, epsilon_N, omega, n_estimators, max_depth, random_seed = random_seed )
            valid_decision = Multistage_Stochastic.Generate_Decision( valid_historical.shape[0], valid_historical.shape[1], valid_historical, x_opt, X_opt, x_bar )
            valid_cost[i], valid_service_level[i] = Multistage_Stochastic.Compute_Cost( valid_historical.shape[0], valid_historical.shape[1], valid_historical, valid_decision, c, h, b )
        
        average_valid_cost = np.mean( valid_cost )
        average_valid_service_level = np.mean( valid_service_level )
        # print( f'epsilon_N = {epsilon_N:.4f}, omega = {omega:.2f}, n_estimators = {n_estimators}, max_depth = {max_depth}, valid_cost = {average_valid_cost:.4f}, valid_service_level = {average_valid_service_level:.2f}' )
        text_area.insert( tk.END, f"epsilon_N = {epsilon_N:.4f}, omega = {omega:.2f}, n_estimators = {n_estimators}, max_depth = {max_depth}, valid_cost = {average_valid_cost:.4f}, valid_service_level = {average_valid_service_level:.2f}\n" )
        text_area.see( tk.END )
        if average_valid_cost < best_dict['valid_cost']:
            best_dict['valid_cost'] = average_valid_cost
            best_dict['valid_service_level'] = average_valid_service_level
            best_dict['epsilon_N'] = epsilon_N
            best_dict['omega'] = omega
            best_dict['n_estimators'] = n_estimators
            best_dict['max_depth'] = max_depth
            # print( 'New best hyperparameters found!' )
            text_area.insert( tk.END, "New best hyperparameters found!\n" )
            text_area.see( tk.END )
        # print( '-.' * 50 )
        text_area.insert( tk.END, '-.' * 60 + '\n' )
        text_area.see( tk.END )
        progress_var.set( combination_idx )
        combination_idx += 1
        window.update_idletasks()
    
    # Fit the model using the best hyperparameters
    x_opt, X_opt = Multistage_Stochastic.Specific( historical, c, h, b, x_bar, best_dict['epsilon_N'], best_dict['omega'], best_dict['n_estimators'], best_dict['max_depth'], random_seed = random_seed )
    
    return x_opt, X_opt

# Create the main window
window = tk.Tk()
window.title( "Robust Inventory Management" )
window.geometry( "900x985+20+5" )
# window.attributes( '-fullscreen', True )
window.resizable( False, False )

row_idx = 0

#* Checkbutton to select the function: 1. Train new model; 2. Load in pre-trained model. 3.Train and make future decisions.
# Add widget
lb_function = tk.Label( window, text = "Function:", height = 1, font = ( "Arial", 12 ) )
function_radioVar = tk.IntVar()
train_radio = tk.Radiobutton( window, text = "Train new model", variable = function_radioVar, value = 1, command = Check_State_Function )
train_radio.select()
load_radio = tk.Radiobutton( window, text = "Make future decisions", variable = function_radioVar, value = 2, command = Check_State_Function )
train_decide_radio = tk.Radiobutton( window, text = "Train and make future decisions", variable = function_radioVar, value = 3, command = Check_State_Function )
# Locate widget
lb_function.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
train_radio.grid( row = row_idx, column = 1, columnspan = 2, padx = 5, pady = 5, sticky = tk.W )
load_radio.grid( row = row_idx, column = 3, columnspan = 3, padx = 5, pady = 5, sticky = tk.W )
train_decide_radio.grid( row = row_idx, column = 6, columnspan = 2, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Folder used to save the results
# Add widget
lb_save = tk.Label( window, text = "Save Results:", height = 1, font = ( "Arial", 12 ) )
save_checkbutton_True = tk.IntVar()
save_checkbutton_False = tk.IntVar()
save_checkbutton_True1 = tk.Checkbutton( window, text = "True", variable = save_checkbutton_True, command = Check_Button_State_Save )
save_checkbutton_False1 = tk.Checkbutton( window, text = "False", variable = save_checkbutton_False, command = Check_Button_State_Save )
lb_folder = tk.Label( window, text = "Folder:", height = 1, font = ( "Arial", 12 ) )
input_save = tk.Entry( window, width = 41 )
input_save_btn = tk.Button( window, text = "...", height = 1, command = Save_Path )
# Locate widget
lb_save.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
save_checkbutton_True1.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
save_checkbutton_False1.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
lb_folder.grid( row = row_idx, column = 3, padx = 5, pady = 5, sticky = tk.W )
input_save.grid( row = row_idx, column = 4, columnspan = 3, padx = 5, pady = 5 )
input_save_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Load in historical demand data
# Add widget
lb_historical = tk.Label( window, text = "Historical Data:", height = 1, font = ( "Arial", 12 ) )
loadFile_historical = tk.Entry( window, width = 91, state = tk.NORMAL )
loadFile_historical_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Historical, state = tk.NORMAL )
# Locate widget
lb_historical.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
loadFile_historical.grid( row = row_idx, column = 1, columnspan = 6, padx = 5, pady = 5, sticky = tk.W )
loadFile_historical_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Load in future demand data
# Add widget
lb_future = tk.Label( window, text = "Future Data:", height = 1, font = ( "Arial", 12 ) )
loadFile_future = tk.Entry( window, width = 91, state = tk.DISABLED )
loadFile_future_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Future, state = tk.DISABLED )
# Locate widget
lb_future.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
loadFile_future.grid( row = row_idx, column = 1, columnspan = 6, padx = 5, pady = 5, sticky = tk.W )
loadFile_future_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Load in pre-trained intercept
# Add widget
lb_intercept = tk.Label( window, text = "Intercept:", height = 1, font = ( "Arial", 12 ) )
loadFile_intercept = tk.Entry( window, width = 91, state = tk.DISABLED )
loadFile_intercept_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Intercept, state = tk.DISABLED )
# Locate widget
lb_intercept.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
loadFile_intercept.grid( row = row_idx, column = 1, columnspan = 6, padx = 5, pady = 5, sticky = tk.W )
loadFile_intercept_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Load in pre-trained uncertainty
# Add widget
lb_uncertainty = tk.Label( window, text = "Uncertainty:", height = 1, font = ( "Arial", 12 ) )
loadFile_uncertainty = tk.Entry( window, width = 91, state = tk.DISABLED )
loadFile_uncertainty_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Uncertainty, state = tk.DISABLED )
# Locate widget
lb_uncertainty.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
loadFile_uncertainty.grid( row = row_idx, column = 1, columnspan = 6, padx = 5, pady = 5, sticky = tk.W )
loadFile_uncertainty_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Production cost
# Add widget
production_radioVar = tk.IntVar()
lb_production = tk.Label( window, text = "Production Cost:", height = 1, font = ( "Arial", 12 ) )
production_single_radio = tk.Radiobutton( window, text = "Single", variable = production_radioVar, value = 1, state = tk.NORMAL )
input_production_single = tk.Entry( window, width = 10, state = tk.NORMAL )
production_multi_radio = tk.Radiobutton( window, text = "Multi", variable = production_radioVar, value = 2, state = tk.NORMAL )
loadFile_production = tk.Entry( window, width = 47, state = tk.NORMAL )
loadFile_production_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Production, state = tk.NORMAL )
# Locate widget
lb_production.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
production_single_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_production_single.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
production_multi_radio.grid( row = row_idx, column = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_production.grid( row = row_idx, column = 4, columnspan = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_production_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Holding cost
# Add widget
holding_radioVar = tk.IntVar()
lb_holding = tk.Label( window, text = "Holding Cost:", height = 1, font = ( "Arial", 12 ) )
holding_single_radio = tk.Radiobutton( window, text = "Single", variable = holding_radioVar, value = 1, state = tk.NORMAL )
input_holding_single = tk.Entry( window, width = 10, state = tk.NORMAL )
holding_multi_radio = tk.Radiobutton( window, text = "Multi", variable = holding_radioVar, value = 2, state = tk.NORMAL )
loadFile_holding = tk.Entry( window, width = 47, state = tk.NORMAL )
loadFile_holding_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Holding, state = tk.NORMAL )
# Locate widget
lb_holding.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
holding_single_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_holding_single.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
holding_multi_radio.grid( row = row_idx, column = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_holding.grid( row = row_idx, column = 4, columnspan = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_holding_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Backorder cost
# Add widget
backorder_radioVar = tk.IntVar()
lb_backorder = tk.Label( window, text = "Backorder Cost:", height = 1, font = ( "Arial", 12 ) )
backorder_single_radio = tk.Radiobutton( window, text = "Single", variable = backorder_radioVar, value = 1, state = tk.NORMAL )
input_backorder_single = tk.Entry( window, width = 10, state = tk.NORMAL )
backorder_multi_radio = tk.Radiobutton( window, text = "Multi", variable = backorder_radioVar, value = 2, state = tk.NORMAL )
loadFile_backorder = tk.Entry( window, width = 47, state = tk.NORMAL )
loadFile_backorder_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Backorder, state = tk.NORMAL )
# Locate widget
lb_backorder.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
backorder_single_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_backorder_single.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
backorder_multi_radio.grid( row = row_idx, column = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_backorder.grid( row = row_idx, column = 4, columnspan = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_backorder_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Capacity
# Add widget
capacity_radioVar = tk.IntVar()
lb_capacity = tk.Label( window, text = "Capacity:", height = 1, font = ( "Arial", 12 ) )
capacity_single_radio = tk.Radiobutton( window, text = "Single", variable = capacity_radioVar, value = 1, state = tk.NORMAL )
input_capacity_single = tk.Entry( window, width = 10, state = tk.NORMAL )
capacity_multi_radio = tk.Radiobutton( window, text = "Multi", variable = capacity_radioVar, value = 2, state = tk.NORMAL )
loadFile_capacity = tk.Entry( window, width = 47, state = tk.NORMAL )
loadFile_capacity_btn = tk.Button( window, text = "...", height = 1, command = loadFile_Capacity, state = tk.NORMAL )

# Locate widget
lb_capacity.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
capacity_single_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_capacity_single.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
capacity_multi_radio.grid( row = row_idx, column = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_capacity.grid( row = row_idx, column = 4, columnspan = 3, padx = 5, pady = 5, sticky = tk.W )
loadFile_capacity_btn.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Hyperparameters checkbutton
# Add widget
lb_hyperparameters = tk.Label( window, text = "Hyperparameters:", height = 1, font = ( "Arial", 12 ) )
cv_checkbutton = tk.IntVar()
cv_checkbutton1 = tk.Checkbutton( window, text = "Cross-Validation", variable = cv_checkbutton, command = Check_Button_State_Hyperparameter, state = tk.NORMAL )
specific_checkbutton = tk.IntVar()
specific_checkbutton1 = tk.Checkbutton( window, text = "Specific", variable = specific_checkbutton, command = Check_Button_State_Hyperparameter, state = tk.NORMAL )
# Locate widget
lb_hyperparameters.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
cv_checkbutton1.grid( row = row_idx, column = 1, columnspan = 2, padx = 5, pady = 5, sticky = tk.W )
specific_checkbutton1.grid( row = row_idx, column = 3, columnspan = 2, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#! Cross-Validation area
#* Labelframe
# Add widget
cv_area = tk.LabelFrame( window, text = "Cross-Validation", height = 1, font = ( "Arial", 12 ), width = 250 )
# Locate widget
cv_area.grid( row = row_idx, column = 0, columnspan = 8, padx = 5, pady = 5 )

#* K-fold
# Add widget
lb_kfold = tk.Label( cv_area, text = "K-fold:", height = 1, font = ( "Arial", 12 ) )
kfold_radioVar = tk.IntVar()
kfold_5_radio = tk.Radiobutton( cv_area, text = "5", variable = kfold_radioVar, value = 5, state = tk.DISABLED )
kfold_5_radio.select()
kfold_10_radio = tk.Radiobutton( cv_area, text = "10", variable = kfold_radioVar, value = 10, state = tk.DISABLED )
kfold_other_radio = tk.Radiobutton( cv_area, text = "Other", variable = kfold_radioVar, value = 0, state = tk.DISABLED )
input_kfold = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
# Locate widget
lb_kfold.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
kfold_5_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
kfold_10_radio.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
kfold_other_radio.grid( row = row_idx, column = 3, padx = 5, pady = 5, sticky = tk.W )
input_kfold.grid( row = row_idx, column = 4, columnspan = 2, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* epsilon_N
# Add widget
lb_epsilon_N = tk.Label( cv_area, text = "Epsilon_N:", height = 1, font = ( "Arial", 12 ) )
epsilon_N_radioVar = tk.IntVar()
logspace_epsilon_N_radio = tk.Radiobutton( cv_area, text = "Logspace", variable = epsilon_N_radioVar, value = 0, state = tk.DISABLED )
logspace_epsilon_N_radio.select()
input_epsilon_lb = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb1 = tk.Label( cv_area, text = "to", height = 1, font = ( "Arial", 12 ) )
input_epsilon_ub = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb2 = tk.Label( cv_area, text = ", nums:", height = 1, font = ( "Arial", 12 ) )
input_epsilon_num = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
other_epsilon_N_radio = tk.Radiobutton( cv_area, text = "Other", variable = epsilon_N_radioVar, value = 1, state = tk.DISABLED )
input_epsilon_other = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
# Locate widget
lb_epsilon_N.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
logspace_epsilon_N_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_epsilon_lb.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
lb1.grid( row = row_idx, column = 3, padx = 5, pady = 5 )
input_epsilon_ub.grid( row = row_idx, column = 4, padx = 5, pady = 5, sticky = tk.W )
lb2.grid( row = row_idx, column = 5, padx = 5, pady = 5, sticky = tk.W )
input_epsilon_num.grid( row = row_idx, column = 6, padx = 5, pady = 5, sticky = tk.W )
other_epsilon_N_radio.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
input_epsilon_other.grid( row = row_idx, column = 8, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* omega
# Add widget
lb_omega = tk.Label( cv_area, text = "Omega:", height = 1, font = ( "Arial", 12 ) )
omega_radioVar = tk.IntVar()
arange_omega_radio = tk.Radiobutton( cv_area, text = "Range", variable = omega_radioVar, value = 0, state = tk.DISABLED )
arange_omega_radio.select()
input_omega_lb = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb1 = tk.Label( cv_area, text = "to", height = 1, font = ( "Arial", 12 ) )
input_omega_ub = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb2 = tk.Label( cv_area, text = ", increment:", height = 1, font = ( "Arial", 12 ) )
input_omega_increment = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
other_omega_radio = tk.Radiobutton( cv_area, text = "Other", variable = omega_radioVar, value = 1, state = tk.DISABLED )
input_omega_other = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
# Locate widget
lb_omega.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
arange_omega_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_omega_lb.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
lb1.grid( row = row_idx, column = 3, padx = 5, pady = 5 )
input_omega_ub.grid( row = row_idx, column = 4, padx = 5, pady = 5, sticky = tk.W )
lb2.grid( row = row_idx, column = 5, padx = 5, pady = 5, sticky = tk.W )
input_omega_increment.grid( row = row_idx, column = 6, padx = 5, pady = 5, sticky = tk.W )
other_omega_radio.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
input_omega_other.grid( row = row_idx, column = 8, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* n_estimators
# Add widget
lb_n_estimators = tk.Label( cv_area, text = "n_estimators:", height = 1, font = ( "Arial", 12 ) )
n_estimators_radioVar = tk.IntVar()
arange_n_estimators_radio = tk.Radiobutton( cv_area, text = "Range", variable = n_estimators_radioVar, value = 0, state = tk.DISABLED )
arange_n_estimators_radio.select()
input_n_estimators_lb = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb1 = tk.Label( cv_area, text = "to", height = 1, font = ( "Arial", 12 ) )
input_n_estimators_ub = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb2 = tk.Label( cv_area, text = ", increment:", height = 1, font = ( "Arial", 12 ) )
input_n_estimators_increment = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
other_n_estimators_radio = tk.Radiobutton( cv_area, text = "Other", variable = n_estimators_radioVar, value = 1, state = tk.DISABLED )
input_n_estimators_other = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
# Locate widget
lb_n_estimators.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
arange_n_estimators_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_n_estimators_lb.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
lb1.grid( row = row_idx, column = 3, padx = 5, pady = 5 )
input_n_estimators_ub.grid( row = row_idx, column = 4, padx = 5, pady = 5, sticky = tk.W )
lb2.grid( row = row_idx, column = 5, padx = 5, pady = 5, sticky = tk.W )
input_n_estimators_increment.grid( row = row_idx, column = 6, padx = 5, pady = 5, sticky = tk.W )
other_n_estimators_radio.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
input_n_estimators_other.grid( row = row_idx, column = 8, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* max_depth
# Add widget
lb_max_depth = tk.Label( cv_area, text = "max_depth:", height = 1, font = ( "Arial", 12 ) )
max_depth_radioVar = tk.IntVar()
arange_max_depth_radio = tk.Radiobutton( cv_area, text = "Range", variable = max_depth_radioVar, value = 0, state = tk.DISABLED )
arange_max_depth_radio.select()
input_max_depth_lb = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb1 = tk.Label( cv_area, text = "to", height = 1, font = ( "Arial", 12 ) )
input_max_depth_ub = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
lb2 = tk.Label( cv_area, text = ", increment:", height = 1, font = ( "Arial", 12 ) )
input_max_depth_increment = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
other_max_depth_radio = tk.Radiobutton( cv_area, text = "Other", variable = max_depth_radioVar, value = 1, state = tk.DISABLED )
input_max_depth_other = tk.Entry( cv_area, width = 10, state = tk.DISABLED )
# Locate widget
lb_max_depth.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
arange_max_depth_radio.grid( row = row_idx, column = 1, padx = 5, pady = 5, sticky = tk.W )
input_max_depth_lb.grid( row = row_idx, column = 2, padx = 5, pady = 5, sticky = tk.W )
lb1.grid( row = row_idx, column = 3, padx = 5, pady = 5 )
input_max_depth_ub.grid( row = row_idx, column = 4, padx = 5, pady = 5, sticky = tk.W )
lb2.grid( row = row_idx, column = 5, padx = 5, pady = 5, sticky = tk.W )
input_max_depth_increment.grid( row = row_idx, column = 6, padx = 5, pady = 5, sticky = tk.W )
other_max_depth_radio.grid( row = row_idx, column = 7, padx = 5, pady = 5, sticky = tk.W )
input_max_depth_other.grid( row = row_idx, column = 8, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#! Specific area
#* Labelframe
# Add widget
specific_area = tk.LabelFrame( window, text = "Specific", height = 1, font = ( "Arial", 12 ), width = 300 )
# Locate widget
specific_area.grid( row = row_idx, column = 0, columnspan = 8, padx = 5, pady = 5 )

#* epsilon_N
# Add widget
lb_epsilon_N = tk.Label( specific_area, text = "Epsilon_N:", height = 1, font = ( "Arial", 12 ) )
input_epsilon_N = tk.Entry( specific_area, width = 60, state = tk.DISABLED )
# Locate widget
lb_epsilon_N.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
input_epsilon_N.grid( row = row_idx, column = 1, columnspan = 5, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* omega
# Add widget
lb_omega = tk.Label( specific_area, text = "Omega:", height = 1, font = ( "Arial", 12 ) )
input_omega = tk.Entry( specific_area, width = 60, state = tk.DISABLED )
# Locate widget
lb_omega.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
input_omega.grid( row = row_idx, column = 1, columnspan = 5, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* n_estimators
# Add widget
lb_n_estimators = tk.Label( specific_area, text = "n_estimators:", height = 1, font = ( "Arial", 12 ) )
input_n_estimators = tk.Entry( specific_area, width = 60, state = tk.DISABLED )
# Locate widget
lb_n_estimators.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
input_n_estimators.grid( row = row_idx, column = 1, columnspan = 5, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* max_depth
# Add widget
lb_max_depth = tk.Label( specific_area, text = "max_depth:", height = 1, font = ( "Arial", 12 ) )
input_max_depth = tk.Entry( specific_area, width = 60, state = tk.DISABLED )
# Locate widget
lb_max_depth.grid( row = row_idx, column = 0, padx = 5, pady = 5, sticky = tk.W )
input_max_depth.grid( row = row_idx, column = 1, columnspan = 5, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

#* Show progress area
# Add widget
text_area = scrolledtext.ScrolledText( window, width = 125, height = 10 )
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar( window, variable = progress_var, length = 400 )
# Locate widget
text_area.grid( row = row_idx, column = 0, columnspan = 8, padx = 5, pady = 5 )
progress_bar.grid( row = row_idx + 1, column = 0, columnspan = 8, padx = 5, pady = 5 )
row_idx += 2

#* Cancel / Run button
# Add widget
cancel_button = tk.Button( window, text = "Cancel", height = 1, width = 10, command = window.destroy )
run_button = tk.Button( window, text = "Train", height = 1, width = 10, command = Run )
# Locate widget
cancel_button.grid( row = row_idx, column = 5, padx = 5, pady = 5, sticky = tk.W )
run_button.grid( row = row_idx, column = 6, padx = 5, pady = 5, sticky = tk.W )
row_idx += 1

# Run the main loop
window.mainloop()