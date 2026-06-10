# Tuning Factors
NG_Tune = 0.95
BL_Tune = 0.92
Hydro_Tune = 0.65
Solar_Tune = 0.833
Wind_Tune = 0.45

# Initial Conditions Parameters (MW Installed)
Solar_Installed = 18169
Wind_Installed = 6117 / 3200
Hydro_Power_Max = 12000000
Baseload_Installed = 5149
Import_Limit = 18500 #Set by >95% scenarios passing in calibration
Surplus_Min = 8200 #Set by finding value that would lead to 4 TwH per year curtailment
total_def_limit = 76387179 #Deficit output of calibration model
c1, c2 = -6.5566, 0.2848  # Logistic regression coefficients

Factor_NSC = 0.15
Factor_Smart_Charge = 1 - Factor_NSC
Full_EV_Fleet = 29000  # 29 million, scaled down to MW units