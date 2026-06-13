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
#Baseload_Installed = 5149
NG_Installed = 39689
Hydro_Installed = 12281
Import_Limit = 12801 #Set by >95% scenarios passing in calibration
Surplus_Min = 8200 #Set by finding value that would lead to 4 TwH per year curtailment
total_def_limit = 76387179 #Deficit output of calibration model
c1, c2 = -6.5566, 0.2848  # Logistic regression coefficients

Factor_NSC = 0.15
Factor_Smart_Charge = 1 - Factor_NSC
Full_EV_Fleet = 29000  # 29 million, scaled down to MW units

###Financial Data
V2G_CR = None
energy_sources = {
    "nuclear": {
        "capital_cost_per_kw": 7030,  # $/kW installed
        "fixed_om_cost_per_kw_year": 127.35,  # $/kW-year
        "variable_om_cost_per_mwh": 10.30,  # $/MWh
        "transmission": 1.1, #  $/MWh
        "operational_lifetime_years": 60,  # years
        "discount_rate": 0.05  # Discount rate
    },
    "wind": {
        "capital_cost_per_kw": 1718,  # $/kW installed
        "fixed_om_cost_per_kw_year": 27.57,  # $/kW-year
        "variable_om_cost_per_mwh": 0,  # $/MWh
        "transmission": 2.74, #  $/MWh
        "operational_lifetime_years": 30,  # years
        "discount_rate": 0.05  # Discount rate
    },
    "solar": {
        "capital_cost_per_kw": 1327,  # $/kW installed
        "fixed_om_cost_per_kw_year": 15.97,  # $/kW-year
        "variable_om_cost_per_mwh": 0,  # $/MWh
        "transmission": 3.59, #  $/MWh
        "operational_lifetime_years": 30,  # years
        "discount_rate": 0.05  # Discount rate
    },
    "battery": {
        "capital_cost_per_kw": 1316,  # $/kW installed
        "fixed_om_cost_per_kw_year": 25.96,  # $/kW-year
        "variable_om_cost_per_mwh": 24.83,  # $/MWh
        "transmission": 10.24, #  $/MWh
        "operational_lifetime_years": 20,  # years
        "discount_rate": 0.05  # Discount rate
    },
    "hydro": {
        #"capital_cost_per_kw": 3083,  # $/kW installed
        "capital_cost_per_kw": 0,  # $/kW installed
        "fixed_om_cost_per_kw_year": 43.78,  # $/kW-year
        "variable_om_cost_per_mwh": 1.46,  # $/MWh
        "transmission": 2.02, #  $/MWh
        "operational_lifetime_years": 60,  # years
        "discount_rate": 0.05  # Discount rate
    },
    "natural_gas": {
        "capital_cost_per_kw": 1062,  # $/kW installed
        "fixed_om_cost_per_kw_year": 12.77,  # $/kW-year
        "variable_om_cost_per_mwh": 27.77,  # $/MWh
        "transmission": 1.14, #  $/MWh
        "operational_lifetime_years": 30,  # years
        "discount_rate": 0.05  # Discount rate
    },

    "V2G": {
        "connection_cost": 3000,  # $ per V2G-enabled EV
        "fixed_om_cost_per_kw_year": 0.0,  # No separate fixed O&M assumed; fixed cost represented by annualized V2G connection cost
        "variable_om_cost_per_mwh": 0,  # $/MWh, figured as additional battery wear
        "transmission": 10.24, #  $/MWh
        "operational_lifetime_years": 30,  # years
        "discount_rate": 0.05  # Discount rate
    },


}

def calculate_lcoe(params, installed_capacity_mw, annual_energy_used_mwh, source_name=None, V2G_CR=None, num_connections=None, V2G_Connect_Cost=None):
    capacity_kw = installed_capacity_mw * 1000

    if source_name == "V2G":
        battery_size_kwh = 40  # assumed per vehicle
        #connection_cost = params["connection_cost"]  # e.g., $900
        connection_cost = V2G_Connect_Cost
        if V2G_CR is None or V2G_CR <= 0:
            raise ValueError("V2G_CR must be a positive value for V2G LCOE")

        # Compute number of V2G connections needed to support the installed capacity
        num_connections = capacity_kw / (V2G_CR * battery_size_kwh)
        capex = num_connections * connection_cost
        print(f"CONN COST {connection_cost}")
        print(f"Capital Expense V2G is #Conns {num_connections}, #CostPerConn {connection_cost}, total capex {capex}")
    else:
        capex = params["capital_cost_per_kw"] * capacity_kw

    fixed_om = params["fixed_om_cost_per_kw_year"] * capacity_kw
    variable_om = params["variable_om_cost_per_mwh"] * annual_energy_used_mwh
    transmission_cost = params["transmission"] * annual_energy_used_mwh
    lifetime = params["operational_lifetime_years"]
    discount_rate = params["discount_rate"]

    crf = (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)
    annualized_capital_cost = capex * crf

    if annual_energy_used_mwh == 0:
        annual_energy_used_mwh = 1

    total_annual_cost = annualized_capital_cost + fixed_om + variable_om + transmission_cost
    return total_annual_cost / annual_energy_used_mwh






