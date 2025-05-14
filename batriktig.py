import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("datacenter_all_loads_timeseries.xlsx", sheet_name="datacenter_all_loads_timeseries")
actual_it = df["Rack Power (kW)"].to_numpy()
actual_cooling = df["Cooling Power (kW)"].to_numpy()
actual_total = actual_it + actual_cooling

hours = np.arange(24)


battery_capacity_kwh = 600
battery_power_limit_kw = 60 

min_soc = 0.1 * battery_capacity_kwh  
max_soc = 0.9 * battery_capacity_kwh  

soc = 300-60  
soc_trace = []
battery_adjusted_load = actual_total.copy()

charge_hours = [0, 1, 2, 3, 4, 5,6,7,21,22,23]
discharge_hours = [10,11,12,13,14, 15, 16, 17,18]


for h in hours:
    if h in discharge_hours and soc > min_soc:
        discharge = min(battery_power_limit_kw, soc - min_soc)
        battery_adjusted_load[h] -= discharge
        soc -= discharge
    elif h in charge_hours and soc < max_soc:
        charge = min(battery_power_limit_kw, max_soc - soc)
        battery_adjusted_load[h] += charge
        soc += charge
    soc_trace.append(soc)


df_battery = pd.DataFrame({
    "Hour": hours,
    "Actual Load (kW)": actual_total,
    "Battery-Adjusted Load (kW)": battery_adjusted_load,
    "State of Charge (kWh)": soc_trace
})
print(df_battery.round(2).to_string(index=False))


plt.figure(figsize=(12, 5))
plt.plot(hours, actual_total, label="Actual Load", color="blue", marker='o')
plt.plot(hours, battery_adjusted_load, label="Battery-Adjusted Load", color="green", linestyle='--', marker='s')
plt.title("Battery Impact on Load Profile (Charging: 0-4h and 20-00, Discharging: 9-17h)")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.ylim(bottom=0)




plt.figure(figsize=(12, 5))
plt.plot(hours, soc_trace, label="State of Charge (kWh)", color="purple", marker='^')
plt.axhline(y=min_soc, color='red', linestyle='--', label="Min SoC (10%)")
plt.axhline(y=max_soc, color='orange', linestyle='--', label="Max SoC (90%)")
plt.title("Battery State of Charge Over 24 Hours")
plt.xlabel("Hour of Day")
plt.ylabel("SoC (kWh)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
