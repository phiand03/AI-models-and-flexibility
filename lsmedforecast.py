import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_actual = pd.read_excel("datacenter_all_loads_timeseries.xlsx", sheet_name="datacenter_all_loads_timeseries")
actual_it = df_actual["Rack Power (kW)"].to_numpy()
actual_cooling = df_actual["Cooling Power (kW)"].to_numpy()
actual_total = actual_it + actual_cooling
hours = np.arange(24)


df_forecast = pd.read_excel("Load_Shift_Model_Data.xlsx", sheet_name="Sheet1")
df_forecast = df_forecast.rename(columns={'NARX': 'Forecast_NARX', 'LSTM': 'Forecast_LSTM', 'Real': 'Actual_Load'})

forecast_narx = df_forecast['Forecast_NARX'].to_numpy()
forecast_lstm = df_forecast['Forecast_LSTM'].to_numpy()


forecast_column = forecast_narx 


reduce_hours = [13, 12, 15, 16]
add_hours = [3, 4, 23, 6]


shift_per_peak_hour = 10  


flexible_load = actual_total.copy()


for h in reduce_hours:
    flexible_load[h] -= shift_per_peak_hour


shift_per_target_hour = (shift_per_peak_hour * len(reduce_hours)) / len(add_hours)
for h in add_hours:
    flexible_load[h] += shift_per_target_hour


flexible_load = np.clip(flexible_load, 0, None)

original_peak = np.max(actual_total)
flexible_peak = np.max(flexible_load)
peak_reduction = original_peak - flexible_peak
peak_reduction_pct = (peak_reduction / original_peak) * 100
load_shift_potential = np.sum(np.abs(actual_total - flexible_load))
# Sort and sum the top 4 peak hours
original_top4_sum = np.sort(actual_total)[-4:].sum()
flexible_top4_sum = np.sort(flexible_load)[-4:].sum()
peak_reduction_top4 = original_top4_sum - flexible_top4_sum
peak_reduction_top4_pct = (peak_reduction_top4 / original_top4_sum) * 100


plt.figure(figsize=(12, 6))
plt.plot(hours, actual_total, label="Actual Data Center Load", color='blue', marker='o')
plt.plot(hours, forecast_column, label="Forecasted Load (AI Model)", color='orange', linestyle='--', marker='s')
plt.plot(hours, flexible_load, label="Flexible Load (Adjusted)", color='green', linestyle='-.', marker='^')
plt.title("Actual vs Forecasted vs Flexible Load Profiles")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




plt.figure(figsize=(12, 5))
bar_width = 0.4
x = np.arange(24)
plt.bar(x - bar_width/2, forecast_column, width=bar_width, label='Forecasted Load', alpha=0.7, color='gold')
plt.bar(x + bar_width/2, flexible_load, width=bar_width, label='Flexible Load', alpha=0.7, color='coral')
plt.xticks(x)
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.title("Forecasted vs Flexible Load per Hour (Bar Chart)")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 5))
plt.plot(hours, actual_total, label="Actual Load", color="blue", marker='o')
plt.plot(hours, forecast_column, label="Forecasted Load", color="orange", linestyle='--', marker='s')
plt.plot(hours, flexible_load, label="Flexible Load", color="green", linestyle='-', marker='^')


plt.fill_between(hours, actual_total, flexible_load, where=(actual_total > flexible_load),
                 interpolate=True, color='lightblue', alpha=0.4, label="Load Shifted Away")
plt.fill_between(hours, actual_total, flexible_load, where=(actual_total < flexible_load),
                 interpolate=True, color='lightgreen', alpha=0.4, label="Load Shifted To")


plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.title("Shaded Area Showing Load Shift (From and To, Compared to Actual Load)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


shifted_down_hours = np.where(flexible_load < actual_total)[0]
shifted_up_hours = np.where(flexible_load > actual_total)[0]

highlight_hours = np.unique(np.concatenate((shifted_down_hours, shifted_up_hours)))


plt.figure(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(highlight_hours))
plt.bar(x - bar_width/2, actual_total[highlight_hours], width=bar_width, label="Actual Load", alpha=0.7, color='gold')
plt.bar(x + bar_width/2, flexible_load[highlight_hours], width=bar_width, label="Flexible Load", alpha=0.7, color='coral')
plt.xticks(x, [f"Hour {h}" for h in highlight_hours])
plt.xlabel("Shifted Hours")
plt.ylabel("Power (kW)")
plt.title("Before and After Load Shift (Detected Shifted Hours)")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


df_output = pd.DataFrame({
    "Hour": hours,
    "Actual Load (kW)": actual_total,
    "Forecasted Load (kW)": forecast_column,
    "Flexible Load (kW)": flexible_load
})
print(df_output.round(2).to_string(index=False))
print("\n--- FLEXIBILITY METRICS ---")
print(f"Original Peak Load: {original_peak:.2f} kW")
print(f"Flexible Peak Load: {flexible_peak:.2f} kW")
print(f"Peak Reduction (Single): {peak_reduction:.2f} kW ({peak_reduction_pct:.2f}%)")

print(f"\nOriginal Top 4 Hour Peak Total: {original_top4_sum:.2f} kW")
print(f"Flexible Top 4 Hour Peak Total: {flexible_top4_sum:.2f} kW")
print(f"Peak Reduction (Top 4 Sum): {peak_reduction_top4:.2f} kW ({peak_reduction_top4_pct:.2f}%)")

print(f"\nTotal Load Shifted (absolute): {load_shift_potential:.2f} kWh")