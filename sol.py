import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


irradiance = np.array([
    0, 0, 0, 50, 120, 250, 400, 600, 700, 800, 850, 900,
    850, 800, 700, 600, 400, 200, 100, 30, 0, 0, 0, 0
])


panel_power_watt = 430
panel_efficiency = 0.204
panel_area_m2 = panel_power_watt / (1000 * panel_efficiency)

num_panels = 40
total_area = panel_area_m2 * num_panels
total_capacity_kw = (panel_power_watt * num_panels) / 1000


solar_output_kwh = (total_area * irradiance * panel_efficiency) / 1000
hours = np.arange(24)


df = pd.read_excel("datacenter_all_loads_timeseries.xlsx", sheet_name="datacenter_all_loads_timeseries")
actual_it = df["Rack Power (kW)"].to_numpy()
actual_cooling = df["Cooling Power (kW)"].to_numpy()
forecast_total = actual_it + actual_cooling  # Using actual total as "forecast"

solar_adjusted_load = forecast_total - solar_output_kwh
solar_adjusted_load = np.clip(solar_adjusted_load, 0, None)


df_solar = pd.DataFrame({
    "Hour": hours,
    "Irradiance (W/m²)": irradiance,
    "Solar Output (kWh)": solar_output_kwh,
    "Baseline load(kW)": forecast_total,
    "Solar-Adjusted Load (kW)": solar_adjusted_load
})
print(df_solar.round(2))


plt.figure(figsize=(12, 5))
plt.plot(hours, forecast_total, label="Baseline load", color="orange", marker='o')
plt.plot(hours, solar_adjusted_load, label="Solar-Adjusted Load", color="green", linestyle='--', marker='s')
plt.title("Baseline load vs Solar-Adjusted Load Profile")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
plt.bar(hours, solar_output_kwh, color="skyblue", label="Solar Output (kWh)")
plt.title("Hourly Solar Energy Generation")
plt.xlabel("Hour of Day")
plt.ylabel("Energy Produced (kWh)")
plt.xticks(hours)
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()


energy_saved = forecast_total - solar_adjusted_load
total_energy_saved = np.sum(energy_saved)

plt.figure(figsize=(12, 5))
plt.plot(hours, forecast_total, label="Baseline load", color="orange", linestyle='--')
plt.plot(hours, solar_adjusted_load, label="Solar-Adjusted Load", color="green", linestyle='-')
plt.fill_between(hours, forecast_total, solar_adjusted_load,
                 where=(forecast_total > solar_adjusted_load),
                 interpolate=True, color='lightgreen', alpha=0.5,
                 label=f"Total Energy Saved: {total_energy_saved:.2f} kWh")

plt.title("Shaded Area Representing Energy Saved by Solar")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
