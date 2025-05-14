import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("datacenter_all_loads_timeseries.xlsx", sheet_name="datacenter_all_loads_timeseries")
actual_it = df["Rack Power (kW)"].to_numpy()
actual_cooling = df["Cooling Power (kW)"].to_numpy()
actual_total = actual_it + actual_cooling
hours = np.arange(24)

panel_power_watt = 430
panel_efficiency = 0.204
panel_area_m2 = panel_power_watt / (1000 * panel_efficiency)
num_panels = 40
total_area = panel_area_m2 * num_panels

irradiance = np.array([
    0, 0, 0, 50, 120, 250, 400, 600, 700, 800, 850, 900,
    850, 800, 700, 600, 400, 200, 100, 30, 0, 0, 0, 0
])
solar_output_kwh = (total_area * irradiance * panel_efficiency) / 1000

battery_capacity_kwh = 600
battery_power_limit_kw = 50
min_soc = 0.1 * battery_capacity_kwh  # 25 kWh
max_soc = 0.9 * battery_capacity_kwh  # 225 kWh
soc = 60
soc_trace = []
battery_adjusted_load = actual_total.copy()
solar_used_directly = []
solar_used_for_battery = []
battery_discharge_power = []

discharge_hours = [11,12,13,14, 15, 16, 17,18,19]
# ---------------------------------------------
for h in hours:
    original_load = battery_adjusted_load[h]
    available_solar = solar_output_kwh[h]

    potential_charge = min(battery_power_limit_kw, available_solar, max_soc - soc)
    soc += potential_charge
    solar_used_for_battery.append(potential_charge)
    available_solar -= potential_charge

    direct_solar_use = min(available_solar, battery_adjusted_load[h])
    battery_adjusted_load[h] -= direct_solar_use
    solar_used_directly.append(direct_solar_use)

    discharge = 0
    if h in discharge_hours and soc > min_soc:
        discharge = min(battery_power_limit_kw, soc - min_soc)
        battery_adjusted_load[h] -= discharge
        soc -= discharge
    battery_discharge_power.append(discharge)

    soc_trace.append(soc)
   


flexibility_per_hour = actual_total - battery_adjusted_load
total_flexibility_kwh = np.sum(flexibility_per_hour)


plt.figure(figsize=(12, 6))
plt.plot(hours, actual_total, label="Original Load", color="blue", linewidth=1.5)
plt.plot(hours, battery_adjusted_load, label="Adjusted Load", color="green", linestyle='--', linewidth=1.5)


plt.fill_between(hours, battery_adjusted_load, actual_total,
                 where=(battery_adjusted_load < actual_total),
                 interpolate=True, color='lightgreen', alpha=0.5,
                 label=f"Flexibility Provided: {total_flexibility_kwh:.2f} kWh")




plt.title("Flexibility Provision in Data Center Load Profile (with kWh Labels)")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

total_energy_saved = np.sum(actual_total - battery_adjusted_load)
original_peak = np.max(actual_total)
adjusted_peak = np.max(battery_adjusted_load)
peak_reduction = original_peak - adjusted_peak
peak_reduction_pct = (peak_reduction / original_peak) * 100


df_combo = pd.DataFrame({
    "Hour": hours,
    "Irradiance (W/m²)": irradiance,
    "Solar Output (kWh)": solar_output_kwh,
    "Solar Used for Battery (kWh)": solar_used_for_battery,
    "Solar Used Directly (kWh)": solar_used_directly,
    "Battery Discharge (kWh)": battery_discharge_power,
    "Actual Load (kW)": actual_total,
    "Battery-Adjusted Load (kW)": battery_adjusted_load,
    "State of Charge (kWh)": soc_trace
})
print(df_combo.round(2).to_string(index=False))


plt.figure(figsize=(12, 5))
plt.plot(hours, actual_total, label="Actual Load", color="blue", marker='o')
plt.plot(hours, battery_adjusted_load, label="Adjusted Load (Solar + Battery)", color="green", linestyle='--', marker='s')
plt.title("Load Profile with Solar and Battery Integration")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(hours, soc_trace, label="State of Charge (kWh)", color="purple", marker='^')
plt.axhline(y=min_soc, color='red', linestyle='--', label="Min SoC (10%)")
plt.axhline(y=max_soc, color='orange', linestyle='--', label="Max SoC (90%)")
plt.title("Battery SoC with Solar Charging and Peak Discharging")
plt.xlabel("Hour of Day")
plt.ylabel("SoC (kWh)")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


fig, ax1 = plt.subplots(figsize=(12, 5))

ax2 = ax1.twinx()
ax2.plot(hours, irradiance, label="Irradiance (W/m²)", color="gold", marker='o', linewidth=2)
ax2.set_ylabel("Irradiance (W/m²)", color="gold")
ax2.tick_params(axis='y', labelcolor="gold")
ax2.grid(False)

ax1.bar(hours, solar_used_directly, label="Used Directly (kWh)", color="mediumseagreen", alpha=0.6)
ax1.bar(hours, solar_used_for_battery, bottom=solar_used_directly,
        label="Stored in Battery (kWh)", color="steelblue", alpha=0.7)

ax1.set_title("Solar Energy Usage Breakdown (with Dual Y-Axis)")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Energy (kWh)")
ax1.set_ylim(0, max(solar_used_directly + solar_used_for_battery) * 1.5)  # Zoom in on energy bars
ax1.legend(loc="upper left")
ax1.grid(True)

plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))


ax1.plot(hours, actual_total, label="Baseline Load", color="gray", linestyle='--', linewidth=1.5)
ax1.plot(hours, battery_adjusted_load, label="Adjusted Load", color="green", linewidth=2)


ax1.plot(hours, solar_output_kwh, label="Solar Output (kWh)", color="gold", linestyle='-', marker='o', linewidth=2)


ax1.fill_between(hours, battery_adjusted_load, actual_total,
                 where=(battery_adjusted_load < actual_total),
                 interpolate=True, color='lightgreen', alpha=0.4,
                 label="Flexibility (Load Reduction)")


ax2 = ax1.twinx()
ax2.plot(hours, soc_trace, label="Battery SoC (kWh)", color="purple", linestyle='-', marker='^', linewidth=2)
ax2.axhline(y=0, color='purple', linestyle=':', linewidth=1)
ax2.set_ylabel("Battery SoC (kWh)", color="purple")
ax2.tick_params(axis='y', labelcolor="purple")


ax1.set_title("Solar, Battery, and Load Flexibility (Start SoC = 60 kWh)")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Power / Energy (kW or kWh)")
ax1.set_xticks(hours)
ax1.grid(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()


print("\n--- FLEXIBILITY METRICS ---")
print(f"Original Peak Load: {original_peak:.2f} kW")
print(f"Adjusted Peak Load: {adjusted_peak:.2f} kW")
print(f"Peak Reduction: {peak_reduction:.2f} kW ({peak_reduction_pct:.2f}%)")
print(f"Total Energy Saved: {total_energy_saved:.2f} kWh")
