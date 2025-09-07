import matplotlib.pyplot as plt
import pandas as pd

# -------------------- 1. 读取 CSV --------------------
csv_file = "../build/usv_trajectory.csv"
df = pd.read_csv(csv_file)

x = df['x'].values
y = df['y'].values
psi = df['psi'].values
u = df['u'].values
v = df['v'].values
r = df['r'].values
opt_time = df['optimization_time_ms'].values

# -------------------- 2. 绘制 XY 轨迹 --------------------
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', markersize=3, linestyle='-', color='blue', label='USV Trajectory')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('USV Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# -------------------- 3. 绘制姿态 psi --------------------
plt.figure(figsize=(8, 4))
plt.plot(psi, color='orange', label='Psi [rad]')
plt.xlabel('Time step')
plt.ylabel('Psi [rad]')
plt.title('USV Yaw (Psi) over Time')
plt.grid(True)
plt.legend()
plt.show()

# -------------------- 4. 绘制速度 u --------------------
plt.figure(figsize=(8, 4))
plt.plot(u, color='green', label='Surge velocity u [m/s]')
plt.xlabel('Time step')
plt.ylabel('u [m/s]')
plt.title('USV Surge Velocity over Time')
plt.grid(True)
plt.legend()
plt.show()

# -------------------- 5. 可选：绘制优化耗时 --------------------
plt.figure(figsize=(8, 4))
plt.plot(opt_time, color='red', label='Optimization time [ms]')
plt.xlabel('Time step')
plt.ylabel('Time [ms]')
plt.title('MPPI Optimization Time per Step')
plt.grid(True)
plt.legend()
plt.show()
