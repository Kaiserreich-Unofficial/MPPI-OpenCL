#include "mppi/mppi.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>

int main()
{
    // ------------------ 1. 圆形轨迹生成 ------------------
    float radius = 6.0f;   // 圆半径
    float omega = 0.1f;    // 角速度 rad/s
    float dt = 0.05f;      // 仿真步长
    int N_waypoints = 200; // 总步数
    std::vector<float> waypoints;
    uint waypoint_dim = 6; // x, y, psi, u, v, r

    for (int i = 0; i < N_waypoints; i++)
    {
        float t = i * dt;
        float x = radius * cosf(omega * t);
        float y = radius * sinf(omega * t);
        float psi = atan2f(y, x);        // 轨迹切线方向
        float u_target = radius * omega; // surge 目标速度
        float v_target = 0.0f;           // sway
        float r_target = omega;          // yaw rate
        waypoints.push_back(x);
        waypoints.push_back(y);
        waypoints.push_back(psi);
        waypoints.push_back(u_target);
        waypoints.push_back(v_target);
        waypoints.push_back(r_target);
    }

    // ------------------ 2. 初始化 MPPI ------------------
    mppi::Params params;
    params.horizon = 50;           // 预测步数
    params.delta = dt;       // 每步时间
    params.local_size = 128; // OpenCL local size
    params.multiplier = 16;  // 样本数倍增
    params.sigma_Tl = 0.2f;
    params.sigma_Tr = 0.2f;
    params.lambda = 0.5f;
    params.platform_id = 0;                 // OpenCL 平台
    params.cl_file = "../src/mppi/mppi.cl"; // OpenCL 核文件路径

    mppi::MPPI mppi_controller(params);

    // ------------------ 3. 初始 USV 状态 ------------------
    std::vector<float> state = {radius, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> objects; // 空障碍物

    // ------------------ 4. 输出文件 ------------------
    std::ofstream traj_file("usv_trajectory.csv");
    traj_file << "x,y,psi,u,v,r,optimization_time_ms\n";

    // ------------------ 5. 运行仿真 ------------------
    std::vector<double> optimize_times;
    int sim_steps = 200;

    for (int step = 0; step < sim_steps; step++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<mppi::Waypoint> trajectory = mppi_controller.run(state, waypoints, objects);
        auto t2 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        optimize_times.push_back(elapsed_ms);

        if (!trajectory.empty())
        {
            state[0] = trajectory[0].x();
            state[1] = trajectory[0].y();
            state[2] = trajectory[0].psi();
            state[3] = trajectory[0].u();
            state[4] = trajectory[0].v();
            state[5] = trajectory[0].r();
        }

        traj_file << state[0] << "," << state[1] << "," << state[2]
                  << "," << state[3] << "," << state[4] << "," << state[5]
                  << "," << elapsed_ms << "\n";

        std::cout << "Step " << step << ": x=" << state[0]
                  << " y=" << state[1] << " psi=" << state[2]
                  << " u=" << state[3] << " optimization_time=" << elapsed_ms << " ms" << std::endl;
    }

    double sum_time = 0.0;
    for (auto t : optimize_times)
        sum_time += t;
    double avg_time = sum_time / optimize_times.size();
    std::cout << "Average optimization time: " << avg_time << " ms" << std::endl;

    traj_file.close();
    std::cout << "Simulation finished. Trajectory saved to usv_trajectory.csv\n";

    return 0;
}
