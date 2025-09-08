/*
 * MIT License
 * Copyright (c) 2024 University of Luxembourg
 */

#ifndef PLANNING_MPPI_H
#define PLANNING_MPPI_H

#include <CL/cl.h>
#include <vector>
#include <random>

namespace mppi
{

    typedef struct
    {
        std::string cl_file = "";
        uint64_t platform_id = 0;
        uint64_t local_size = 128;
        uint64_t multiplier = 15;
        uint64_t horizon = 50;
        float delta = 0.05;
        float lambda = 1.0;
        float sigma_Tl = 0.2;
        float sigma_Tr = 0.2;
        float Tl_min = -0.5;
        float Tl_max = 0.5;
        float Tr_min = -0.5;
        float Tr_max = 0.5;
        float vehicle_length = 1.3; // 船体长度
        float reaction_time = 0.5;  // 预测反应时间
        float safe_dist_coef = 1.5; // 障碍物安全距离系数
        float safe_dist_min = 1.5;  // 障碍物最小安全距离
        bool enable_nln = false;
        float state_weights[6] = {100.0f, 100.0f, 1.0f, .1f, .1f, .1f};
        float input_weights[2] = {1e-5, 1e-5};
    } Params;

    class Waypoint
    {
    public:
        Waypoint(float x, float y, float psi,
                 float u, float v, float r)
            : _x(x), _y(y), _psi(psi),
              _u(u), _v(v), _r(r)
        {
        }

        float x() const { return _x; }
        float y() const { return _y; }
        float psi() const { return _psi; }
        float u() const { return _u; }
        float v() const { return _v; }
        float r() const { return _r; }

    private:
        float _x, _y, _psi;
        float _u, _v, _r;
    };

    class MPPI
    {
    public:
        MPPI(Params params);

        std::vector<Waypoint> run(const std::vector<float> &state,
                                  std::vector<float> &waypoints,
                                  std::vector<float> &objects);

    private:
        Params _p;
        uint64_t _num_rollout;
        std::vector<float> _params;

        cl_platform_id *_platforms;
        cl_device_id _device;
        cl_context _context;
        cl_command_queue _queue;
        cl_program _program;
        cl_kernel _kernel;

        cl_mem __params;
        cl_mem __state;
        cl_mem __inputs;
        cl_mem __seed;
        cl_mem __noise;
        cl_mem __costs;

        std::random_device _rd;
        std::mt19937 _rg_seed;
        std::uniform_int_distribution<uint32_t> _dist_seed;

        uint64_t _prev_time = 0;
        size_t _state_size = 6;
        float *_state;
        size_t _waypoint_dim = 6;
        size_t _object_dim = 6;
        size_t _input_dim = 2; // Tl, Tr
        size_t _inputs_size;
        float *_inputs;
        uint32_t *_seed;
        size_t _noise_size;
        float *_noise;
        float *_costs;
    };

}

#endif // PLANNING_MPPI_H
