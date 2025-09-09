/*
 * MIT License
 * Copyright (c) 2024 University of Luxembourg
 */

#define M_PI 3.1415927

// private version
float get_noise(uint *seed, float mean, float std) {
  uint r = *seed;
  float s = 0.0f;
  float u = 0.0f;
  float v = 0.0f;

  while ((s == 0.0f) || (s >= 1.0f)) {
    r = (1103515245 * r + 12345) & 0xffffffff;
    u = 2.0f * ((float)r / 4294967295.0f) - 1.0f;

    r = (1103515245 * r + 12345) & 0xffffffff;
    v = 2.0f * ((float)r / 4294967295.0f) - 1.0f;

    s = u * u + v * v;
  }

  *seed = r;
  return mean + (u * sqrt(-2.0f * log(s) / s)) * std;
}

// 对数正态分布（返回的是指数）
float get_log_noise(uint *seed, float mean, float std) {
  uint r = *seed;
  float s = 0.0f;
  float u = 0.0f;
  float v = 0.0f;

  while ((s == 0.0f) || (s >= 1.0f)) {
    r = (1103515245 * r + 12345) & 0xffffffff;
    u = 2.0f * ((float)r / 4294967295.0f) - 1.0f;

    r = (1103515245 * r + 12345) & 0xffffffff;
    v = 2.0f * ((float)r / 4294967295.0f) - 1.0f;

    s = u * u + v * v;
  }
  *seed = r;
  return exp(mean + (u * sqrt(-2.0f * log(s) / s)) * std);
}

float dir_obj_dist(float x, float y, float yaw, float obj_x, float obj_y,
                   float obj_r) {
  float cos_theta = cos(yaw);
  float sin_theta = sin(yaw);
  float w_x =
      cos_theta * obj_x + sin_theta * obj_y - cos_theta * x - sin_theta * y;
  if (w_x < -obj_r)
    return -1.0f;

  float x0 = x;
  float y0 = y;
  float x1 = x + cos_theta;
  float y1 = y + sin_theta;
  float a = y0 - y1;
  float b = x1 - x0;
  float c = x0 * y1 - x1 * y0;
  float dist = fabs(a * obj_x + b * obj_y + c) / sqrt(a * a + b * b);
  return dist;
}

float get_obstacle_dist(float reaction_time, float vehicle_length, float x,
                        float y, float u, float v, __constant float *waypoints,
                        uint waypoints_size, uint waypoint_dim,
                        __constant float *objects, uint objects_size,
                        uint object_dim) {

  const uint nb_objects = objects_size / object_dim;
  const uint nb_waypoints = waypoints_size / waypoint_dim;
  float min_obj_d = FLT_MAX;

  float vel_abs = sqrt(u * u + v * v);
  float safe_horizon = vel_abs * reaction_time + vehicle_length;

  for (uint i = 0; i < nb_objects; i++) {
    float obj_x = objects[i * object_dim + 0];
    float obj_y = objects[i * object_dim + 1];
    float obj_r = objects[i * object_dim + 2];

    float dx = obj_x - x;
    float dy = obj_y - y;
    float dist = sqrt(dx * dx + dy * dy) - obj_r;

    // 船在安全航向内
    if (dist <= safe_horizon) {
      min_obj_d = fmin(min_obj_d, dist);
    }

    // waypoint 引导约束
    uint closest_wp = 0;
    float min_wp_dist = FLT_MAX;
    for (uint j = 0; j < nb_waypoints; j++) {
      float wx = waypoints[j * waypoint_dim + 0];
      float wy = waypoints[j * waypoint_dim + 1];
      float dx_wp = obj_x - wx;
      float dy_wp = obj_y - wy;
      float wp_dist = sqrt(dx_wp * dx_wp + dy_wp * dy_wp);
      if (wp_dist < min_wp_dist) {
        min_wp_dist = wp_dist;
        closest_wp = j;
      }
    }

    // 如果障碍物接近参考 waypoint，增加 cost 惩罚
    if (min_wp_dist < obj_r + 0.5f * vehicle_length) {
      min_obj_d = fmin(min_obj_d, dist);
    }
  }

  return min_obj_d;
}

void dynamics(float x, float y, float psi, float u, float v, float r, float Tl,
              float Tr, float dx[6]) {
  dx[0] = u * cos(psi) - v * sin(psi);
  dx[1] = u * sin(psi) + v * cos(psi);
  dx[2] = r;
  dx[3] = -0.1189f * u - 0.1131f * u * fabs(u) + 0.72696f * v * r +
          1.04965f * Tl + 0.72806f * Tr - 0.05309f * r * r;
  dx[4] = 0.1577f * v + 0.0365f * r - 0.7457f * v * fabs(v) -
          0.0348f * r * fabs(r) - 0.22145f * u * v - 0.0678f * u * r -
          0.0899f * Tl + 0.1015f * Tr;
  dx[5] = 1.11f * v - 0.221f * r - 2.712f * v * fabs(v) -
          0.2083f * r * fabs(r) - 0.6508f * u * v - 0.6683f * u * r -
          1.9953f * Tl + 2.1045f * Tr;
}

__kernel void
mppi(uint horizon, __constant float *params, __constant float *state,
     __constant float *waypoints, uint waypoints_size, uint waypoint_dim,
     __constant float *objects, uint objects_size, uint object_dim,
     __constant float *inputs, uint input_dim, __global uint *seed,
     __global float *noise, __global float *costs, uint enable_nln,
     __constant float *state_weights, __constant float *input_weights) {

  const float delta = params[0];
  const float sigma_Tl = params[1];
  const float sigma_Tr = params[2];
  const float Tl_min = params[3];
  const float Tl_max = params[4];
  const float Tr_min = params[5];
  const float Tr_max = params[6];
  const float vehicle_length = params[7];
  const float reaction_time = params[8];
  const float safe_dist_coef = params[9];
  const float safe_dist_min = params[10];

  const size_t K = get_global_size(0);
  const size_t global_id = get_global_id(0);
  const uint nb_waypoints = waypoints_size / waypoint_dim;

  float vehicle_x = state[0];
  float vehicle_y = state[1];
  float vehicle_psi = state[2];
  float vehicle_u = state[3];
  float vehicle_v = state[4];
  float vehicle_r = state[5];

  uint _seed = seed[global_id];

  float x = vehicle_x;
  float y = vehicle_y;
  float psi = vehicle_psi;
  float u = vehicle_u;
  float v = vehicle_v;
  float r = vehicle_r;

  float prev_target_dist = 0.0f;
  float cost = 0.0f;

  uint offset_n = global_id * input_dim;
  uint offset_i = 0;
  float Tl, Tr = 0.0f;
  for (uint i = 0; i < horizon; i++) {
    if (enable_nln) { // 乘性噪声
      Tl = inputs[offset_i] * get_log_noise(&_seed, 0.0f, sigma_Tl);
      Tr = inputs[offset_i + 1] * get_log_noise(&_seed, 0.0f, sigma_Tr);
    } else {
      Tl = inputs[offset_i] + get_noise(&_seed, 0.0f, sigma_Tl);
      Tr = inputs[offset_i + 1] + get_noise(&_seed, 0.0f, sigma_Tr);
    }

    Tl = clamp(Tl, Tl_min, Tl_max);
    Tr = clamp(Tr, Tr_min, Tr_max);

    float k1[6], k2[6], k3[6], k4[6];
    dynamics(x, y, psi, u, v, r, Tl, Tr, k1);
    dynamics(x + 0.5f * delta * k1[0], y + 0.5f * delta * k1[1],
             psi + 0.5f * delta * k1[2], u + 0.5f * delta * k1[3],
             v + 0.5f * delta * k1[4], r + 0.5f * delta * k1[5], Tl, Tr, k2);
    dynamics(x + 0.5f * delta * k2[0], y + 0.5f * delta * k2[1],
             psi + 0.5f * delta * k2[2], u + 0.5f * delta * k2[3],
             v + 0.5f * delta * k2[4], r + 0.5f * delta * k2[5], Tl, Tr, k3);
    dynamics(x + delta * k3[0], y + delta * k3[1], psi + delta * k3[2],
             u + delta * k3[3], v + delta * k3[4], r + delta * k3[5], Tl, Tr,
             k4);

    x += delta / 6.0f * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    y += delta / 6.0f * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    psi += delta / 6.0f * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    u += delta / 6.0f * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
    v += delta / 6.0f * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]);
    r += delta / 6.0f * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]);

    // 三元运算裁剪到 [0, 2*pi]
    psi = (psi >= 2.0f * M_PI) ? (psi - 2.0f * M_PI)
                               : ((psi < 0.0f) ? (psi + 2.0f * M_PI) : psi);

    // --- compute per-step cost: compare to appropriate waypoint
    // choose a look-ahead waypoint index; here we simply use i-th ahead from
    // start (clamped to last)
    uint wp_idx = (i < nb_waypoints) ? i : (nb_waypoints - 1);
    uint base_wp = wp_idx * waypoint_dim;

    // state error (x,y,psi,u,v,r)
    float diff_s0 = x - waypoints[base_wp + 0];
    float diff_s1 = y - waypoints[base_wp + 1];
    float diff_s2 = psi - waypoints[base_wp + 2];
    float diff_s3 = u - waypoints[base_wp + 3];
    float diff_s4 = v - waypoints[base_wp + 4];
    float diff_s5 = r - waypoints[base_wp + 5];

    // accumulate weighted state cost
    cost += state_weights[0] * diff_s0 * diff_s0;
    cost += state_weights[1] * diff_s1 * diff_s1;
    cost += state_weights[2] * diff_s2 * diff_s2;
    cost += state_weights[3] * diff_s3 * diff_s3;
    cost += state_weights[4] * diff_s4 * diff_s4;
    cost += state_weights[5] * diff_s5 * diff_s5;

    // accumulate input cost for this timestep (assume input_weights length ==
    // input_dim)
    cost += input_weights[0] * (Tl * Tl);
    cost += input_weights[1] * (Tr * Tr);

    // advance input offset
    offset_i += input_dim;
  }

  // obstacle distance & cost computation
  float min_obj_d = get_obstacle_dist(reaction_time, vehicle_length, x, y, u, v,
                                      waypoints, waypoints_size, waypoint_dim,
                                      objects, objects_size, object_dim);

  float target_velocity = waypoints[(nb_waypoints - 1) * waypoint_dim + 3];
  if (u > target_velocity)
    u = target_velocity;

  float target_safe_dist = safe_dist_coef * u + safe_dist_min;
  float safe_dist = fmax(target_safe_dist - min_obj_d, 0.0f);

  // 保留 obstacle / safe distance
  // cost += 15.0f * fmax(0.0f, -min_obj_d) * fmax(0.0f, -min_obj_d);
  // cost += 25.0f * safe_dist * safe_dist;
  // 存储噪声
  noise[offset_n] = Tl - inputs[offset_i];
  noise[offset_n + 1] = Tr - inputs[offset_i + 1];

  costs[global_id] = cost;
  seed[global_id] = _seed;
}
