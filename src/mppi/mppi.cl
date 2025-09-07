/*
 * MIT License
 * Copyright (c) 2024 University of Luxembourg
 */

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

float get_diff_angle(float a, float b) {
  const float PI = 3.1415927f;
  float diff = a - b;
  if (diff > PI)
    diff -= 2.0f * PI;
  if (diff < -PI)
    diff += 2.0f * PI;
  return fabs(diff);
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

float get_obstacle_dist(float reaction_time, float go_around,
                        float stop_velocity, float vehicle_length,
                        float vehicle_x, float vehicle_y, float vehicle_vel,
                        float x, float y, float yaw, float vel, float accel,
                        __constant float *waypoints, uint waypoints_size,
                        uint waypoint_dim, uint offset_c_w,
                        __constant float *objects, uint objects_size,
                        uint object_dim) {

  const uint nb_waypoints = waypoints_size / waypoint_dim;
  const uint nb_objects = objects_size / object_dim;
  const float cos_theta = cos(yaw);
  const float sin_theta = sin(yaw);

  float v_dx = x - vehicle_x;
  float v_dy = y - vehicle_y;
  float v_dist = sqrt(v_dx * v_dx + v_dy * v_dy);
  if (v_dist < (vehicle_vel * reaction_time))
    return FLT_MAX;

  if (vel <= stop_velocity)
    return FLT_MAX;

  float min_obj_d = FLT_MAX;
  uint offset_o = 0;

  for (uint i = 0; i < nb_objects; i++) {
    float obj_x = objects[offset_o];
    float obj_y = objects[offset_o + 1];
    float obj_r = objects[offset_o + 2];

    float dx = obj_x - x;
    float dy = obj_y - y;
    float obj_d = sqrt(dx * dx + dy * dy) - obj_r;
    if (obj_d <= 0.0f)
      min_obj_d = fmin(min_obj_d, obj_d);

    float dir_dist = dir_obj_dist(x, y, yaw, obj_x, obj_y, obj_r);
    if (dir_dist != -1.0f) {
      float horizon =
          (vel + accel * reaction_time) * reaction_time + vehicle_length;
      if ((dir_dist <= obj_r) && (obj_d < horizon))
        min_obj_d = fmin(min_obj_d, obj_d);
    }

    uint offset_w = 0;
    for (uint j = 0; j < nb_waypoints; j++) {
      float w_x =
          cos_theta * obj_x + sin_theta * obj_y - cos_theta * x - sin_theta * y;
      float w_y = -sin_theta * obj_x + cos_theta * obj_y + sin_theta * x -
                  cos_theta * y;
      if (w_x < -obj_r) {
        offset_w += waypoint_dim;
        continue;
      }

      float dxw = obj_x - waypoints[offset_w];
      float dyw = obj_y - waypoints[offset_w + 1];
      float dist = sqrt(dxw * dxw + dyw * dyw) - obj_r;

      if ((w_y < 0.0f) && (dist < -go_around) && (obj_d < min_obj_d))
        min_obj_d = obj_d;
      if ((w_y >= 0.0f) && (dist < 0.0f) && (obj_d < min_obj_d))
        min_obj_d = obj_d;

      offset_w += waypoint_dim;
    }

    offset_o += object_dim;
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

__kernel void mppi(uint N, __constant float *params, __constant float *state,
                   __constant float *waypoints, uint waypoints_size,
                   uint waypoint_dim, __constant float *objects,
                   uint objects_size, uint object_dim, __constant float *inputs,
                   uint input_dim, __global uint *seed, __global float *noise,
                   __global float *costs) {

  const float delta = params[0];
  const float sigma_Tl = params[1];
  const float sigma_Tr = params[2];
  const float vehicle_length = params[3];
  const float reaction_time = params[9];
  const float go_around = params[10];
  const float safe_dist_coef = params[11];
  const float safe_dist_min = params[12];
  const float stop_velocity = params[13];

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

  for (uint i = 0; i < N; i++) {
    float n_Tl = get_noise(&_seed, 0.0f, sigma_Tl);
    float n_Tr = get_noise(&_seed, 0.0f, sigma_Tr);

    float Tl = inputs[offset_i] + n_Tl;
    float Tr = inputs[offset_i + 1] + n_Tr;

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
    float M_PI = 3.1415927f;
    psi = (psi >= 2.0f * M_PI) ? (psi - 2.0f * M_PI)
                               : ((psi < 0.0f) ? (psi + 2.0f * M_PI) : psi);

    offset_i += input_dim;
  }

  // obstacle distance & cost computation
  float min_obj_d = get_obstacle_dist(
      reaction_time, go_around, stop_velocity, vehicle_length, vehicle_x,
      vehicle_y, vehicle_u, x, y, psi, u, 0.0f, waypoints, waypoints_size,
      waypoint_dim, 0, objects, objects_size, object_dim);

  float target_velocity = waypoints[(nb_waypoints - 1) * waypoint_dim + 3];
  if (u > target_velocity)
    u = target_velocity;

  float target_safe_dist = safe_dist_coef * u + safe_dist_min;
  float safe_dist = fmax(target_safe_dist - min_obj_d, 0.0f);

  // 提取目标状态（waypoint 最后一个点）
  float target_x = waypoints[(nb_waypoints - 1) * waypoint_dim + 0];
  float target_y = waypoints[(nb_waypoints - 1) * waypoint_dim + 1];
  float target_psi = waypoints[(nb_waypoints - 1) * waypoint_dim + 2];
  float target_u = waypoints[(nb_waypoints - 1) * waypoint_dim + 3];
  float target_v = waypoints[(nb_waypoints - 1) * waypoint_dim + 4];
  float target_r = waypoints[(nb_waypoints - 1) * waypoint_dim + 5];

  // 计算各状态误差
  float dx = x - target_x;
  float dy = y - target_y;
  float dpsi = get_diff_angle(psi, target_psi);
  float du = u - target_u;
  float dv = v - target_v;
  float dr = r - target_r;

  // 定义权重
  float wx = 100.f;  // x
  float wy = 100.f;  // y
  float wpsi = 1.0f; // psi
  float wu = 1.0f;   // surge
  float wv = .1f;    // sway
  float wr = .1f;    // yaw rate

  // 全维状态代价
  cost += wx * dx * dx + wy * dy * dy + wpsi * dpsi * dpsi + wu * du * du +
          wv * dv * dv + wr * dr * dr;

  // 保留 obstacle / safe distance
  cost += 15.0f * min_obj_d * min_obj_d;
  cost += 25.0f * safe_dist * safe_dist;

  noise[offset_n] = 0.0f; // 可选，如果需要存噪声可以改写
  noise[offset_n + 1] = 0.0f;

  costs[global_id] = cost;
  seed[global_id] = _seed;
}
