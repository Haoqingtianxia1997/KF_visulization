import time
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class CameraIntrinsics:
    """相机内参"""
    
    def __init__(self, width=640, height=360, fx=366.9034423828125, fy=366.9034423828125, 
                 cx=316.4409484863281, cy=177.47792053222656):
        """
        ZED相机内参（从camera_info获取）
        
        Args:
            width: 图像宽度（像素，默认640）
            height: 图像高度（像素，默认360）
            fx: 水平焦距（像素，默认366.9034）
            fy: 垂直焦距（像素，默认366.9034）
            cx: 光心x坐标（像素，默认316.4409）
            cy: 光心y坐标（像素，默认177.4779）
        """
        self.width = width
        self.height = height
        
        # 使用实际的相机内参
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # 内参矩阵
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    # def __init__(self, width, height, fovy_deg):
    #     """
    #     计算相机内参矩阵
        
    #     Args:
    #         width: 图像宽度（像素）
    #         height: 图像高度（像素）
    #         fovy_deg: 垂直视场角（度）
    #     """
    #     self.width = width
    #     self.height = height
    #     self.fovy_deg = fovy_deg
        
    #     # 计算焦距
    #     self.fy = height / (2.0 * np.tan(np.radians(fovy_deg / 2.0)))
    #     self.fx = self.fy 
        
    #     # 光心
    #     self.cx = width / 2.0
    #     self.cy = height / 2.0
        
    #     # 内参矩阵
    #     self.K = np.array([
    #         [self.fx, 0, self.cx],
    #         [0, self.fy, self.cy],
    #         [0, 0, 1]
    #     ])
        
    def pixel_to_camera_ray(self, u, v):
        """
        将像素坐标转换为相机坐标系中的射线方向
        
        ROS相机坐标系（optical frame）：
        - X轴：右
        - Y轴：下
        - Z轴：前（观察方向）
        
        Args:
            u, v: 像素坐标
        
        Returns:
            归一化的射线方向 [x, y, z]
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0  
        ray = np.array([x, y, z])
        return ray / np.linalg.norm(ray)
        
    def deproject(self, u, v, depth):
        """
        反投影：像素+深度 -> 3D点（相机坐标系）
        
        MuJoCo深度值是Z-buffer深度（沿着-Z轴方向的距离）
        相机坐标系定义：
        - +X: 相机右侧
        - +Y: 相机上方  
        - -Z: 相机观察方向（前方）
        
        深度值 d 表示点在相机坐标系中的 -Z 坐标
        标准针孔相机投影：x_cam = (u - cx) * z / fx, y_cam = (v - cy) * z / fy
        其中 z 是深度（这里是负值，因为观察方向是-Z）
        
        Args:
            u, v: 像素坐标
            depth: 深度值（米，沿着-Z轴方向的距离，总是正值）
        
        Returns:
            3D点 [x, y, z] in camera frame
        """
        # Z坐标（观察方向是-Z，所以z是负的深度值）
        z = -depth
        
        # X和Y坐标根据针孔相机模型计算
        # X: 需要负号修正（因为z是负值）
        # Y: 不需要额外负号（像素v向下，相机Y向上，符号已经相反）
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.array([x, y, z])

class KalmanFilter3D:
    """
    简化版3D卡尔曼滤波器（纯观测驱动，无速度估计）
    状态向量: [x, y, z, vx, vy, vz]
    """
    
    def __init__(self, dt=0.002, g=9.8, process_noise=0.1, measurement_noise=0.05, drag_coefficient=0.0, verbose=False):
        """
        初始化卡尔曼滤波器
        
        Args:
            dt: 时间步长（秒）
            g: 重力加速度（m/s²）
            process_noise: 过程噪声标准差（加速度噪声标准差，单位 m/s²）
            measurement_noise: 测量噪声标准差（位置测量标准差，单位 m）
            drag_coefficient: 空气阻力系数（二次阻力模型中的 k，单位 1/m）
                             满足 a_drag = -k * ||v|| * v
            verbose: 是否打印详细日志
        """
        self.dt = dt
        self.g = g
        self.drag_coefficient = drag_coefficient
        self.verbose = verbose
        
        # 状态向量 [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=float)
        
        # 状态协方差矩阵
        self.P = np.eye(6, dtype=float) * 10.0
        self.velocity_uncertainty = np.diag(self.P)[3:]  # 取速度分量的方差
        
        # 基础状态转移矩阵（保留变量名；实际 predict 时会构造线性化后的 self.F）
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # 控制输入矩阵（保留变量名；predict 里仍会更新使用）
        self.B = np.array([0, 0, -0.5 * g * dt**2, 0, 0, -g * dt], dtype=float)
        
        # 观测矩阵（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=float)
        
        # 过程噪声协方差 Q（6×6，基于加速度扰动模型）
        if isinstance(process_noise, (list, np.ndarray)):
            sigma_a = np.array(process_noise, dtype=float).reshape(3)  # [ax, ay, az]
        else:
            sigma_a = np.full(3, process_noise, dtype=float)
        
        Da = np.diag(sigma_a ** 2)   # 3×3 加速度扰动协方差
        Q_pp = (dt**4 / 4.0) * Da    # 位置-位置块
        Q_pv = (dt**3 / 2.0) * Da    # 位置-速度块
        Q_vv = (dt**2) * Da          # 速度-速度块
        self.Q = np.block([
            [Q_pp, Q_pv],
            [Q_pv, Q_vv]
        ]).astype(float)
        
        # 测量噪声协方差 R（3×3，各向异性）
        if isinstance(measurement_noise, (list, np.ndarray)):
            sigma_m = np.array(measurement_noise, dtype=float).reshape(3)  # [mx, my, mz]
        else:
            sigma_m = np.full(3, measurement_noise, dtype=float)
        
        self.R = np.diag(sigma_m ** 2).astype(float)
        
        self.initialized = False
        self.update_count = 0  # 记录更新次数，用于判断速度估计是否可靠
        
    def initialize(self, position, velocity=None):
        """
        初始化滤波器状态
        
        Args:
            position: 初始位置 [x, y, z]
            velocity: 初始速度 [vx, vy, vz]，默认为0
        """
        if velocity is None:
            velocity = np.zeros(3, dtype=float)
        
        self.x[:3] = np.asarray(position, dtype=float)
        self.x[3:] = np.asarray(velocity, dtype=float)
        
        # 初始协方差：位置较确定，速度不确定
        self.P = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).astype(float)
        self.velocity_uncertainty = np.diag(self.P)[3:]  # 取速度分量的方差
        
        self.initialized = True
    
    def _compute_drag_acceleration_and_jacobian(self, velocity):
        """
        计算二次阻力加速度及其对速度的雅可比
        
        a_drag = -k * ||v|| * v
        
        Returns:
            a_drag: shape (3,)
            J_drag: da_drag / dv, shape (3,3)
        """
        v = np.asarray(velocity, dtype=float)
        speed = np.linalg.norm(v)
        k = self.drag_coefficient
        
        if k <= 0.0 or speed < 1e-12:
            a_drag = np.zeros(3, dtype=float)
            J_drag = np.zeros((3, 3), dtype=float)
            return a_drag, J_drag
        
        # 二次阻力加速度
        a_drag = -k * speed * v
        
        # 雅可比：
        # d/dv ( ||v|| v ) = ||v|| I + (v v^T)/||v||
        I3 = np.eye(3, dtype=float)
        outer_v = np.outer(v, v)
        J_drag = -k * (speed * I3 + outer_v / speed)
        
        return a_drag, J_drag
        
    def predict(self):
        """预测步骤（考虑重力和二次空气阻力）"""
        if not self.initialized:
            return
        
        start_time = time.perf_counter() if self.verbose else None
        
        dt = self.dt
        pos = self.x[:3].copy()
        vel = self.x[3:].copy()
        
        # 重力加速度
        a_gravity = np.array([0.0, 0.0, -self.g], dtype=float)
        
        # 二次阻力加速度及其雅可比
        a_drag, J_drag = self._compute_drag_acceleration_and_jacobian(vel)
        
        # 总加速度
        a_total = a_gravity + a_drag
        
        # 状态预测（非线性）
        pos_new = pos + vel * dt + 0.5 * a_total * dt**2
        vel_new = vel + a_total * dt
        
        self.x[:3] = pos_new
        self.x[3:] = vel_new
        
        # 用当前位置线性化后的雅可比传播协方差
        # p_{k+1} = p_k + v_k dt + 0.5 a(v_k) dt^2
        # v_{k+1} = v_k + a(v_k) dt
        #
        # 因此：
        # d p_{k+1}/d v_k = dt I + 0.5 dt^2 J_drag
        # d v_{k+1}/d v_k = I + dt J_drag
        I3 = np.eye(3, dtype=float)
        self.F = np.block([
            [I3, dt * I3 + 0.5 * dt**2 * J_drag],
            [np.zeros((3, 3), dtype=float), I3 + dt * J_drag]
        ])
        
        # 保留变量名 B，但此时重力和阻力已经直接进了非线性状态更新
        self.B = np.array([0.0, 0.0, -0.5 * self.g * dt**2, 0.0, 0.0, -self.g * dt], dtype=float)
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 数值对称化，避免浮点误差把协方差搞歪
        self.P = 0.5 * (self.P + self.P.T)
        self.velocity_uncertainty = np.diag(self.P)[3:]  # 取速度分量的方差
        
        if self.verbose:
            elapsed_time = (time.perf_counter() - start_time) * 1000.0
            print(f"[KF Predict] 耗时: {elapsed_time:.4f}ms")
        
    def update(self, measurement, current_time=None):
        """
        更新步骤（简化版：纯卡尔曼更新，无额外速度估计）
        
        Args:
            measurement: 观测值 [x, y, z]
            current_time: 当前时间（未使用，保留接口兼容性）
        """
        start_time = time.perf_counter() if self.verbose else None
        
        measurement = np.asarray(measurement, dtype=float)
        
        if not self.initialized:
            # 第一次观测：初始化，速度设为0
            self.initialize(measurement, velocity=np.zeros(3, dtype=float))
            if self.verbose:
                elapsed_time = (time.perf_counter() - start_time) * 1000.0
                print(f"[KF Update-Init] 耗时: {elapsed_time:.4f}ms")
            return
        
        # 标准卡尔曼更新
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益（避免显式求逆）
        PHt = self.P @ self.H.T
        K = np.linalg.solve(S.T, PHt.T).T
        
        # 状态更新
        self.x = self.x + K @ y
        
        # Joseph form 协方差更新
        I = np.eye(6, dtype=float)
        KH = K @ self.H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ self.R @ K.T
        
        # 数值对称化
        self.P = 0.5 * (self.P + self.P.T)
        self.velocity_uncertainty = np.diag(self.P)[3:]  # 取速度分量的方差

        # 增加更新计数
        self.update_count += 1
        
        if self.verbose:
            elapsed_time = (time.perf_counter() - start_time) * 1000.0
            print(f"[KF Update] 耗时: {elapsed_time:.4f}ms")
        
    def get_state(self):
        """获取当前状态"""
        if not self.initialized:
            return None  # 未初始化时返回None,而不是全0状态
        return {
            'position': self.x[:3].copy(),
            'velocity': self.x[3:].copy(),
            'full_state': self.x.copy(),
            'velocity_uncertainty': self.velocity_uncertainty.copy()
        }
    
    def predict_landing_position(self, z_threshold=0.15, min_updates=5, max_velocity_uncertainty=100.0):
        """
        预测小球落地时的位置
        
        Args:
            z_threshold: 地面高度阈值（米）
            min_updates: 最小更新次数（用于判断速度估计是否可靠）
            max_velocity_uncertainty: 最大速度不确定性（协方差对角元素的阈值）
        
        Returns:
            landing_pos: 落地位置 [x, y, z] 或 None（如果无法预测）
            landing_time: 落地时间（秒）或 None
        """
        if not self.initialized:
            return None, None
        
        if self.update_count < min_updates:
            return None, None
        
        velocity_uncertainty = np.diag(self.P)[3:]  # 取速度分量的方差
        if np.any(velocity_uncertainty > max_velocity_uncertainty):
            return None, None
        
        pos = self.x[:3].copy()
        vel = self.x[3:].copy()
        
        if pos[2] <= z_threshold or np.linalg.norm(vel) < 1e-6:
            return None, None
        
        # 为了不改变你的对外接口，这里返回格式保持不变
        # 但因为现在模型已是二次阻力，解析抛体公式就不再严格成立
        # 所以这里改成数值积分预测落地
        dt_sim = min(self.dt, 0.002)
        max_steps = 20000
        
        sim_pos = pos.copy()
        sim_vel = vel.copy()
        sim_time = 0.0
        
        prev_pos = sim_pos.copy()
        prev_time = sim_time
        
        for _ in range(max_steps):
            if sim_pos[2] <= z_threshold:
                break
            
            prev_pos = sim_pos.copy()
            prev_time = sim_time
            
            speed = np.linalg.norm(sim_vel)
            if self.drag_coefficient > 0.0 and speed > 1e-12:
                a_drag = -self.drag_coefficient * speed * sim_vel
            else:
                a_drag = np.zeros(3, dtype=float)
            
            a_total = np.array([0.0, 0.0, -self.g], dtype=float) + a_drag
            
            sim_pos = sim_pos + sim_vel * dt_sim + 0.5 * a_total * dt_sim**2
            sim_vel = sim_vel + a_total * dt_sim
            sim_time += dt_sim
        
        if sim_pos[2] > z_threshold:
            return None, None
        
        # 线性插值过地面时刻
        z1 = prev_pos[2]
        z2 = sim_pos[2]
        
        if abs(z2 - z1) < 1e-12:
            alpha = 1.0
        else:
            alpha = (z_threshold - z1) / (z2 - z1)
            alpha = np.clip(alpha, 0.0, 1.0)
        
        landing_time = prev_time + alpha * (sim_time - prev_time)
        landing_pos = prev_pos + alpha * (sim_pos - prev_pos)
        landing_pos[2] = z_threshold
        
        return landing_pos, landing_time
    
    def reset(self):
        """重置滤波器"""
        self.x = np.zeros(6, dtype=float)
        self.P = np.eye(6, dtype=float) * 10.0
        self.velocity_uncertainty = np.diag(self.P)[3:]  # 取速度分量的方差
        self.initialized = False
        self.update_count = 0


class MultiRedBallDetector:
    """多红色球体检测器 - 同时检测多个球（基于轮廓的方法）
    
    重要：此检测器依赖于输入图像已经过深度mask处理。
    输入图像应该是深度大于阈值（通常1米）的区域被设置为黑色的图像。
    检测器通过识别非黑色区域的轮廓来检测球体。
    """
    
    def __init__(self):
        """初始化检测器参数"""
        # 形态学操作的核
        self.open_kernel = np.ones((21, 21), np.uint8)
        self.close_kernel = np.ones((21, 21), np.uint8)
        
        # 球体形状约束
        self.min_area = 1500  # 最小面积
        self.max_area = 9000  # 最大面积
        self.min_circularity = 0.5  # 最小圆形度（与zed_image_saver.py保持一致）
    
    def detect_all(self, image, camera_intrinsics=None):
        """
        检测图像中的所有球体（使用轮廓检测）
        
        重要：输入图像必须已经过深度mask处理！
        即：深度>1m的区域应该已经被设置为黑色(0,0,0)
        
        Args:
            image: BGR图像（必须已经过深度mask处理，黑色区域为深度>阈值的区域）
            camera_intrinsics: CameraIntrinsics对象（可选，本检测器不使用）
            
        Returns:
            list of detections: 每个元素包含检测信息字典
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化（检测非黑色区域）
        # 阈值设为1，将所有非完全黑色的像素识别为前景
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 形态学操作，去除噪声
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.close_kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 收集所有符合条件的球体
        detections = []
        
        for contour in contours:
            # 面积过滤
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # 计算圆形度（越接近 1 越圆）
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 如果圆形度足够高，认为是球体
            if circularity > self.min_circularity:
                # 获取最小外接圆的半径
                _, radius = cv2.minEnclosingCircle(contour)
                
                # 找边界点
                contour_points = contour.reshape(-1, 2)
                
                leftmost_idx = contour_points[:, 0].argmin()
                leftmost = tuple(contour_points[leftmost_idx])
                
                rightmost_idx = contour_points[:, 0].argmax()
                rightmost = tuple(contour_points[rightmost_idx])
                
                topmost_idx = contour_points[:, 1].argmin()
                topmost = tuple(contour_points[topmost_idx])
                
                bottommost_idx = contour_points[:, 1].argmax()
                bottommost = tuple(contour_points[bottommost_idx])
                
                hori_line = (leftmost, rightmost)
                vert_line = (topmost, bottommost)
                
                # 计算角平分线交点（球心）
                if camera_intrinsics is not None:
                    # X方向角平分线
                    left_x_norm = (leftmost[0] - camera_intrinsics.cx) / camera_intrinsics.fx
                    right_x_norm = (rightmost[0] - camera_intrinsics.cx) / camera_intrinsics.fx
                    
                    ray_left = np.array([left_x_norm, 0, -1])
                    ray_right = np.array([right_x_norm, 0, -1])
                    ray_left = ray_left / np.linalg.norm(ray_left)
                    ray_right = ray_right / np.linalg.norm(ray_right)
                    
                    bisector_xz = ray_left + ray_right
                    bisector_xz = bisector_xz / np.linalg.norm(bisector_xz)
                    
                    t_x = -1.0 / bisector_xz[2]
                    bisector_x_cam = t_x * bisector_xz[0]
                    bisector_x_pixel = bisector_x_cam * camera_intrinsics.fx + camera_intrinsics.cx
                    
                    # Y方向角平分线
                    top_y_norm = (topmost[1] - camera_intrinsics.cy) / camera_intrinsics.fy
                    bottom_y_norm = (bottommost[1] - camera_intrinsics.cy) / camera_intrinsics.fy
                    
                    ray_top = np.array([0, top_y_norm, -1])
                    ray_bottom = np.array([0, bottom_y_norm, -1])
                    ray_top = ray_top / np.linalg.norm(ray_top)
                    ray_bottom = ray_bottom / np.linalg.norm(ray_bottom)
                    
                    bisector_yz = ray_top + ray_bottom
                    bisector_yz = bisector_yz / np.linalg.norm(bisector_yz)
                    
                    t_y = -1.0 / bisector_yz[2]
                    bisector_y_cam = t_y * bisector_yz[1]
                    bisector_y_pixel = bisector_y_cam * camera_intrinsics.fy + camera_intrinsics.cy
                    
                    intersection = (int(bisector_x_pixel), int(bisector_y_pixel))
                    center = intersection
                else:
                    # 如果没有相机内参，回退到简单的中点计算
                    x = int((leftmost[0] + rightmost[0]) / 2)
                    y = int((topmost[1] + bottommost[1]) / 2)
                    center = (x, y)
                    intersection = center
                
                # 计算综合评分（用于兼容性）
                score = circularity
                
                # 添加检测结果
                detections.append({
                    'center': center,
                    'radius': int(radius),
                    'contour': contour,
                    'vert_line': vert_line,
                    'hori_line': hori_line,
                    'intersection': intersection,
                    'score': score,
                    'area': area
                })
        
        return detections


class BallTracker:
    """
    多球追踪器 - 使用改进的匈牙利算法进行数据关联
    为每个球维护独立的卡尔曼滤波器
    
    重要概念：
    - Tracker ID: 卡尔曼滤波器的索引 (0, 1, 2, ...)
    - GT Ball ID: 真实球的索引 (0, 1, 2, ...)
    - 映射关系: gt_binding[tracker_id] = gt_ball_id
    
    数据关联策略（综合代价矩阵）：
    1. 位置距离：预测位置与检测位置的欧氏距离
    2. 速度大小：预测速度与估计速度的大小差异
    3. 速度方向：预测速度与估计速度的方向差异（角度）
    4. 历史轨迹一致性：当前检测方向与历史移动方向的一致性
    
    权重可调：
    - w_position: 位置权重（默认1.0）
    - w_velocity_mag: 速度大小权重（默认0.3）
    - w_velocity_dir: 速度方向权重（默认0.3）
    - w_history: 历史一致性权重（默认0.2）
    """
    
    def __init__(self, num_balls=3, dt=0.002, g=9.8093, 
                 process_noise=0.0001, measurement_noise=0.03,
                 max_distance=0.5, drag_coefficient=0.0, verbose=False):
        """
        初始化多球追踪器
        
        Args:
            num_balls: 球的数量
            dt: 时间步长
            g: 重力加速度
            process_noise: 过程噪声标准差
            measurement_noise: 测量噪声标准差
            max_distance: 最大匹配距离（米），超过此距离认为不匹配
            drag_coefficient: 空气阻力系数（默认0.0表示无阻力）
            verbose: 是否打印详细日志
        """
        self.num_balls = num_balls
        self.dt = dt
        self.max_distance = max_distance
        self.verbose = verbose
        
        # 创建球检测器
        self.detector = MultiRedBallDetector()
        
        # 为每个球创建独立的卡尔曼滤波器
        self.kf_filters = [
            KalmanFilter3D(dt=dt, g=g, 
                          process_noise=process_noise, 
                          measurement_noise=measurement_noise,
                          drag_coefficient=drag_coefficient,
                          verbose=verbose)
            for _ in range(num_balls)
        ]
        
        # 追踪状态（通过 kf_filters[i].initialized 判断球是否正在追踪）
        self.ball_grounded = [False] * num_balls  # 球是否已经落地（一旦落地就不再更新）
        
        # 首次观测验证状态
        self.consecutive_detections = [0] * num_balls  # 连续检测计数器
        self.ever_validated = [False] * num_balls      # 是否曾经通过5帧验证（True=已验证，False=第一次观测需验证）
        self.required_detections = 2  # 首次观测需要连续检测的帧数
        
        # 历史信息（用于匹配）
        self.position_history = [[] for _ in range(num_balls)]  # 位置历史
        self.velocity_history = [[] for _ in range(num_balls)]  # 速度历史
        self.history_length = 10  # 保留的历史长度
        
        # 轨迹数据（用于可视化）
        self.kf_trajectories = [[] for _ in range(num_balls)]  # 卡尔曼滤波轨迹
        self.kf_velocity = [[] for _ in range(num_balls)]  # 卡尔曼滤波速度
        self.detected_trajectories = [[] for _ in range(num_balls)]  # 检测轨迹
        self.max_trajectory_length = 1000  # 最大轨迹长度
        
    def predict_all(self, ground_z_threshold=0.15):
        """
        对所有激活的滤波器执行预测步骤，并检测落地
        
        Args:
            ground_z_threshold: 地面高度阈值（米），低于此高度认为球已落地
        """
        start_time = time.perf_counter() if self.verbose else None
        active_count = 0
        
        for tracker_id in range(self.num_balls):
            # 跳过已落地或未初始化的追踪器
            if self.ball_grounded[tracker_id] or not self.kf_filters[tracker_id].initialized:
                continue
            
            active_count += 1
            # 正常预测
            self.kf_filters[tracker_id].predict()
            
            # 预测后检查是否落地（使用tracker自己的预测位置）
            if self.ever_validated[tracker_id]:  # 只对已验证的追踪器检测落地
                state = self.get_state(tracker_id)
                if state and state['position'] is not None:
                    predicted_pos = state['position']
                    if predicted_pos[2] < ground_z_threshold:
                        # 标记为已落地
                        self.ball_grounded[tracker_id] = True
                        print(f"[落地检测] 追踪器{tracker_id}检测到落地（预测z={predicted_pos[2]:.3f}m < {ground_z_threshold}m）") if tracker_id == 2 else None
        
        if self.verbose and active_count > 0:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"[Tracker predict_all] 预测{active_count}个追踪器，总耗时: {elapsed_time:.4f}ms，平均: {elapsed_time/active_count:.4f}ms/tracker")
    
    def _match_detections(self, detections):
        """
        仅执行数据关联（匹配），不更新KF状态
        
        Args:
            detections: 检测到的球的3D位置列表 [(x,y,z), ...]
        
        Returns:
            assignments: 分配结果 {ball_id: detection_index}
        """
        detections = [np.array(d) for d in detections]
        num_detections = len(detections)
        
        if num_detections == 0:
            return {}
        
        # 构建代价矩阵（与update方法相同的逻辑）
        cost_matrix = np.zeros((self.num_balls, num_detections))
        
        for tracker_id in range(self.num_balls):
            if self.kf_filters[tracker_id].initialized:
                state = self.kf_filters[tracker_id].get_state()
                predicted_pos = state['position']
                
                for det_idx in range(num_detections):
                    position_cost = np.linalg.norm(predicted_pos - detections[det_idx])
                    cost_matrix[tracker_id, det_idx] = position_cost
            else:
                for det_idx in range(num_detections):
                    cost_matrix[tracker_id, det_idx] = self.max_distance
        
        # 使用匈牙利算法求解最优分配
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 处理分配结果
        assignments = {}
        for tracker_id, det_idx in zip(row_indices, col_indices):
            if self.ball_grounded[tracker_id]:
                continue
            if cost_matrix[tracker_id, det_idx] <= self.max_distance:
                assignments[tracker_id] = det_idx
        
        return assignments
    
    def update(self, detections):
        """
        更新追踪器（使用匈牙利算法进行数据关联）
        
        Args:
            detections: 检测到的球的3D位置列表 [(x,y,z), ...]
            gt_positions: Ground truth位置列表（用于首次绑定）
        
        Returns:
            assignments: 分配结果 {ball_id: detection_index}
        """
        start_time = time.perf_counter() if self.verbose else None
        
        detections = [np.array(d) for d in detections]
        num_detections = len(detections)
        
        if num_detections == 0:
            # 没有检测到球，保持追踪状态（继续预测）
            print("[更新] 未检测到球，保持当前追踪状态")
            return {}
        
        # 估计检测点之间的速度（改进版：使用卡尔曼滤波器的速度预测）
        detection_velocities = []
        for det_idx in range(num_detections):
            # 方法改进：使用最近的激活追踪器的预测速度作为初始估计
            # 而不是通过历史位置差分计算（容易出错）
            best_velocity = None
            min_dist = float('inf')
            
            for tracker_id in range(self.num_balls):
                if self.kf_filters[tracker_id].initialized:
                    state = self.kf_filters[tracker_id].get_state()
                    predicted_pos = state['position']
                    predicted_vel = state['velocity']
                    
                    dist = np.linalg.norm(detections[det_idx] - predicted_pos)
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        # 使用预测速度，而不是差分速度
                        best_velocity = predicted_vel
            
            detection_velocities.append(best_velocity)
        
        # 构建综合代价矩阵
        cost_matrix = np.zeros((self.num_balls, num_detections))
        
        # 权重参数
        # 关键思路：先用位置做粗匹配，然后用速度等作为修正项
        # 代价权重配置
        w_position = 1.0    # 位置代价权重
        w_speed = 0.000       # 速度大小代价权重
        w_direction = 0.05   # 速度方向代价权重

        for tracker_id in range(self.num_balls):
            if self.kf_filters[tracker_id].initialized:
                # 获取预测状态
                state = self.kf_filters[tracker_id].get_state()
                predicted_pos = state['position']
                predicted_vel = state['velocity']
                predicted_speed = np.linalg.norm(predicted_vel)
                
                for det_idx in range(num_detections):
                    # 1. 位置距离代价（主导项）
                    position_cost = np.linalg.norm(predicted_pos - detections[det_idx])
                    
                    # 2. 速度代价（分解为速度大小和方向）
                    speed_cost = 0.0
                    direction_cost = 0.0
                    
                    if detection_velocities[det_idx] is not None:
                        detected_vel = detection_velocities[det_idx]
                        detected_speed = np.linalg.norm(detected_vel)
                        
                        # 2a. 速度大小代价（绝对差值）
                        speed_cost = abs(predicted_speed - detected_speed)
                        
                        # 2b. 速度方向代价（使用余弦距离）
                        # 避免除零：当速度很小时方向不可靠
                        if predicted_speed > 1e-3 and detected_speed > 1e-3:
                            # 归一化速度向量
                            predicted_dir = predicted_vel / predicted_speed
                            detected_dir = detected_vel / detected_speed
                            # 余弦相似度：[-1, 1]，1表示方向相同，-1表示方向相反
                            cos_similarity = np.dot(predicted_dir, detected_dir)
                            # 转换为代价：0表示方向相同，2表示方向相反
                            direction_cost = 1.0 - cos_similarity
                        else:
                            # 速度太小时，方向代价设为0（不惩罚）
                            direction_cost = 0.0
                    
                    # 综合代价（加权求和）
                    total_cost = (w_position * position_cost + 
                                  w_speed * speed_cost +
                                  w_direction * direction_cost
                                 )
                    
                    cost_matrix[tracker_id, det_idx] = total_cost
                    
            else:
                # 未初始化的追踪器：使用中等代价，允许初始化
                for det_idx in range(num_detections):
                    cost_matrix[tracker_id, det_idx] = self.max_distance
        
        # 使用匈牙利算法求解最优分配
        row_indices, col_indices = linear_sum_assignment(cost_matrix) #row_indices: 追踪器ID, col_indices: 检测ID
        
        # 记录哪些追踪器被成功匹配
        matched_trackers = set()
        for tracker_id, det_idx in zip(row_indices, col_indices):
            if cost_matrix[tracker_id, det_idx] <= self.max_distance:
                matched_trackers.add(tracker_id)
        
        # 处理未匹配的追踪器：重置首次观测验证状态
        for tracker_id in range(self.num_balls):
            if tracker_id not in matched_trackers and not self.ever_validated[tracker_id]:
                # 首次观测期间检测中断，重置
                if self.consecutive_detections[tracker_id] > 0:
                    print(f"[验证中断] 追踪器{tracker_id}首次观测期间检测中断（已检测{self.consecutive_detections[tracker_id]}帧），重置")
                    self.consecutive_detections[tracker_id] = 0
                    self.kf_filters[tracker_id].reset()
        
        # 处理分配结果
        assignments = {}
        unassigned_detections = set(range(num_detections))
        update_count = 0
        
        for tracker_id, det_idx in zip(row_indices, col_indices):
            # 已经落地的球不再更新
            if self.ball_grounded[tracker_id]:
                continue
                
            # 检查距离是否在阈值内
            if cost_matrix[tracker_id, det_idx] <= self.max_distance:
                # 分配成功
                assignments[tracker_id] = det_idx
                unassigned_detections.discard(det_idx)
                
                # 区分首次观测和后续观测
                if not self.ever_validated[tracker_id]:
                    # 首次观测：需要连续N帧验证
                    self.consecutive_detections[tracker_id] += 1
                    
                    if self.consecutive_detections[tracker_id] >= self.required_detections:
                        # 达到验证阈值，真正初始化卡尔曼滤波器
                        print(f"[验证成功] 追踪器{tracker_id}连续检测{self.consecutive_detections[tracker_id]}帧，开始初始化卡尔曼滤波器")
                        self.kf_filters[tracker_id].update(detections[det_idx])
                        self.ever_validated[tracker_id] = True
                        update_count += 1
                    else:
                        # 未达到阈值，仅累加计数，不更新滤波器
                        # print(f"[验证中] 追踪器{tracker_id}连续检测{self.consecutive_detections[tracker_id]}/{self.required_detections}帧")
                        pass
                else:
                    # 已验证过，正常更新卡尔曼滤波器
                    self.kf_filters[tracker_id].update(detections[det_idx])
                    update_count += 1

            else:
                pass
        
        if self.verbose:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"[Tracker update] 更新{update_count}个追踪器，总耗时: {elapsed_time:.4f}ms")
        return assignments
    
    def get_state(self, tracker_id):
        """获取指定追踪器的状态"""
        if 0 <= tracker_id < self.num_balls:
            return self.kf_filters[tracker_id].get_state()
        return None
    
    def get_all_states(self):
        """获取所有追踪器的状态"""
        return [self.get_state(i) for i in range(self.num_balls)]

    def get_landing_prediction(self, tracker_id, z_threshold=0.15, min_updates=5, max_velocity_uncertainty=100.0, verbose=False):
        """
        获取指定追踪器的落地预测位置
        
        Args:
            tracker_id: 追踪器的ID
            z_threshold: 地面高度阈值
            min_updates: 最小更新次数
            max_velocity_uncertainty: 最大速度不确定性
            verbose: 是否打印详细日志
        
        Returns:
            landing_pos: 落地位置 [x, y, z] 或 None
            landing_time: 落地时间（秒）或 None
        """
        start_time = time.perf_counter() if verbose else None
        if 0 <= tracker_id < self.num_balls:
            if self.kf_filters[tracker_id].initialized and self.ever_validated[tracker_id]:
                landing_pos, landing_time = self.kf_filters[tracker_id].predict_landing_position(
                    z_threshold, min_updates, max_velocity_uncertainty
                )
                if verbose:
                    elapsed_time = (time.perf_counter() - start_time) * 1000
                    print(f"[获取落地预测] Tracker{tracker_id} 落地位置: {landing_pos}, 落地时间: {landing_time}, 耗时: {elapsed_time:.4f}ms")
                return landing_pos, landing_time
        return None, None
    
    def get_all_landing_predictions(self, z_threshold=0.15, min_updates=5, max_velocity_uncertainty=100.0):
        """
        获取所有追踪器的落地预测位置
        
        Args:
            z_threshold: 地面高度阈值
            min_updates: 最小更新次数
            max_velocity_uncertainty: 最大速度不确定性
        
        Returns:
            list of (landing_pos, landing_time) tuples
        """
        predictions = []
        for i in range(self.num_balls):
            landing_pos, landing_time = self.get_landing_prediction(
                i, z_threshold, min_updates, max_velocity_uncertainty, verbose=self.verbose
            )
            predictions.append((landing_pos, landing_time))
        return predictions
    
    def is_active(self, tracker_id):
        """检查追踪器是否正在追踪（通过卡尔曼滤波器的初始化状态判断）"""
        if 0 <= tracker_id < self.num_balls:
            return self.kf_filters[tracker_id].initialized
        return False
    
    def is_validated(self, tracker_id):
        """检查追踪器是否已通过首次观测验证（连续检测）"""
        if 0 <= tracker_id < self.num_balls:
            return self.ever_validated[tracker_id]
        return False
    

    
    def is_grounded(self, tracker_id):
        """检查追踪器追踪的球是否已落地（一旦落地就不再更新）"""
        if 0 <= tracker_id < self.num_balls:
            return self.ball_grounded[tracker_id]
        return False
    
    def record_kf_trajectory(self, tracker_id, position, velocity):
        """记录卡尔曼滤波轨迹和速度"""
        if 0 <= tracker_id < self.num_balls:
            self.kf_trajectories[tracker_id].append(position.copy())
            self.kf_velocity[tracker_id].append(velocity.copy())
            
            # 限制轨迹长度
            if len(self.kf_trajectories[tracker_id]) > self.max_trajectory_length:
                self.kf_trajectories[tracker_id].pop(0)
                self.kf_velocity[tracker_id].pop(0)
    
    def record_detection(self, tracker_id, position):
        """记录检测位置"""
        if 0 <= tracker_id < self.num_balls:
            self.detected_trajectories[tracker_id].append(position.copy())
            
            # 限制轨迹长度
            if len(self.detected_trajectories[tracker_id]) > self.max_trajectory_length:
                self.detected_trajectories[tracker_id].pop(0)
    
    def clear_trajectories(self, tracker_id):
        """清空指定追踪器的所有轨迹"""
        if 0 <= tracker_id < self.num_balls:
            self.kf_trajectories[tracker_id].clear()
            self.kf_velocity[tracker_id].clear()
            self.detected_trajectories[tracker_id].clear()
    
    def reset(self):
        """重置所有追踪器"""
        for tracker_id in range(self.num_balls):
            self.kf_filters[tracker_id].reset()  # 自动设置 initialized=False
            self.ball_grounded[tracker_id] = False  # 重置落地状态
            self.consecutive_detections[tracker_id] = 0  # 重置连续检测计数
            self.ever_validated[tracker_id] = False  # 重置验证状态
            self.clear_trajectories(tracker_id)  # 清空轨迹
    
    def reset_ball(self, tracker_id):
        """重置单个追踪器"""
        if 0 <= tracker_id < self.num_balls:
            self.kf_filters[tracker_id].reset()  # 自动设置 initialized=False
            self.ball_grounded[tracker_id] = False  # 重置落地状态
            self.consecutive_detections[tracker_id] = 0  # 重置连续检测计数
            self.ever_validated[tracker_id] = False  # 重置验证状态
            self.clear_trajectories(tracker_id)  # 清空轨迹
    
    def record_prediction_states(
        self,
        base_site_rot,
        base_site_pos,
        kf_obs,
        kf_obs_body,
        max_velocity_uncertainty
    ):
        """
        记录预测状态到轨迹和观测字典
        
        Args:
            base_site_rot: 基座旋转矩阵
            base_site_pos: 基座位置
            kf_obs: 卡尔曼滤波观测字典（世界坐标系）
            kf_obs_body: 卡尔曼滤波观测字典（体坐标系）
            max_velocity_uncertainty: 最大速度不确定性阈值
        """
        import numpy as np
        
        for tracker_id in range(self.num_balls):
            if self.is_validated(tracker_id) and not self.is_grounded(tracker_id):
                state = self.get_state(tracker_id)
                if state and state['position'] is not None and state['velocity'] is not None:
                    self.record_kf_trajectory(tracker_id, state['position'], state['velocity'])
                    
                    # 判断是否使用速度（基于不确定性）
                    use_velocity = np.any(state['velocity_uncertainty'] < max_velocity_uncertainty)
                    velocity_to_store = state['velocity'].copy() if use_velocity else np.array([0.0, 0.0, 0.0])
                    
                    kf_obs[tracker_id] = {
                        'position': state['position'].copy(),
                        'velocity': velocity_to_store
                    }
                    
                    # 转换到体坐标系
                    position_body = np.dot(base_site_rot.T, state['position'] - base_site_pos)
                    velocity_body = np.dot(base_site_rot.T, state['velocity'])
                    kf_obs_body[tracker_id] = {
                        'position': position_body.copy(),
                        'velocity': velocity_body.copy()
                    }
    
    def cleanup_grounded_balls(self, kf_obs, kf_obs_body):
        """
        清理已落地的球，重置追踪器和清空轨迹
        
        Args:
            kf_obs: 卡尔曼滤波观测字典
            kf_obs_body: 卡尔曼滤波观测字典（体坐标系）
        """
        for tracker_id in range(self.num_balls):
            if self.is_grounded(tracker_id):
                # 清空观测
                kf_obs[tracker_id] = None
                kf_obs_body[tracker_id] = None
                
                # 重置追踪器（包括清空轨迹）
                self.reset_ball(tracker_id)
    
    @staticmethod
    def get_camera_extrinsics(m, d, camera_name, use_api=False, body_pos=None):
        """获取相机的外参（位置和姿态）
        
        Args:
            m: MuJoCo模型
            d: MuJoCo数据
            camera_name: 相机名称
            use_api: 如果为True，使用MuJoCo API直接获取
            body_pos: 体位置（仅在use_api=False时使用）
        
        Returns:
            position: 相机在世界坐标系中的位置
            rotation_matrix: 3x3 旋转矩阵（相机坐标系 -> 世界坐标系）
        """
        import mujoco
        from scipy.spatial.transform import Rotation as R
        
        camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"相机 '{camera_name}' 不存在")
        
        mujoco.mj_forward(m, d)
        
        if use_api:
            cam_pos_world = d.cam_xpos[camera_id].copy()
            cam_rot_world = d.cam_xmat[camera_id].reshape(3, 3).copy()
        else:
            if body_pos is None:
                # 使用IMU传感器数据重建相机父体的位姿
                body_pos = d.sensor("position").data.copy()
            forward = d.sensor("forwardvector").data.copy()
            up = d.sensor("upvector").data.copy()
            left = np.cross(up, forward)   # y轴 = z叉乘x（右手系）
            body_mat = np.stack([forward, left, up], axis=1)  # 列为体坐标系各轴

            # Camera pose in parent body frame (torso_link origin)
            cam_pos_body = m.cam_pos[camera_id].copy()
            cam_quat_body = m.cam_quat[camera_id].copy()  # [w, x, y, z]
            cam_rot_body = R.from_quat([cam_quat_body[1], cam_quat_body[2],
                                        cam_quat_body[3], cam_quat_body[0]]).as_matrix()

            # IMU site pose in the same parent body frame (torso_link origin)
            imu_site_id = m.site("imu").id
            imu_pos_body = m.site_pos[imu_site_id].copy()
            imu_quat_body = m.site_quat[imu_site_id].copy()  # [w, x, y, z]
            imu_rot_body = R.from_quat([imu_quat_body[1], imu_quat_body[2],
                                        imu_quat_body[3], imu_quat_body[0]]).as_matrix()

            # Camera pose expressed in IMU frame
            cam_pos_local = imu_rot_body.T @ (cam_pos_body - imu_pos_body)
            cam_rot_local = imu_rot_body.T @ cam_rot_body
            
            # Reconstruct world pose using IMU world position/orientation from sensors
            cam_pos_world = body_pos + body_mat @ cam_pos_local
            cam_rot_world = body_mat @ cam_rot_local
        
        return cam_pos_world, cam_rot_world
    
    @staticmethod
    def get_valid_depth(depth_image, x, y, radius=2):
        """
        获取有效的深度值，如果中心点深度无效，则在周围搜索
        
        Args:
            depth_image: 深度图像
            x, y: 中心点像素坐标
            radius: 搜索半径（像素）
            
        Returns:
            有效的深度值，如果找不到则返回None
        """
        # 检查中心点深度
        center_depth = depth_image[y, x]
        if not np.isnan(center_depth) and center_depth > 0 and center_depth < 1.0:
            return center_depth
        
        # 中心点深度无效，在周围搜索
        valid_depths = []
        h, w = depth_image.shape
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                # 检查边界
                if 0 <= ny < h and 0 <= nx < w:
                    depth_val = depth_image[ny, nx]
                    # 检查是否有效
                    if not np.isnan(depth_val) and depth_val > 0 and depth_val < 10.0:
                        valid_depths.append(depth_val)
        
        if len(valid_depths) == 0:
            return None
        
        # 剔除离群值：使用IQR方法
        if len(valid_depths) >= 4:
            valid_depths = np.array(valid_depths)
            q1 = np.percentile(valid_depths, 25)
            q3 = np.percentile(valid_depths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # 过滤离群值
            filtered_depths = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]
            if len(filtered_depths) > 0:
                return np.mean(filtered_depths)
        
        # 深度值太少，直接取均值
        return np.mean(valid_depths)
    
    def detect_and_localize_balls(self, rgb_image, depth_image, 
                                camera_intrinsics, camera_pos, camera_rot):
        """检测并重建所有球的3D位置

        Args:
            rgb_image: RGB图像（BGR格式）
            depth_image: 深度图
            camera_intrinsics: CameraIntrinsics对象
            camera_pos: 相机位置（世界坐标系）
            camera_rot: 相机旋转矩阵

        Returns:
            list of (ball_position_world, detection_info, ray_info)
        """
        # 检测所有球
        detections = self.detector.detect_all(rgb_image, camera_intrinsics)

        if not detections:
            return []

        results = []
        ball_radius = 0.0375  # 米

        for det in detections:
            center = det['center']
            x, y = center
            
            # 检查边界
            if y >= depth_image.shape[0] or x >= depth_image.shape[1]:
                continue
            
            # 获取深度（支持NaN处理）
            depth_surface = self.get_valid_depth(depth_image, x, y, radius=2)
            if depth_surface is None:
                continue
            
            # 球半径补偿
            ray_direction = camera_intrinsics.pixel_to_camera_ray(x, y)
            actual_ray_length = abs(depth_surface / ray_direction[2]) + ball_radius
            point_cam = actual_ray_length * ray_direction 
            
            # 转换到世界坐标系
            point_world = camera_pos + camera_rot @ point_cam
            
            # 射线信息
            ray_info = {
                'center': center,
                'ray_direction': ray_direction.copy(),
                'depth': depth_surface,
                'point_cam': point_cam.copy(),
                'actual_ray_length': actual_ray_length
            }
            
            results.append((point_world, det, ray_info))
            print(f"[检测] 球位置（世界坐标系）: {point_world}, 深度: {depth_surface:.3f}m, 2D中心: {center}, 相机位置: {camera_pos}, 相机旋转矩阵: {camera_rot}") 
        return results
    
    def process_detection_and_update(
        self,
        rgb_bgr,
        depth_array,
        camera_intrinsics,
        cam_pos,
        cam_mat,
        base_site_rot,
        base_site_pos,
        kf_obs,
        kf_obs_body,
        max_velocity_uncertainty,
        test_mode_enabled=False,
        upgrade_counter=None,
        max_upgrades_after_valid=10
    ):
        """
        处理球检测、追踪更新和状态维护的完整流程
        
        Args:
            rgb_bgr: BGR格式的RGB图像
            depth_array: 深度图像
            camera_intrinsics: 相机内参
            cam_pos: 相机位置（世界坐标系）
            cam_mat: 相机旋转矩阵
            base_site_rot: 基座旋转矩阵
            base_site_pos: 基座位置
            kf_obs: 卡尔曼滤波观测字典（世界坐标系）
            kf_obs_body: 卡尔曼滤波观测字典（体坐标系）
            max_velocity_uncertainty: 最大速度不确定性阈值
            test_mode_enabled: 是否启用测试模式
            upgrade_counter: 每个追踪器的upgrade计数器（数组）
            max_upgrades_after_valid: 验证后最大upgrade次数
            
        Returns:
            has_detection: dict，记录每个追踪器是否有检测
            detection_results: 检测结果列表 [(pos_world, det_info, ray_info), ...]
            actually_updated: dict，记录每个追踪器是否真正执行了update操作
        """
        import numpy as np
        
        # 检测并定位所有球
        detection_results = self.detect_and_localize_balls(
            rgb_bgr, depth_array,
            camera_intrinsics, cam_pos, cam_mat
        )
        
        # 提取3D位置
        detected_positions = [pos for pos, _, _ in detection_results]
        
        # 初始化检测标记和实际更新标记
        has_detection = {}
        actually_updated = {}
        for tracker_id in range(self.num_balls):
            has_detection[tracker_id] = False
            actually_updated[tracker_id] = False
        
        if detected_positions:
            # 步骤1: 先进行匹配（数据关联），但不更新KF
            # 我们需要自己实现匹配逻辑，而不是直接调用self.update()
            # 因为self.update()会自动执行KF的update操作
            
            # 执行数据关联（匹配）
            assignments = self._match_detections(detected_positions)
            
            # 步骤2: 根据test_mode决定是否真正执行update
            for tracker_id, det_idx in assignments.items():
                has_detection[tracker_id] = True
                
                # 判断是否应该执行update
                should_update = True
                if test_mode_enabled and upgrade_counter is not None:
                    # test模式下，检查upgrade_counter
                    if self.is_validated(tracker_id):
                        if upgrade_counter[tracker_id] >= max_upgrades_after_valid:
                            should_update = False
                
                # 根据should_update决定操作
                if should_update:
                    # 执行真正的update操作
                    if not self.ever_validated[tracker_id]:
                        # 首次观测验证逻辑
                        self.consecutive_detections[tracker_id] += 1
                        if self.consecutive_detections[tracker_id] >= self.required_detections:
                            self.kf_filters[tracker_id].update(detected_positions[det_idx])
                            self.ever_validated[tracker_id] = True
                            actually_updated[tracker_id] = True
                    else:
                        # 已验证，正常更新
                        self.kf_filters[tracker_id].update(detected_positions[det_idx])
                        actually_updated[tracker_id] = True
                else:
                    # 不执行update，只保持predict状态
                    pass
                
                # 只有已验证的追踪器才处理状态记录
                if self.is_validated(tracker_id):
                    self.record_detection(tracker_id, detected_positions[det_idx])
                    
                    # 获取当前状态（可能是update后的，也可能是predict后的）
                    state = self.get_state(tracker_id)
                    if state and state['position'] is not None and state['velocity'] is not None:
                        # 转换到体坐标系
                        position_body = np.dot(base_site_rot.T, state['position'] - base_site_pos)
                        velocity_body = np.dot(base_site_rot.T, state['velocity'])
                        
                        # 判断是否使用速度（基于不确定性）
                        use_velocity = np.any(state['velocity_uncertainty'] < max_velocity_uncertainty)
                        velocity_to_store = state['velocity'].copy() if use_velocity else np.array([0.0, 0.0, 0.0])
                        
                        # 替换最后一个预测位置为当前状态
                        if (len(self.kf_trajectories[tracker_id]) > 0 and 
                            len(self.kf_velocity[tracker_id]) > 0 and 
                            len(kf_obs) > 0):
                            self.kf_trajectories[tracker_id][-1] = state['position'].copy()
                            self.kf_velocity[tracker_id][-1] = state['velocity'].copy()
                            kf_obs[tracker_id] = {
                                'position': state['position'].copy(),
                                'velocity': velocity_to_store
                            }
                            kf_obs_body[tracker_id] = {
                                'position': position_body.copy(),
                                'velocity': velocity_body.copy()
                            }
                        else:
                            # 首次记录
                            self.record_kf_trajectory(tracker_id, state['position'], state['velocity'])
                            kf_obs[tracker_id] = {
                                'position': state['position'].copy(),
                                'velocity': velocity_to_store
                            }
                            kf_obs_body[tracker_id] = {
                                'position': position_body.copy(),
                                'velocity': velocity_body.copy()
                            }
            
            # 处理未匹配的追踪器（重置首次观测验证状态）
            matched_trackers = set(assignments.keys())
            for tracker_id in range(self.num_balls):
                if tracker_id not in matched_trackers and not self.ever_validated[tracker_id]:
                    if self.consecutive_detections[tracker_id] > 0:
                        print(f"[验证中断] 追踪器{tracker_id}首次观测期间检测中断，重置")
                        self.consecutive_detections[tracker_id] = 0
                        self.kf_filters[tracker_id].reset()
        
        return has_detection, detection_results, actually_updated
    
    @staticmethod
    def catch_info_from_kf_obs_body(kf_obs_body):
        """
        从 kf_obs_body 中提取要接的球的信息（单手版本）
        
        选择逻辑:
        - 优先选速度向下 (vz < 0) 的球中 z 位置最低的
        - 若无速度向下的球, 则选所有有效球中 z 最低的
        
        Args:
            kf_obs_body: list of dict, 每个元素包含 'position' 和 'velocity' (体坐标系)
        
        Returns:
            catch_info: dict with 'position' and 'velocity', 或 None
        """
        # 收集有效的球信息
        valid_balls = []
        for tracker_id, obs in enumerate(kf_obs_body):
            if obs is not None and 'position' in obs and 'velocity' in obs:
                pos = obs['position']
                vel = obs['velocity']
                if pos is not None and vel is not None:
                    valid_balls.append({
                        'tracker_id': tracker_id,
                        'position': pos.copy(),
                        'velocity': vel.copy()
                    })

        if len(valid_balls) == 0:
            return None

        # 优先选速度向下(vz<0)的球，若无则用全部
        falling = [b for b in valid_balls if b['velocity'][2] < 0]
        pool = falling if len(falling) > 0 else valid_balls
        # 在 pool 中选 z 位置最低的球
        best_ball = min(pool, key=lambda b: b['position'][2])

        return {
            'position': best_ball['position'].copy(),
            'velocity': best_ball['velocity'].copy()
        }


class BallTrackingVisualizer:
    """
    球追踪可视化器 - 使用 matplotlib 3D 图表显示追踪结果
    不再内部存储轨迹数据，而是从BallTracker接收轨迹数据进行可视化
    """
    

    
    def __init__(self, num_balls=3):
        """
        初始化可视化器
        
        Args:
            num_balls: 球的数量
        """
        self.num_balls = num_balls
        
        # 设置 matplotlib
        import matplotlib.pyplot as plt
        plt.ion()  # 开启交互模式
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Ball Tracking: Ground Truth vs Kalman Filter')
        
        # 设置固定的视图范围
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 1.5])
        
        # 颜色
        self.colors = ['red', 'green', 'blue']
        
        # 创建轨迹线和标记
        self.kf_lines = []
        self.detected_lines = []
        self.kf_current_points = []
        self.error_texts = []
        self.velocity_texts = []
        
        for i in range(num_balls):
            # KF轨迹线 (实线)
            kf_line, = self.ax.plot([], [], [], c=self.colors[i], linewidth=3, 
                               label=f'Tracker {i}', alpha=0.8, linestyle='-')
            self.kf_lines.append(kf_line)
            
            # KF当前位置 (实心圆)
            kf_point = self.ax.scatter([], [], [], c=self.colors[i], marker='o', s=200, 
                                  alpha=1.0, edgecolors='black', linewidths=2, zorder=10)
            self.kf_current_points.append(kf_point)
            
            # 检测轨迹线 (黑色点线)
            detected_line, = self.ax.plot([], [], [], c='black', linewidth=1.5, 
                               label=f'Detected {i}', alpha=0.6, linestyle=':', marker='x', markersize=4)
            self.detected_lines.append(detected_line)
            
            # 误差文本
            error_text = self.ax.text2D(0.02, 0.98 - i*0.16, '', transform=self.ax.transAxes, 
                                  fontsize=9, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor=self.colors[i], alpha=0.3))
            self.error_texts.append(error_text)
            
            # 速度文本
            velocity_text = self.ax.text2D(0.02, 0.90 - i*0.16, '', transform=self.ax.transAxes, 
                                     fontsize=8, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor=self.colors[i], alpha=0.2))
            self.velocity_texts.append(velocity_text)
        
        self.ax.legend(loc='upper right', fontsize=8, ncol=2)
        self.ax.grid(True, alpha=0.3)
        plt.draw()
        plt.pause(0.001)
    
    @staticmethod
    def visualize_detections_on_image(rgb_image, detection_results):
        """
        在RGB图像上绘制检测结果
        
        Args:
            rgb_image: BGR格式的RGB图像
            detection_results: 检测结果列表 [(pos_world, det_info, ray_info), ...]
            
        Returns:
            绘制后的图像
        """
        import cv2
        
        result_image = rgb_image.copy()
        
        for pos_world, det_info, ray_info in detection_results:
            center = det_info['center']
            contour = det_info['contour']
            vert_line = det_info['vert_line']
            hori_line = det_info['hori_line']
            intersection = det_info['intersection']
            
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
            if vert_line:
                cv2.line(result_image, vert_line[0], vert_line[1], (0, 255, 255), 2)
            if hori_line:
                cv2.line(result_image, hori_line[0], hori_line[1], (255, 0, 255), 2)
            if intersection:
                cv2.circle(result_image, intersection, 4, (0, 0, 255), -1)
        
        return result_image
    
    def update_visualization(self, ball_tracker, has_detection):
        """
        更新可视化
        
        Args:
            ball_tracker: BallTracker实例（从中获取轨迹数据）
            has_detection: 字典，标记每个追踪器是否有检测
        """
        import numpy as np
        
        for tracker_id in range(self.num_balls):
            # 从ball_tracker获取轨迹数据
            detected_traj = ball_tracker.detected_trajectories[tracker_id][-ball_tracker.max_trajectory_length:] if len(ball_tracker.detected_trajectories[tracker_id]) > 0 else []
            if len(detected_traj) > 0:
                detected_array = np.array(detected_traj)
                self.detected_lines[tracker_id].set_data(detected_array[:, 0], detected_array[:, 1])
                self.detected_lines[tracker_id].set_3d_properties(detected_array[:, 2])
            else:
                self.detected_lines[tracker_id].set_data([], [])
                self.detected_lines[tracker_id].set_3d_properties([])
            
            # 更新KF轨迹
            if ball_tracker.is_validated(tracker_id):
                kf_traj = ball_tracker.kf_trajectories[tracker_id][-ball_tracker.max_trajectory_length:] if len(ball_tracker.kf_trajectories[tracker_id]) > 0 else []
         
                if len(kf_traj) > 0:
                    kf_array = np.array(kf_traj)
                    self.kf_lines[tracker_id].set_data(kf_array[:, 0], kf_array[:, 1])
                    self.kf_lines[tracker_id].set_3d_properties(kf_array[:, 2])
                    
                    # 更新KF当前位置
                    kf_pos = kf_traj[-1]
                    self.kf_current_points[tracker_id]._offsets3d = ([kf_pos[0]], [kf_pos[1]], [kf_pos[2]])
                    
                    # 获取速度信息
                    kf_vel = ball_tracker.kf_velocity[tracker_id][-1] if len(ball_tracker.kf_velocity[tracker_id]) > 0 else None
                    
                    # 根据是否有检测显示不同的标签
                    if has_detection.get(tracker_id, False):
                        self.error_texts[tracker_id].set_text(
                            f'Tracker {tracker_id} (detected)'
                        )
                    else:
                        self.error_texts[tracker_id].set_text(
                            f'Tracker {tracker_id} (predict)'
                        )
                    
                    # 显示速度信息
                    if kf_vel is not None:
                        kf_speed = np.linalg.norm(kf_vel)
                        self.velocity_texts[tracker_id].set_text(
                            f'  Vel: {kf_speed:.2f}m/s'
                        )
                    else:
                        self.velocity_texts[tracker_id].set_text('')
                else:
                    self.kf_lines[tracker_id].set_data([], [])
                    self.kf_lines[tracker_id].set_3d_properties([])
                    self.kf_current_points[tracker_id]._offsets3d = ([], [], [])
                    self.error_texts[tracker_id].set_text(f'Tracker {tracker_id}: Not tracked')
                    self.velocity_texts[tracker_id].set_text('')
            else:
                # 追踪器未验证
                self.detected_lines[tracker_id].set_data([], [])
                self.detected_lines[tracker_id].set_3d_properties([])
                self.kf_lines[tracker_id].set_data([], [])
                self.kf_lines[tracker_id].set_3d_properties([])
                self.kf_current_points[tracker_id]._offsets3d = ([], [], [])
                self.error_texts[tracker_id].set_text(f'Tracker {tracker_id}: Validating...')
                self.velocity_texts[tracker_id].set_text('')
        
        # 刷新图表
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass
    
    def close(self):
        """关闭可视化窗口"""
        import matplotlib.pyplot as plt
        plt.close(self.fig)
    
    @staticmethod
    def visualize_camera_axes(viewer, cam_pos, cam_mat, axis_length=0.3, axis_width=0.01):
        """
        使用MuJoCo的可视化API绘制相机坐标轴
        
        Args:
            viewer: MuJoCo viewer实例
            cam_pos: 相机位置
            cam_mat: 相机旋转矩阵
            axis_length: 坐标轴长度（米）
            axis_width: 坐标轴宽度
        """
        import mujoco
        import numpy as np
        
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
            
            # Reset ngeom to base to clear previous frame's camera axis geoms.
            # Debug markers occupy fixed slots 0 and 1 (when active), so start after them.
            cam_geom_base = 2 if getattr(viewer, '_dbg_markers_ready', False) else 0
            viewer.user_scn.ngeom = cam_geom_base

            # Ensure we have space in the scene for new geoms
            if viewer.user_scn.maxgeom >= cam_geom_base + 3:
                # Draw X-axis (Red) - right from camera
                end_pos_x = cam_pos + cam_mat[:, 0] * axis_length
                geom_idx = viewer.user_scn.ngeom
                geom = viewer.user_scn.geoms[geom_idx]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.array([1, 0, 0, 0.8])
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, axis_width, 
                                    cam_pos.reshape(3, 1), end_pos_x.reshape(3, 1))
                viewer.user_scn.ngeom += 1
                
                # Draw Y-axis (Green) - down from camera
                end_pos_y = cam_pos + cam_mat[:, 1] * axis_length
                geom_idx = viewer.user_scn.ngeom
                geom = viewer.user_scn.geoms[geom_idx]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.array([0, 1, 0, 0.8])
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, axis_width,
                                    cam_pos.reshape(3, 1), end_pos_y.reshape(3, 1))
                viewer.user_scn.ngeom += 1
                
                # Draw Z-axis (Blue) - forward from camera (viewing direction)
                end_pos_z = cam_pos + cam_mat[:, 2] * axis_length
                geom_idx = viewer.user_scn.ngeom
                geom = viewer.user_scn.geoms[geom_idx]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.array([0, 0, 1, 0.8])
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, axis_width,
                                    cam_pos.reshape(3, 1), end_pos_z.reshape(3, 1))
                viewer.user_scn.ngeom += 1
