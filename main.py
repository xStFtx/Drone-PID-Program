#For a RQ-1 Predator
import numpy as np
from scipy.spatial.transform import Rotation as R

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class Drone:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.target = np.array([10.0, 10.0, 5.0])
        self.rotation = R.from_quat([0.0, 0.0, 0.0, 1.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.mass = 1000.0  # kg
        self.area = 19.68  # m^2
        self.drag_coefficient = 0.2
        self.lift_coefficient = 0.4
        self.pid_controller = PIDController(0.1, 0.01, 0.1)
        self.time_step = 0.1  # s

    def move(self, direction, wind_speed, wind_direction):
        wind_effect = self.calculate_wind_effect(wind_speed, wind_direction)
        
        if direction == "avoid":
            new_target = self.calculate_3d_path()
            if new_target is not None:
                self.target = new_target

        elif direction == "follow":
            error = self.target - self.position
            control_signal = self.pid_controller.compute(error, self.time_step)

            lift_force = self.calculate_lift_force()
            drag_force = self.calculate_drag_force()

            net_force = lift_force - drag_force + wind_effect + control_signal
            self.acceleration = net_force / self.mass
            
            self.velocity += self.acceleration * self.time_step
            self.position += self.velocity * self.time_step

            rotation_change = R.from_quat([0, 0, control_signal[2], 1])
            self.rotation *= rotation_change

    def calculate_wind_effect(self, wind_speed, wind_direction):
        wind_force = 0.5 * self.drag_coefficient * self.area * (wind_speed ** 2)
        return wind_force * wind_direction

    def calculate_drag_force(self):
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm == 0:
            return np.zeros_like(self.velocity)
        drag_force = 0.5 * self.drag_coefficient * self.area * (velocity_norm ** 2)
        return drag_force * -self.velocity / velocity_norm
    
    def calculate_lift_force(self):
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm == 0:
            return np.zeros_like(self.velocity)
        lift_force = 0.5 * self.lift_coefficient * self.area * (velocity_norm ** 2)
        return lift_force * self.velocity / velocity_norm

    def calculate_3d_path(self):
        # Simulated obstacle
        obstacle = np.array([5.0, 5.0, 5.0])
        dist_to_obstacle = np.linalg.norm(self.position - obstacle)

        if dist_to_obstacle < 2.0:  # If within 2 meters
            return self.target + np.array([1.0, 0.0, 1.0])  # Change the target to avoid the obstacle
        else:
            return None

# Example usage
drone = Drone()
wind_direction = np.array([0.0, 1.0, 0.0])  # Wind is blowing along the Y-axis

for _ in range(100):
    drone.move("follow", 5.0, wind_direction)
    print(f"Position: {drone.position}, Rotation: {drone.rotation.as_euler('xyz', degrees=True)}")
