import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from ament_index_python.packages import get_package_share_directory
import pandas as pd
import numpy as np
import os

drone_ids = ['_1', '_2', '_3', '_4','_5', '_6']
rescue_drones = [ '_7', '_8', '_9', '_10', '_11', '_12']
class MultiDroneCoordinatePublisher(Node):
    def __init__(self):
        super().__init__('drone_publisher')
        
        self.all_drone_ids = drone_ids + rescue_drones
        self.stop_all_drones = False
        self.drone_publishers1 = {drone_id: 0 for drone_id in self.all_drone_ids}
        self.drone_publishers2 = {drone_id: 0 for drone_id in self.all_drone_ids}
        self.dataframes = []
        self.indices = []
        self.store_position = {drone_id: [] for drone_id in drone_ids+rescue_drones}
          
        package_path = get_package_share_directory('swarm_robot')
        csv_folder = os.path.join(package_path, 'hexagon')

        self.last_pos_pub = self.create_publisher(String,'last_position',10) 
        for i, drone_id in enumerate(self.all_drone_ids):
            topic_name1 = f'{drone_id}/drone_coordinates'
            publisher1 = self.create_publisher(String, topic_name1, 10)
            # self.drone_publishers1.append(publisher1)
            self.drone_publishers1[drone_id]=publisher1
            topic_name2 = f'{drone_id}/drone_battery'
            publisher2 = self.create_publisher(Float32, topic_name2, 10)
            self.drone_publishers2[drone_id]=publisher2

            if i<6:
                # To access the csv files
                csv_file = os.path.join(csv_folder, f'drone{drone_id}_NED_coordinates_hex.csv')
                try:
                    df = pd.read_csv(csv_file)
                except FileNotFoundError:
                    self.get_logger().error(f"CSV file not found: {csv_file}")
                    df = pd.DataFrame(columns=['Time (s)', 'North (m)', 'East (m)', 'Down (m)'])

                self.dataframes.append(df)
                self.indices.append(0)
        self.T = self.dataframes[0]['Time (s)']

        self.res_start_pos = {drone_id: [0, -20-i*5, 0] if drone_id in rescue_drones 
                            else [round(self.dataframes[i-6].iloc[0]['North (m)'], 2),
                                    round(self.dataframes[i-6].iloc[0]['East (m)'], 2),
                                    round(self.dataframes[i-6].iloc[0]['Down (m)'], 2)] 
                                    for i, drone_id in enumerate(rescue_drones+drone_ids)}
      
        self.num_drones = len(self.all_drone_ids)
        self.num_frames = len(self.dataframes[0]['Time (s)'])
        self.reduction_factor = 2
        self.short_num_frames = self.num_frames // self.reduction_factor

        self.initial_battery = 100.0
        self.batteries = [self.initial_battery] * self.num_drones
        self.drain_rates = [0.45 + i * 0.1 for i in range(self.num_drones)]  # Different drain rates for each drone
        # self.drain_rates = [0.02, 0.02, 0.02, 1 ]
        self.stopped_drain_rate = 0.05  # Small drain rate when stopped
        self.new_drone_drain_rate = 0.5 # Slightly higher than drone 1
        self.charging_rate = 0.8 # Battery charging rate for drones at home

        self.battery_status={drone_id: [] for drone_id in self.all_drone_ids}
        # Initialize battery levels
        for drone_id in self.all_drone_ids:
            self.battery_status[drone_id].append(self.initial_battery)
        
        self.rescue_battery_status = {drone_id: [] for drone_id in rescue_drones}
        self.keys_list = list(self.rescue_battery_status.keys())
        
        # Rescue mission state variables
        self.stop_all_drones = False
        self.low_batt_drone = None
        self.low_batt_index = None
        self.return_path = []
        self.return_step = 0
        self.return_steps = 50

        self.rescue_path = []
        self.rescue_step = 0
        self.rescue_steps = 50

        self.rescue_started = False
        self.rescue_finished = False
        self.active_rescue_drones = [False] * len(rescue_drones)
        self.new_drones_start_pos = [None] * len(rescue_drones)
        self.replaced_drone = {drone_id: [] for drone_id in range(8)}
        self.ghost_path = pd.DataFrame(columns=['North (m)', 'East (m)', 'Down (m)'])

        # Battery change variables
        self.time_step = self.T.iloc[-1] / self.short_num_frames
        self.current_time = 0.0
        self.time_points = [0]
        
        self.drone_at_home = {drone_id: False for drone_id in self.all_drone_ids}
        self.in_return_path = {drone_id: False for drone_id in self.all_drone_ids}
        
        self.new_drone_mission_started = False
        self.timer = self.create_timer(0.2, self.timer_callback)

    def update_batteries(self):
        self.current_time += self.time_step
        self.time_points.append(self.current_time)

        for idx, drone_id in enumerate(drone_ids):
            if self.drone_at_home[drone_id]:
                new_level = min(self.initial_battery, self.battery_status[drone_id][-1] + self.charging_rate * self.time_step)
            elif self.stop_all_drones:
                if not self.rescue_finished and not self.in_return_path[drone_id]:
                    new_level = max(0.0, self.battery_status[drone_id][-1] - self.stopped_drain_rate * self.time_step)
                elif not self.rescue_finished and self.in_return_path[drone_id]:
                    new_level = max(0.0, self.battery_status[drone_id][-1] - self.drain_rates[idx] * self.time_step)        
            else:
                new_level = max(0.0, self.battery_status[drone_id][-1] - self.drain_rates[idx] * self.time_step)

            self.battery_status[drone_id].append(new_level) 

        for idx, drone_id in enumerate(rescue_drones):
            if self.active_rescue_drones[idx] and self.rescue_finished:
                new_level = max(0.0, self.battery_status[drone_id][-1] - self.drain_rates[idx] * self.time_step)
                self.battery_status[drone_id].append(new_level)
            elif self.active_rescue_drones[idx] and not self.rescue_finished:
                new_level = max(0.0, self.battery_status[drone_id][-1] - self.new_drone_drain_rate * self.time_step)
                self.battery_status[drone_id].append(new_level)
            elif self.drone_at_home[drone_id]:
                new_level = min(self.initial_battery, self.battery_status[drone_id][-1] + self.charging_rate * self.time_step)
                self.battery_status[drone_id].append(new_level)
            else:
                self.battery_status[drone_id].append(self.initial_battery)

    def timer_callback(self):
        self.update_batteries()

        # Regular mission loop
        all_finished = True
        for i, drone_id in enumerate(drone_ids):
            df = self.dataframes[i]
            index = self.indices[i]

            if index < len(df):
                all_finished = False
                row = df.iloc[index]

                # Publish position and battery
                msg = String()
                battery = Float32()

                # Flying drones Coordinates and battery
                if not self.stop_all_drones:
                    if drone_id != self.low_batt_drone:
                        msg.data = f"N: {row['North (m)']:.2f}, E: {row['East (m)']:.2f}, D: {row['Down (m)']:.2f}"
                        battery.data = self.battery_status[drone_id][-1]
                        self.drone_publishers1[drone_id].publish(msg)
                        self.drone_publishers2[drone_id].publish(battery)
                        self.get_logger().info(f"[Flying Drone: {drone_id}] Position & Battery Level: {msg.data} | {battery.data:.2f}%")
                        self.store_position[drone_id].append(msg.data)
                else:
                    if drone_id != self.low_batt_drone:
                        if not self.drone_at_home[drone_id]:
                            msg.data = self.store_position[drone_id][-1]
                            battery.data = self.battery_status[drone_id][-1]
                            self.drone_publishers1[drone_id].publish(msg)
                            self.drone_publishers2[drone_id].publish(battery)
                            self.get_logger().info(f"[Flying Drone: {drone_id}] Position & Battery Level: {msg.data} | {battery.data:.2f}%")
                        else:
                            battery.data = self.battery_status[drone_id][-1]
                            self.drone_publishers2[drone_id].publish(battery)
                            if battery.data<100.00:
                                self.get_logger().info(f"[Drone: {drone_id}] is charging: {battery.data:.2f}%")
                            else:
                                self.get_logger().info(f"[Drone: {drone_id}] fully charged: {battery.data:.2f}%")
                    else:
                        battery.data = self.battery_status[drone_id][-1]
                        self.drone_publishers2[drone_id].publish(battery)
                        if self.drone_at_home[drone_id]:
                            if battery.data<100.00:
                                self.get_logger().info(f"[Drone: {drone_id}] is charging: {battery.data:.2f}%")
                            else:
                                self.get_logger().info(f"[Drone: {drone_id}] fully charged: {battery.data:.2f}%")
                
                # LOW BATTERY DETECTION AND RESCUE
                if battery.data <= 20.0 and not self.stop_all_drones:
                    msg.data = self.store_position[drone_id][-1]
                    self.last_pos_pub.publish(msg)
                    if self.drone_at_home[drone_id]:
                        self.get_logger().warn(f"[{rescue_drones[i]}] Low battery!{battery.data:.2f}% Initiating return-rescue protocol. Stopping all drones!'\n")
                    else: 
                        self.get_logger().warn(f"[{drone_id}] Low battery!{battery.data:.2f}% Initiating return-rescue protocol. Stopping all drones!'\n")
                    self.get_logger().info(f"Low battery drone last position: {msg.data}")
                    self.stop_all_drones = True
                    self.low_batt_drone = drone_id #i
                    self.low_batt_index = index
                    self.in_return_path[self.low_batt_drone] = True

                    curr_pos = row[['North (m)', 'East (m)', 'Down (m)']].to_numpy()
                    start_pos = self.res_start_pos[drone_id]
                    return_vec = start_pos - curr_pos

                    self.return_path = [curr_pos + (return_vec * (step / self.return_steps))
                                            for step in range(self.return_steps + 1)]                    

                    # Setup rescue
                    res_start_position = np.array(self.res_start_pos[rescue_drones[drone_ids.index(self.low_batt_drone)]])
                    self.new_drones_start_pos[drone_ids.index(self.low_batt_drone)] = res_start_position 
                    rescue_vec = curr_pos - self.new_drones_start_pos[drone_ids.index(self.low_batt_drone)]
                    self.rescue_path = [self.new_drones_start_pos[drone_ids.index(self.low_batt_drone)] + (rescue_vec * (step / self.rescue_steps))
                                        for step in range(self.rescue_steps + 1)]

                    # Ghost path (trajectory before failure)
                    self.ghost_path = df.iloc[:index][['North (m)', 'East (m)', 'Down (m)']].copy()
                    
                    self.return_step = 0
                    self.rescue_step = 0
                    self.rescue_started = False
                    self.rescue_finished = False
                    self.active_rescue_drones[drone_ids.index(self.low_batt_drone)] = False
                    self.new_drone_mission_started = False
                # Pause index updates
                if not self.stop_all_drones:
                    self.indices[i] += 1

        # Rescue drones Coordinates and Battery
        for i, drone_id in enumerate(rescue_drones):
            battery.data = self.battery_status[drone_id][-1]
            self.drone_publishers2[drone_id].publish(battery)
            if not self.active_rescue_drones[i]:
                pos = self.res_start_pos[drone_id]
                msg.data = f"N: {pos[0]:.2f}, E: {pos[1]:.2f}, D: {pos[2]:.2f}"
                self.drone_publishers1[drone_id].publish(msg)
                if battery.data<100:
                    self.get_logger().info(f"[Drone: {drone_id}] Positioon: {msg.data} | charging: {battery.data:.2f}%")
                else:
                    if self.drone_at_home[drone_id]==True:
                        self.drone_at_home[drone_id]=False
                    self.get_logger().info(f"[Rescue Drone: {drone_id}] Position & Battery Level: {msg.data} | {battery.data:.2f}%")

        # Return and Rescue Operation
        if self.stop_all_drones and not self.rescue_finished:
            for i, drone_id in enumerate(drone_ids):
                if drone_id == self.low_batt_drone:
                    battery.data = self.battery_status[drone_id][-1]
                    # Return phase
                    if self.return_step <= self.return_steps:
                        pos = self.return_path[self.return_step]
                        msg_ret = String()
                        msg_ret.data = f"Position: N: {pos[0]:.2f}, E: {pos[1]:.2f}, D: {pos[2]:.2f}"
                        self.drone_publishers1[drone_id].publish(msg_ret)
                        self.get_logger().info(f"[Return] Drone {self.low_batt_drone} returning to base: {msg_ret.data} | {battery.data:.2f}%")
                        self.return_step += 1

                    if self.return_step > self.return_steps and not self.rescue_started:
                        self.get_logger().info(f"Drone {self.low_batt_drone} returned home. Charging started.\n")
                        self.rescue_started = True
                        self.active_rescue_drones[i] = True
                        self.rescue_step = 0
                        self.drone_at_home[self.low_batt_drone] = True
                        self.new_drone_mission_started = True
                        return
                    
            for i, drone_id in enumerate(rescue_drones):
                if i== drone_ids.index(self.low_batt_drone):
                    battery.data = self.battery_status[drone_id][-1]
                    # Rescue drone flight
                    if self.rescue_started and self.rescue_step <= self.rescue_steps:
                        pos = self.rescue_path[self.rescue_step]
                        msg_res = String()
                        msg_res.data = f"N: {pos[0]:.2f}, E: {pos[1]:.2f}, D: {pos[2]:.2f}"
                        self.drone_publishers1[drone_id].publish(msg_res)
                        self.get_logger().info(f"[Rescue] New drone {drone_id} route to rescue Position & Battery status: {msg_res.data} | {battery.data:.2f}%")
                        self.rescue_step += 1
                    # Finalize
                    if self.rescue_step > self.rescue_steps and not self.rescue_finished:
                        self.get_logger().info(f"[Rescue] Rescue Mission complete. Drone {drone_ids[i]} replaced by Drone {rescue_drones[i]}.\n")
                        # Replace the low battery drone with the rescue drone list
                        drone_ids[i]=rescue_drones[i]
                        rescue_drones[i] = self.low_batt_drone
                        self.active_rescue_drones[i] = False
                        self.in_return_path[self.low_batt_drone] = False
                        self.rescue_finished = True
                        self.stop_all_drones = False  # Resume others
                        return

        print('\n')

        if all_finished:
            self.get_logger().info("Finished publishing all drone data.")
            self.destroy_timer(self.timer)

def main(args=None):
    rclpy.init(args=args)
    node = MultiDroneCoordinatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
