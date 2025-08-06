import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib.animation import FuncAnimation
import threading
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d.proj3d import proj_transform

drone_ids = ['_1', '_2', '_3', '_4','_5', '_6']
rescue_drones = [ '_7', '_8', '_9', '_10', '_11', '_12']
class MultiDroneSubscriber(Node):
    def __init__(self):
        super().__init__('drone_subscriber')
        
        self.all_drone_ids = drone_ids + rescue_drones

        self.positions = {drone_id: {'N': [], 'E': [], 'D': []} for drone_id in self.all_drone_ids}
        self.batteries = {drone_id: [] for drone_id in self.all_drone_ids}
        self.times = {drone_id: [] for drone_id in self.all_drone_ids}
        self.current_time = 0.0
        self.time_step = 0.2
        self.rescue_step = 0
        self.last_pos= None
        self.low_batt_drone = None
        self.rescue_started = False
        self.active_rescue_drone = None
        self.returned_drones = {drone_id: False for drone_id in self.all_drone_ids}

        self.colors = {
            '_1': 'red', '_2': 'blue', '_3': 'green', '_4': 'orange',
            '_5': 'purple', '_6': 'teal', '_7': 'magenta', '_8': 'cyan',
            '_9': 'tan', '_10': 'wheat', '_11': 'silver', '_12': 'pink'
        }
        # Load once in __init__
        self.drone_icon = mpimg.imread('/home/soumen/swam_ws/src/ChatGPT Image Jul 29, 2025, 09_14_24 AM.png')
        self.drone_annotations = {drone_id: None for drone_id in self.all_drone_ids}

        # Subscriptions
        for drone_id in self.all_drone_ids:
            coord_topic = f'{drone_id}/drone_coordinates'
            self.create_subscription(String, coord_topic, lambda msg, did=drone_id: self.coord_callback(msg, did), 10)
            battery_topic = f'{drone_id}/drone_battery'
            self.create_subscription(Float32, battery_topic, lambda msg, did=drone_id: self.battery_callback(msg, did), 10)
        self.create_subscription(String,'last_position',self.last_positon,10) 
        
        # Plot setup
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        self.ax_3d = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_2d = self.fig.add_subplot(gs[1])
        self.setup_plot()

        # Animation
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=200)
        plt.tight_layout()

    def setup_plot(self):
        self.ax_3d.set_xlim(-15, 121.593)
        self.ax_3d.set_ylim(-50, 20)
        self.ax_3d.set_zlim(0, -25.882)
        self.ax_3d.set_xlabel('North (m)')
        self.ax_3d.set_ylabel('East (m)')
        self.ax_3d.set_zlabel('Down (m)')
        self.ax_3d.set_title('Drone Trajectories')
        self.ax_3d.set_box_aspect([1, 1, 1])
        self.ax_3d.grid(True)

        self.ax_2d.set_xlabel('Time (s)')
        self.ax_2d.set_ylabel('Battery Level (%)')
        self.ax_2d.set_title('Battery Levels')
        self.ax_2d.set_xlim(0, 800)
        self.ax_2d.set_ylim(0, 110)
        self.ax_2d.set_facecolor('#f5f5f5')
        self.ax_2d.grid(True, alpha=0.3)

        # Initialize empty lines (they start hidden)
        self.lines_3d = {}
        self.markers_3d = {}
        self.battery_lines = {}

        for drone_id in drone_ids:
            color = self.colors[drone_id]
            self.lines_3d[drone_id], = self.ax_3d.plot([], [], [], color=color, label=f'Drone {drone_id}')
            self.markers_3d[drone_id], = self.ax_3d.plot([], [], [], color=color, marker='*', markersize=10)
            self.battery_lines[drone_id], = self.ax_2d.plot([], [], color=color, label=f'Drone {drone_id}')
        for drone_id in rescue_drones:
            color = self.colors[drone_id]
            self.lines_3d[drone_id], = self.ax_3d.plot([], [], [], linestyle=':', color=color, label=f'Drone {drone_id}')
            self.markers_3d[drone_id], = self.ax_3d.plot([], [], [], color=color, marker='*', markersize=10)
            self.battery_lines[drone_id], = self.ax_2d.plot([], [], color=color, label=f'Drone {drone_id}')

        self.ax_3d.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        self.ax_2d.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    
    def parse_position(self, msg_data):
        try:
            matches = re.findall(r'N: ([-]?\d+\.\d+), E: ([-]?\d+\.\d+), D: ([-]?\d+\.\d+)', msg_data)
            if matches:
                return tuple(map(float, matches[0]))
        except Exception as e:
            self.get_logger().error(f"Error parsing position: {msg_data}, {e}")
        return None

    def last_positon(self,msg):
        pos = self.parse_position(msg.data)
        self.last_pos= list(pos)
        print(f"last position: {self.last_pos}")
    
    def coord_callback(self, msg, drone_id):
        pos = self.parse_position(msg.data)
        if pos:
            n, e, d = pos
            self.positions[drone_id]['N'].append(n)
            self.positions[drone_id]['E'].append(e)
            self.positions[drone_id]['D'].append(d)
        self.get_logger().info(f"Drone{drone_id} coordinates: {pos}")
        
    def battery_callback(self, msg, drone_id):
        self.batteries[drone_id].append(msg.data)
        self.times[drone_id].append(self.current_time)
        self.get_logger().info(f"Drone{drone_id} battery: {msg.data:.2f}%")
        if msg.data < 20 and drone_id in drone_ids:
            self.low_batt_drone = drone_id
            print(f"Low drone {self.low_batt_drone} and the step: {self.rescue_step}")
            self.rescue_step+=1
            print('rescue start')
            self.rescue_started = True  # Start the rescue process
       
        # Handle rescue drone replacement
        if self.rescue_started and drone_id in rescue_drones:
            if self.last_pos == [lst[-1] for lst in self.positions[drone_id].values()]:
                drone_ids[drone_ids.index(self.low_batt_drone)] = drone_id
                rescue_drones[rescue_drones.index(drone_id)] = self.low_batt_drone
                self.get_logger().info(f"Rescue complete: Drone{drone_id} replaced Drone{self.low_batt_drone}")
                print(drone_ids)
                print(rescue_drones)
                # Reset flags
                self.low_batt_drone = None
                self.active_rescue_drone = None
                self.rescue_started = False
        if drone_id == rescue_drones[-1]:
            print('\n')

    def update_plot(self, frame):
        self.current_time += self.time_step

        # Update each drone
        for drone_id in self.all_drone_ids:
            N = self.positions[drone_id]['N']
            E = self.positions[drone_id]['E']
            D = self.positions[drone_id]['D']

            if N:
                # Update trajectory line
                self.lines_3d[drone_id].set_visible(True)
                self.lines_3d[drone_id].set_data_3d(N, E, D)

                # Remove default marker
                # self.markers_3d[drone_id].set_visible(False)

                # Project last position to 2D for image placement
                x2, y2, _ = proj_transform(N[-1], E[-1], D[-1], self.ax_3d.get_proj())

                # Remove previous image if exists
                if self.drone_annotations[drone_id]:
                    self.drone_annotations[drone_id].remove()

                # Create and place new image
                imagebox = OffsetImage(self.drone_icon, zoom=0.05)
                ab = AnnotationBbox(imagebox, (x2, y2), frameon=False)
                self.ax_3d.add_artist(ab)
                self.drone_annotations[drone_id] = ab

            # Battery graph update
            if self.batteries[drone_id]:
                self.battery_lines[drone_id].set_visible(True)
                self.battery_lines[drone_id].set_data(self.times[drone_id], self.batteries[drone_id])

        return (
            list(self.lines_3d.values())
            + list(self.battery_lines.values())
            + [ab for ab in self.drone_annotations.values() if ab is not None]
        )

def main(args=None):
    rclpy.init(args=args)
    node = MultiDroneSubscriber()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()