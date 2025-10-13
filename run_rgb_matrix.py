#!/usr/bin/env python3

import configparser
import serial
import time
import os
import sys
import psutil
import logging
import numpy as np

# Setup logging (level will be set after config is loaded)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiskMonitor:
    def __init__(self, hysterisis_time=20):
        self.read_usage_history = [0]
        self.write_usage_history = [0]
        self.history_times = [0]
        self.highest_read_rate = 0.00001
        self.highest_write_rate = 0.00001
        self.max_history_size = hysterisis_time

    def get(self):
        try:
            disk_io = psutil.disk_io_counters()
            read_usage = disk_io.read_bytes
            write_usage = disk_io.write_bytes
            self.read_usage_history.append(read_usage)
            self.write_usage_history.append(write_usage)
            self.history_times.append(time.time())
            if len(self.read_usage_history) > self.max_history_size:
                self.read_usage_history = self.read_usage_history[-self.max_history_size:]
                self.write_usage_history = self.write_usage_history[-self.max_history_size:]
                self.history_times = self.history_times[-self.max_history_size:]

            read_diff = self.read_usage_history[-1] - self.read_usage_history[0]
            write_diff = self.write_usage_history[-1] - self.write_usage_history[0]
            time_diff = self.history_times[-1] - self.history_times[0]
            read_rate = read_diff / time_diff
            write_rate = write_diff / time_diff
            self.highest_read_rate = max(self.highest_read_rate, read_rate)
            self.highest_write_rate = max(self.highest_write_rate, write_rate)

            effective_max_read = self.highest_read_rate * 0.75
            effective_max_write = self.highest_write_rate * 0.75

            read_percent = min(1.0, read_rate / effective_max_read) if effective_max_read > 0 else 0
            write_percent = min(1.0, write_rate / effective_max_write) if effective_max_write > 0 else 0

            return read_percent, write_percent
        except Exception as e:
            logger.error(f"Error in DiskMonitor.get(): {e}")
            return 0, 0

class NetworkMonitor:
    def __init__(self, hysterisis_time=20):
        self.sent_usage_history = [0]
        self.recv_usage_history = [0]
        self.history_times = [0]
        self.highest_sent_rate = 0.00001
        self.highest_recv_rate = 0.00001
        self.max_history_size = hysterisis_time

    def get(self):
        try:
            net_io = psutil.net_io_counters()
            sent_usage = net_io.bytes_sent
            recv_usage = net_io.bytes_recv
            self.sent_usage_history.append(sent_usage)
            self.recv_usage_history.append(recv_usage)
            self.history_times.append(time.time())
            if len(self.sent_usage_history) > self.max_history_size:
                self.sent_usage_history = self.sent_usage_history[-self.max_history_size:]
                self.recv_usage_history = self.recv_usage_history[-self.max_history_size:]
                self.history_times = self.history_times[-self.max_history_size:]

            sent_diff = self.sent_usage_history[-1] - self.sent_usage_history[0]
            recv_diff = self.recv_usage_history[-1] - self.recv_usage_history[0]
            time_diff = self.history_times[-1] - self.history_times[0]
            sent_rate = sent_diff / time_diff
            recv_rate = recv_diff / time_diff
            self.highest_sent_rate = max(self.highest_sent_rate, sent_rate)
            self.highest_recv_rate = max(self.highest_recv_rate, recv_rate)

            effective_max_sent = self.highest_sent_rate * 0.75
            effective_max_recv = self.highest_recv_rate * 0.75

            sent_percent = min(1.0, sent_rate / effective_max_sent) if effective_max_sent > 0 else 0
            recv_percent = min(1.0, recv_rate / effective_max_recv) if effective_max_recv > 0 else 0

            return sent_percent, recv_percent
        except Exception as e:
            logger.error(f"Error in NetworkMonitor.get(): {e}")
            return 0, 0

class LEDMatrixController:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.load_config()
        debug_enabled = self.config['Settings'].getboolean('debug', fallback=False)
        logger.setLevel(logging.DEBUG if debug_enabled else logging.INFO)

        self.panels = {
            0: {'serial': None, 'device': self.config['Settings']['panel_1_dev']},
            1: {'serial': None, 'device': self.config['Settings']['panel_2_dev']}
        }
        self.running = True
        self.matrix_width = 9
        self.matrix_height = 32
        self.total_leds = self.matrix_width * self.matrix_height
        self.current_leds = [0] * 16
        self.current_cpu2_leds = [0] * 8
        self.current_drive_leds = {'read': 0, 'write': 0}
        self.current_net_leds = {'sent': 0, 'recv': 0}
        self.current_gpu_leds = 0
        self.current_igpu_load = 0  # For gpu2 iGPU load smoothing
        self.current_dgpu_load = 0  # For gpu2 dGPU load smoothing
        self.current_igpu_mem = 0   # For gpu2 iGPU memory smoothing
        self.current_dgpu_mem = 0   # For gpu2 dGPU memory smoothing
        self.disk_monitor = DiskMonitor()
        self.net_monitor = NetworkMonitor()
        self.lid_closed = False
        self.colors = {
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'red': (255, 0, 0),
            'orange': (255, 165, 0),
            'light_green': (144, 238, 144),
            'blue': (0, 0, 255),
            'light_blue': (173, 216, 230),
            'purple': (128, 0, 128),
            'brown': (165, 42, 42),
            'aqua': (0, 255, 255),
            'pink': (255, 105, 180),
            'white': (255, 255, 255)
        }
        # Add alphanumeric grids (3x4) for numbers and letters
        self.alpha_grids = {
            '0': [[0,1,0], [1,0,1], [1,0,1], [0,1,0]],
            '1': [[1,1,0], [0,1,0], [0,1,0], [1,1,1]],
            '2': [[1,1,0], [0,0,1], [0,1,0], [1,1,1]],
            '3': [[1,1,0], [0,1,1], [0,0,1], [1,1,0]],
            '4': [[1,0,1], [1,1,1], [0,0,1], [0,0,1]],
            '5': [[1,1,1], [1,0,0], [0,1,1], [1,1,1]],
            '6': [[1,1,1], [1,0,0], [1,1,1], [1,1,1]],
            '7': [[1,1,1], [0,0,1], [0,0,1], [0,0,1]],
            '8': [[1,1,1], [0,1,0], [1,0,1], [1,1,1]],
            '9': [[1,1,1], [1,1,1], [0,0,1], [1,1,1]],
            '.': [[0,0,0], [0,0,0], [0,0,0], [0,1,0]],
            ':': [[0,1,0], [0,0,0], [0,0,0], [0,1,0]],
            'A': [[0,1,0], [1,0,1], [1,1,1], [1,0,1]],
            'P': [[1,1,1], [1,0,1], [1,1,1], [1,0,0]],
            'W': [[1,0,1], [1,0,1], [1,1,1], [1,1,1]],
            ' ': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]  # Blank space
        }

        # Add power tracking variables
        self.last_battery_percent = None
        self.last_battery_time = None
        self.last_power_watts = 0
    def load_config(self):
        logger.debug("Loading configuration")
        self.config.read('led_config.ini')
        if not self.config.sections():
            logger.info("Creating default configuration")
            self.config['Settings'] = {
                'number_of_panels': '2',
                'panel_1_dev': '/dev/ttyRGBM1',
                'panel_2_dev': '/dev/ttyRGBM2',
                'brightness_source': '/sys/class/backlight/amdgpu_bl2/brightness',
                'panel_1_order': 'cpu_temp,cpu_load2,gpu2,dgpu_temp',
                'panel_2_order': 'battery2,power1,line_pink,ram3,line_brown,net2,line_brown,drive1',
                'cpu_temp': 'k10temp',
                'igpu_load': '/sys/class/drm/card2/device/gpu_busy_percent',
                'igpu_mem_total': '/sys/class/drm/card2/device/mem_info_vram_total',
                'igpu_mem_used': '/sys/class/drm/card2/device/mem_info_vram_used',
                'igpu_temp': '/sys/class/drm/card2/device/hwmon/hwmon4/temp1_input',
                'dgpu_load': '/sys/class/drm/card1/device/gpu_busy_percent',
                'dgpu_mem_total': '/sys/class/drm/card1/device/mem_info_vram_total',
                'dgpu_mem_used': '/sys/class/drm/card1/device/mem_info_vram_used',
                'dgpu_temp': '/sys/class/drm/card1/device/hwmon/hwmon3/temp1_input',
                'dgpu_powerstate': '/sys/class/drm/card1/device/power_state',
                'dgpu_sleep_time': '6',
                'max_brightness': '150',
                'min_brightness': '1',
                'debug': 'false',
                'clock1_colors': 'red,orange,yellow,green,blue',  # 5 colors for digits
                'clock1_type': '12',  # Default to 12-hour clock
                'clock2_colors': 'red,orange,yellow,green,blue,purple',  # Default colors for 6 characters
                'gpu_text_colors': 'aqua,blue,aqua'  # Default colors for "G", "P", "U"
            }
            with open('led_config.ini', 'w') as configfile:
                self.config.write(configfile)
    def get_brightness(self):
        try:
            with open(self.config['Settings']['brightness_source'], 'r') as f:
                brightness = int(f.read().strip())
            logger.debug(f"Got brightness: {brightness}")

            # Derive max_brightness file path from brightness_source
            brightness_source = self.config['Settings']['brightness_source']
            max_brightness_path = '/'.join(brightness_source.split('/')[:-1]) + '/max_brightness'

            # Read max_brightness from the system file
            with open(max_brightness_path, 'r') as f:
                max_brightness_source = int(f.read().strip())
            logger.debug(f"Got max_brightness_source: {max_brightness_source}")

            # rgb matrix brightness ranges
            max_brightness = int(self.config['Settings'].get('max_brightness', 255))
            min_brightness = int(self.config['Settings'].get('min_brightness', 1))

            max_brightness = max(1, min(255, max_brightness))
            min_brightness = max(0, min(max_brightness, min_brightness))

            if brightness <= 0:
                scaled_brightness = min_brightness
            else:
                scaled_brightness = min_brightness + (brightness / max_brightness_source) * (max_brightness - min_brightness)

            return max(0.0, min(1.0, scaled_brightness / 255))
        except Exception as e:
            logger.warning(f"Failed to read brightness: {e}")
            return 0.5

    def get_gpu_load(self, load_file_key='igpu_load'):
        """Get GPU utilization percentage from sysfs"""
        try:
            load_file = self.config['Settings'].get(load_file_key, '/sys/class/drm/card0/device/gpu_busy_percent')
            with open(load_file, 'r') as f:
                gpu_load = float(f.read().strip())
            logger.debug(f"GPU load ({load_file_key}): {gpu_load}%")
            return max(0.0, min(100.0, gpu_load)) / 100.0  # Return as fraction (0.0 to 1.0)
        except Exception as e:
            logger.warning(f"Failed to read GPU load ({load_file_key}): {e}")
            return 0.0

    def get_gpu_temp(self, temp_file_key='igpu_temp'):
        """Get GPU temperature in degrees Celsius from hwmon sysfs"""
        try:
            temp_file = self.config['Settings'].get(temp_file_key, '/sys/class/drm/card0/device/hwmon/hwmon0/temp1_input')
            with open(temp_file, 'r') as f:
                temp_milli = int(f.read().strip())  # Value in millidegrees Celsius
            temp_celsius = temp_milli / 1000.0  # Convert to degrees Celsius
            logger.debug(f"GPU temp ({temp_file_key}): {temp_celsius}°C")
            return temp_celsius
        except Exception as e:
            logger.warning(f"Failed to read GPU temp ({temp_file_key}): {e}")
            return 50.0

    def get_gpu_mem_percent(self, total_key='igpu_mem_total', used_key='igpu_mem_used'):
        """Get GPU memory usage percentage"""
        try:
            total_file = self.config['Settings'].get(total_key)
            used_file = self.config['Settings'].get(used_key)
            with open(total_file, 'r') as f:
                mem_total = int(f.read().strip())
            with open(used_file, 'r') as f:
                mem_used = int(f.read().strip())
            if mem_total > 0:
                mem_percent = (mem_used / mem_total) * 100.0
                logger.debug(f"GPU memory ({used_key}/{total_key}): {mem_percent}%")
                return max(0.0, min(100.0, mem_percent)) / 100.0
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to read GPU memory ({used_key}/{total_key}): {e}")
            return 0.0

    def get_dgpu_powerstate(self):
        """Get dGPU power state"""
        try:
            powerstate_file = self.config['Settings'].get('dgpu_powerstate')
            with open(powerstate_file, 'r') as f:
                state = f.read().strip().lower()
            logger.debug(f"dGPU power state: {state}")
            return state
        except Exception as e:
            logger.warning(f"Failed to read dGPU power state: {e}")
            return "unknown"

    def is_lid_closed(self):
        """Check if laptop lid is closed (Linux-specific)"""
        try:
            with open('/proc/acpi/button/lid/LID0/state', 'r') as f:
                state = f.read().strip()
                return 'closed' in state.lower()
        except FileNotFoundError:
            logger.debug("Lid state file not found, assuming lid is open")
            return False
        except Exception as e:
            logger.error(f"Error checking lid state: {e}")
            return False

    def connect_panels(self):
        for panel_id, panel in self.panels.items():
            if not panel['serial'] and self.running and not self.lid_closed:
                try:
                    panel['serial'] = serial.Serial(panel['device'], 115200, timeout=1)
                    time.sleep(2)
                    response = panel['serial'].readline()
                    logger.info(f"Connected to panel {panel_id} at {panel['device']}, response: {response}")
                except serial.SerialException as e:
                    logger.info(f"Waiting for panel {panel_id} at {panel['device']}: {e}")
                    time.sleep(1)

    def send_frame(self, panel_id, frame):
        if not self.panels[panel_id]['serial'] or not self.running:
            logger.warning(f"Panel {panel_id} not connected")
            return False

        brightness = self.get_brightness()
        header = bytearray([65, 100, 97])  # 'Ada'
        led_count = self.total_leds - 1
        header.extend([led_count >> 8, led_count & 0xFF])
        header.append(header[3] ^ header[4] ^ 0x55)

        data = bytearray()
        for r, g, b in frame:
            r_adjusted = max(0, min(255, int(r * brightness)))
            g_adjusted = max(0, min(255, int(g * brightness)))
            b_adjusted = max(0, min(255, int(b * brightness)))
            data.extend([r_adjusted, g_adjusted, b_adjusted])

        try:
            self.panels[panel_id]['serial'].write(header + data)
            self.panels[panel_id]['serial'].flush()
            logger.debug(f"Sent frame to panel {panel_id} ({len(data)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to send frame to panel {panel_id}: {e}")
            self.panels[panel_id]['serial'].close()
            self.panels[panel_id]['serial'] = None
            return False

    def clear_panels(self):
        """Send a clear frame to all panels and ensure it's processed"""
        logger.info("Clearing all panels")
        frame = [(0, 0, 0)] * self.total_leds
        for panel_id in self.panels:
            if self.panels[panel_id]['serial']:
                for _ in range(2):
                    success = self.send_frame(panel_id, frame)
                    if success:
                        logger.debug(f"Clear frame sent to panel {panel_id}")
                        time.sleep(0.5)
                    else:
                        logger.warning(f"Failed to send clear frame to panel {panel_id}")

    def disconnect_panels(self):
        """Clear and disconnect all panels"""
        logger.info("Disconnecting panels")
        self.clear_panels()
        for panel_id, panel in self.panels.items():
            if panel['serial']:
                try:
                    panel['serial'].flush()
                    time.sleep(0.5)
                    panel['serial'].close()
                    panel['serial'] = None
                    logger.info(f"Disconnected panel {panel_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting panel {panel_id}: {e}")

    def battery1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        battery = psutil.sensors_battery()
        percent = battery.percent if battery else 0
        charging = battery.power_plugged if battery else False

        border_color = self.colors['green']
        if charging:
            t = (time.time() % 1) * 2
            border_color = tuple(int(a + (b - a) * t) for a, b in 
                               zip(self.colors['yellow'], self.colors['green']))

        for col in range(3, 6):
            frame[start_row * 9 + col] = border_color
        for col in range(9):
            frame[(start_row + 1) * 9 + col] = border_color
        for row in range(start_row + 2, start_row + 12):
            frame[row * 9] = border_color
            frame[row * 9 + 8] = border_color
        for col in range(9):
            frame[(start_row + 11) * 9 + col] = border_color

        spiral_coords = (
            [(start_row + 10) * 9 + col for col in range(1, 8)] +
            [(start_row + row) * 9 + 7 for row in range(9, 1, -1)] +
            [(start_row + 2) * 9 + col for col in range(6, 0, -1)] +
            [(start_row + row) * 9 + 1 for row in range(3, 10)] +
            [(start_row + 9) * 9 + col for col in range(2, 7)] +
            [(start_row + row) * 9 + 6 for row in range(8, 2, -1)] +
            [(start_row + 3) * 9 + col for col in range(5, 1, -1)] +
            [(start_row + row) * 9 + 2 for row in range(4, 9)] +
            [(start_row + 8) * 9 + col for col in range(3, 6)] +
            [(start_row + row) * 9 + 5 for row in range(7, 3, -1)] +
            [(start_row + 4) * 9 + col for col in range(4, 2, -1)] +
            [(start_row + row) * 9 + 3 for row in range(5, 8)] +
            [(start_row + 7) * 9 + 4] +
            [(start_row + row) * 9 + 4 for row in range(6, 4, -1)]
        )

        leds_to_fill = int((percent / 100) * 63)
        spiral_color = self.colors['light_green']
        if percent == 100:
            spiral_color = self.colors['blue']
        elif percent < 10:
            spiral_color = self.colors['red']
        elif percent < 25:
            spiral_color = self.colors['orange']
        elif percent < 50:
            spiral_color = self.colors['yellow']

        for i in range(min(leds_to_fill, len(spiral_coords))):
            frame[spiral_coords[i]] = spiral_color

        return frame

    def get_cpu_temp(self):
        """Get CPU temperature directly from k10temp sysfs"""
        try:
            # Dynamically find k10temp; adjust if your hwmon index differs
            for hwmon in os.listdir('/sys/class/hwmon'):
                name_path = f'/sys/class/hwmon/{hwmon}/name'
                if os.path.exists(name_path):
                    with open(name_path, 'r') as f:
                        if f.read().strip() == 'k10temp':
                            with open(f'/sys/class/hwmon/{hwmon}/temp1_input', 'r') as f:
                                temp_milli = int(f.read().strip())
                            logger.debug(f"CPU temp from k10temp: {temp_milli / 1000.0}°C")
                            return temp_milli / 1000.0
            logger.warning("k10temp not found, using default temp")
            return 50.0
        except Exception as e:
            logger.warning(f"Failed to read CPU temp from k10temp: {e}")
            return 50.0

    def cpu_load1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        cpu_load = psutil.cpu_percent(percpu=True)
        try:
            temp = psutil.sensors_temperatures()[self.config['Settings']['cpu_temp']][0].current
        except:
            temp = 50

        temp_factor = max(0, min(1, (temp - 50) / 50))
        temp_color = tuple(int(a + (b - a) * temp_factor) for a, b in 
                         zip(self.colors['blue'], self.colors['red']))
        for row in range(start_row, start_row + 16):
            frame[row * 9 + 4] = temp_color

        for core, load in enumerate(cpu_load[:16]):
            target_leds = min(8, int((load / 100) * 8 + 0.5))
            current = self.current_leds[core]

            if current < target_leds:
                self.current_leds[core] += 1
            elif current > target_leds:
                self.current_leds[core] -= 1

            leds_to_fill = self.current_leds[core]
            load_factor = load / 100
            load_color = tuple(int(a + (b - a) * load_factor) for a, b in 
                             zip(self.colors['light_blue'], self.colors['purple']))

            if core % 2 == 0:  # Even cores (left side)
                outward_row = start_row + core
                inward_row = start_row + core + 1
                if leds_to_fill <= 4:
                    for col in range(3, 3 - leds_to_fill, -1):
                        frame[outward_row * 9 + col] = load_color
                else:
                    for col in range(0, 4):
                        frame[outward_row * 9 + col] = load_color
                    leds_in = leds_to_fill - 4
                    for col in range(leds_in):
                        frame[inward_row * 9 + col] = load_color
            else:  # Odd cores (right side)
                outward_row = start_row + core - 1
                inward_row = start_row + core
                if leds_to_fill <= 4:
                    for col in range(5, 5 + leds_to_fill):
                        frame[outward_row * 9 + col] = load_color
                else:
                    for col in range(5, 9):
                        frame[outward_row * 9 + col] = load_color
                    leds_in = leds_to_fill - 4
                    for col in range(8, 8 - leds_in, -1):
                        frame[inward_row * 9 + col] = load_color

        return frame


    def cpu_load2_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        full_cpu_load = psutil.cpu_percent(percpu=True)
        cpu_load = []
        for i in range(0, len(full_cpu_load), 2):
            if i + 1 < len(full_cpu_load):
                cpu_load.append(max(full_cpu_load[i], full_cpu_load[i + 1]))
            else:
                cpu_load.append(full_cpu_load[i])
        # Optionally limit to 8 if more pairs exist
        cpu_load = cpu_load[:8]
        temp = self.get_cpu_temp()
        temp_factor = max(0, min(1, (temp - 50) / 50))
        temp_color = tuple(int(a + (b - a) * temp_factor) for a, b in
                           zip(self.colors['blue'], self.colors['red']))
        for row in range(start_row, start_row + 16):
            frame[row * 9 + 4] = temp_color
            
        lookup_table = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ])

        for core, load in enumerate(cpu_load):
            target_idx = min(16, int((load / 100) * 16 + 0.5))
            current = self.current_cpu2_leds[core]
            if current < target_idx:
                self.current_cpu2_leds[core] += 1
            elif current > target_idx:
                self.current_cpu2_leds[core] -= 1
            idx = self.current_cpu2_leds[core]
            pattern = lookup_table[idx]
            factor = idx / 16
            r = int(self.colors['aqua'][0] + (self.colors['red'][0] - self.colors['aqua'][0]) * factor)
            g = int(self.colors['aqua'][1] + (self.colors['red'][1] - self.colors['aqua'][1]) * factor)
            b = int(self.colors['aqua'][2] + (self.colors['red'][2] - self.colors['aqua'][2]) * factor)
            load_color = (r, g, b)
            if core < 4:
                base_row = start_row + core * 4
                base_col = 0
                for row in range(4):
                    for col in range(4):
                        if pattern[row, col] == 1:
                            frame[(base_row + row) * 9 + (base_col + col)] = load_color
            else:
                base_row = start_row + (core - 4) * 4
                base_col = 5
                for row in range(4):
                    for col in range(4):
                        if pattern[row, 3 - col] == 1:
                            frame[(base_row + row) * 9 + (base_col + col)] = load_color
        return frame



    
    def drive1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        read_percent, write_percent = self.disk_monitor.get()

        edge_coords = [
            (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 8), (5, 8),
            (6, 7), (7, 6), (8, 5), (8, 4), (8, 3), (7, 2), (6, 1),
            (5, 0), (4, 0), (3, 0), (2, 1), (1, 2), (0, 3)
        ]
        circle_coords = [
            (0, 5), (1, 5), (2, 5), (1, 6), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7), 
            (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7), (5, 6), (6, 7), 
            (5, 5), (6, 6), (7, 6), (6, 5), (5, 4), (7, 5), (8, 5), (6, 4), (7, 4), 
            (8, 4), (8, 3), (7, 3), (6, 3), (7, 2), (5, 3), (6, 2), (6, 1), (5, 2), 
            (4, 3), (5, 1), (5, 0), (4, 2), (4, 1), (4, 0), (3, 0), (3, 1), (3, 2), 
            (2, 1), (3, 3), (2, 2), (2, 3), (1, 2), (1, 3), (0, 3)
        ]

        total_leds = len(circle_coords)
        half_circle = 26

        line_color = self.colors['blue']
        for row in range(start_row, start_row + 5):
            frame[row * 9 + 4] = line_color

        edge_color = self.colors['aqua']
        for row, col in edge_coords:
            frame[(start_row + row) * 9 + col] = edge_color

        target_read_leds = min(half_circle, int(read_percent * half_circle + 0.5))
        target_write_leds = min(half_circle, int(write_percent * half_circle + 0.5))

        if self.current_drive_leds['read'] < target_read_leds:
            self.current_drive_leds['read'] += 1
        elif self.current_drive_leds['read'] > target_read_leds:
            self.current_drive_leds['read'] -= 1

        if self.current_drive_leds['write'] < target_write_leds:
            self.current_drive_leds['write'] += 1
        elif self.current_drive_leds['write'] > target_write_leds:
            self.current_drive_leds['write'] -= 1

        read_leds = self.current_drive_leds['read']
        write_leds = self.current_drive_leds['write']

        read_color = self.colors['orange']
        for i in range(read_leds):
            row, col = circle_coords[i]
            frame[(start_row + row) * 9 + col] = read_color

        write_color = self.colors['brown']
        start_idx = total_leds - 1
        for i in range(write_leds):
            idx = (start_idx - i) % total_leds
            row, col = circle_coords[idx]
            frame[(start_row + row) * 9 + col] = write_color

        return frame

    def net1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        sent_percent, recv_percent = self.net_monitor.get()

        max_leds = 18

        target_sent_leds = min(max_leds, int(sent_percent * max_leds + 0.5))
        if target_sent_leds > self.current_net_leds['sent']:
            self.current_net_leds['sent'] += 1
        elif target_sent_leds < self.current_net_leds['sent']:
            self.current_net_leds['sent'] -= 1

        sent_leds = self.current_net_leds['sent']
        sent_color = self.colors['red']
        for i in range(sent_leds):
            row = 0 if i < 9 else 1
            col = i % 9
            frame[(start_row + row) * 9 + col] = sent_color

        target_recv_leds = min(max_leds, int(recv_percent * max_leds + 0.5))
        if target_recv_leds > self.current_net_leds['recv']:
            self.current_net_leds['recv'] += 1
        elif target_recv_leds < self.current_net_leds['recv']:
            self.current_net_leds['recv'] -= 1

        recv_leds = self.current_net_leds['recv']
        recv_color = self.colors['blue']
        for i in range(recv_leds):
            row = 3 if i < 9 else 2
            col = 8 - (i % 9)
            frame[(start_row + row) * 9 + col] = recv_color

        return frame

    def net2_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        sent_percent, recv_percent = self.net_monitor.get()

        send_coords = [
            (2, 0), (1, 0), (0, 0),
            (2, 1), (1, 1), (0, 1),
            (2, 2), (1, 2), (0, 2),
            (2, 3), (1, 3), (0, 3),
            (2, 4), (1, 4), (0, 4)
        ]
        recv_coords = [
            (2, 8), (1, 8), (0, 8),
            (2, 7), (1, 7), (0, 7),
            (2, 6), (1, 6), (0, 6),
            (2, 5), (1, 5), (0, 5),
            (2, 4), (1, 4), (0, 4)
        ]

        max_leds = 15

        target_sent_leds = min(max_leds, int(sent_percent * max_leds + 0.5))
        if target_sent_leds > self.current_net_leds['sent']:
            self.current_net_leds['sent'] += 1
        elif target_sent_leds < self.current_net_leds['sent']:
            self.current_net_leds['sent'] -= 1
        sent_leds = self.current_net_leds['sent']

        target_recv_leds = min(max_leds, int(recv_percent * max_leds + 0.5))
        if target_recv_leds > self.current_net_leds['recv']:
            self.current_net_leds['recv'] += 1
        elif target_recv_leds < self.current_net_leds['recv']:
            self.current_net_leds['recv'] -= 1
        recv_leds = self.current_net_leds['recv']

        col_4_send = set((row, 4) for row in range(3) if send_coords.index((row, 4)) < sent_leds)
        col_4_recv = set((row, 4) for row in range(3) if recv_coords.index((row, 4)) < recv_leds)

        for i in range(sent_leds):
            row, col = send_coords[i]
            idx = (start_row + row) * 9 + col
            frame[idx] = self.colors['red']

        for i in range(recv_leds):
            row, col = recv_coords[i]
            idx = (start_row + row) * 9 + col
            frame[idx] = self.colors['blue']

        for row in range(3):
            if (row, 4) in col_4_send and (row, 4) in col_4_recv:
                idx = (start_row + row) * 9 + 4
                frame[idx] = self.colors['purple']

        return frame

    def battery2_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        battery = psutil.sensors_battery()
        percent = battery.percent if battery else 0
        charging = battery.power_plugged if battery else False
    
        # Border color logic
        border_color = self.colors['green']
        if percent < 10:
            border_color = self.colors['red']
        elif charging:
            t = (time.time() % 1) * 2
            border_color = tuple(int(a + (b - a) * t) for a, b in
                                 zip(self.colors['yellow'], self.colors['green']))
    
        # Draw border
        for col in range(8):
            frame[start_row * 9 + col] = border_color
            frame[(start_row + 3) * 9 + col] = border_color
        frame[(start_row + 1) * 9] = border_color
        frame[(start_row + 1) * 9 + 8] = border_color
        frame[(start_row + 2) * 9] = border_color
        frame[(start_row + 2) * 9 + 8] = border_color
    
        # Progress bar: 14 steps across rows 1 and 2, cols 1-7
        max_leds = 14  # 7 LEDs per row, alternating
        leds_to_fill = int((percent / 100) * max_leds + 0.5) if percent < 100 else max_leds
        fill_color = self.colors['green']
        if percent == 100:
            fill_color = self.colors['blue']
        elif percent <= 30:
            fill_color = self.colors['yellow']
        elif percent <= 15:
            fill_color = self.colors['red']
    
        # Define the zigzag order: row 2 col 1, row 1 col 1, row 2 col 2, row 1 col 2, etc.
        led_order = []
        for col in range(1, 8):  # Columns 1 to 7
            led_order.append((start_row + 2) * 9 + col)  # Row 2
            led_order.append((start_row + 1) * 9 + col)  # Row 1
    
        # Fill LEDs in zigzag order
        for i in range(min(leds_to_fill, max_leds)):
            frame[led_order[i]] = fill_color
    
        return frame

    def ram1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram_percent = ram.percent
        swap_percent = swap.percent if swap.total > 0 else 0

        ranges = [
            (0, 25, start_row),
            (25, 50, start_row + 1),
            (50, 75, start_row + 2),
            (75, 100, start_row + 3)
        ]

        for min_percent, max_percent, row in ranges:
            if ram_percent > min_percent:
                percent_in_range = min(ram_percent, max_percent) - min_percent
                leds_to_fill = int((percent_in_range / 25) * 9)
                
                for col in range(leds_to_fill):
                    factor = col / 8
                    color = tuple(int(a + (b - a) * factor) for a, b in 
                                zip(self.colors['blue'], self.colors['purple']))
                    frame[row * 9 + col] = color
                
                if swap_percent > 50 and leds_to_fill > 0:
                    frame[row * 9 + 8] = self.colors['red']

        return frame

    def ram2_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram_percent = ram.percent
        swap_percent = swap.percent if swap.total > 0 else 0

        leds_to_fill = int((ram_percent / 100) * 9)

        for col in range(leds_to_fill):
            hue = (col / 8) * 360
            if hue < 60:
                r, g, b = 255, int(255 * (hue / 60)), 0
            elif hue < 120:
                r, g, b = int(255 * ((120 - hue) / 60)), 255, 0
            elif hue < 180:
                r, g, b = 0, 255, int(255 * ((hue - 120) / 60))
            elif hue < 240:
                r, g, b = 0, int(255 * ((240 - hue) / 60)), 255
            elif hue < 300:
                r, g, b = int(255 * ((hue - 240) / 60)), 0, 255
            else:
                r, g, b = 255, 0, int(255 * ((360 - hue) / 60))
            frame[start_row * 9 + col] = (r, g, b)
            frame[(start_row + 1) * 9 + col] = (r, g, b)

        if swap_percent > 50 and leds_to_fill > 0:
            frame[start_row * 9 + 8] = self.colors['red']
            frame[(start_row + 1) * 9 + 8] = self.colors['red']

        return frame

    def gpu1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        gpu_load = self.get_gpu_load('igpu_load')
        gpu_temp = self.get_gpu_temp('igpu_temp')

        max_leds_per_side = 4
        target_leds = int(gpu_load * max_leds_per_side + 0.5)

        if self.current_gpu_leds < target_leds:
            self.current_gpu_leds += 1
        elif self.current_gpu_leds > target_leds:
            self.current_gpu_leds -= 1

        leds_to_fill = self.current_gpu_leds

        temp_factor = max(0.0, min(1.0, (gpu_temp - 50) / 50))
        gpu_color = tuple(int(a + (b - a) * temp_factor) for a, b in 
                         zip(self.colors['blue'], self.colors['red']))

        for row in range(start_row, start_row + 4):
            frame[row * 9 + 4] = gpu_color
            for offset in range(1, leds_to_fill + 1):
                if 4 - offset >= 0:
                    frame[row * 9 + (4 - offset)] = gpu_color
                if 4 + offset <= 8:
                    frame[row * 9 + (4 + offset)] = gpu_color

        return frame
    def gpu_text_module(self, start_row):
        """Display 'GPU' text in the first 4 rows with configurable colors"""
        frame = [(0, 0, 0)] * self.total_leds
    
        # Get gpu_text colors from config, default to aqua, blue, purple
        gpu_colors_str = self.config['Settings'].get('gpu_text_colors', 'aqua,blue,purple')
        color_names = [name.strip() for name in gpu_colors_str.split(',')]
        # Ensure we have exactly 3 colors, padding with white or trimming excess
        while len(color_names) < 3:
            color_names.append('white')
        gpu_colors = [self.colors.get(name, self.colors['white']) for name in color_names[:3]]
    
        # Define grids for "GPU"
        g_grid = [[0,1,1], [1,0,0], [1,0,1], [0,1,1]]
        p_grid = [[1,1,1], [1,0,1], [1,1,1], [1,0,0]]
        u_grid = [[1,0,1], [1,0,1], [1,0,1], [1,1,1]]
        
        # Draw "GPU" across cols 0-8 with configurable colors
        for row in range(4):
            for col in range(3):
                if g_grid[row][col]:
                    frame[(start_row + row) * 9 + col] = gpu_colors[0]  # Color for "G"
                if p_grid[row][col]:
                    frame[(start_row + row) * 9 + (3 + col)] = gpu_colors[1]  # Color for "P"
                if u_grid[row][col]:
                    frame[(start_row + row) * 9 + (6 + col)] = gpu_colors[2]  # Color for "U"
    
        return frame

    def gpu2_module(self, start_row):
        """Dual GPU monitor with 'OFF' when in D3cold and sleep logic, now 8 rows"""
        frame = [(0, 0, 0)] * self.total_leds
    
        # Define a darker orange color for dGPU
        dark_orange = (200, 100, 0)
    
        # Rows 0-3: iGPU Info
        igpu_load = self.get_gpu_load('igpu_load')
        igpu_temp = self.get_gpu_temp('igpu_temp')
        igpu_mem_percent = self.get_gpu_mem_percent('igpu_mem_total', 'igpu_mem_used')
    
        # "i" in yellow (columns 0-2)
        i_grid = [[1,1,1], [0,1,0], [0,1,0], [1,1,1]]
        for row in range(4):
            for col in range(3):
                if i_grid[row][col]:
                    frame[(start_row + row) * 9 + col] = self.colors['yellow']
    
        # iGPU Load (rows 0, 1, 2)
        max_load_leds = 9
        target_load_pixels = int(igpu_load * max_load_leds * 3 + 0.5)
        if self.current_igpu_load < target_load_pixels:
            self.current_igpu_load += 1
        elif self.current_igpu_load > target_load_pixels:
            self.current_igpu_load -= 1
        temp_factor = max(0.0, min(1.0, (igpu_temp - 50) / 50))
        load_color = tuple(int(a + (b - a) * temp_factor) for a, b in
                           zip(self.colors['blue'], self.colors['red']))
    
        middle_row_leds = min(max_load_leds, (self.current_igpu_load + 1) // 2)
        side_rows_leds = min(max_load_leds, self.current_igpu_load // 2)
        for col in range(max_load_leds):
            if col < middle_row_leds:
                frame[(start_row + 1) * 9 + (8 - col)] = load_color
            if col < side_rows_leds:
                frame[(start_row + 0) * 9 + (8 - col)] = load_color
                frame[(start_row + 2) * 9 + (8 - col)] = load_color
    
        # iGPU Memory (row 3)
        max_mem_leds = 9
        target_mem_leds = int(igpu_mem_percent * max_mem_leds + 0.5)
        if self.current_igpu_mem < target_mem_leds:
            self.current_igpu_mem += 1
        elif self.current_igpu_mem > target_mem_leds:
            self.current_igpu_mem -= 1
        for col in range(max_mem_leds):
            if col < self.current_igpu_mem:
                factor = col / (max_mem_leds - 1) if max_mem_leds > 1 else 0
                mem_color = tuple(int(a + (b - a) * factor) for a, b in
                                 zip(self.colors['aqua'], self.colors['aqua']))
                frame[(start_row + 3) * 9 + (8 - col)] = mem_color
    
        # Rows 4-7: dGPU Info
        dgpu_sleep_time = float(self.config['Settings'].get('dgpu_sleep_time', 6))
    
        # Initialize tracking attributes
        if not hasattr(self, 'last_dgpu_poll_time'):
            self.last_dgpu_poll_time = 0
        if not hasattr(self, 'last_dgpu_powerstate_time'):
            self.last_dgpu_powerstate_time = 0
        if not hasattr(self, 'dgpu_sleeping'):
            self.dgpu_sleeping = False
        if not hasattr(self, 'last_dgpu_load'):
            self.last_dgpu_load = 0.0
            self.last_dgpu_temp = 0.0
            self.last_dgpu_mem_percent = 0.0
        if not hasattr(self, 'last_dgpu_powerstate'):
            self.last_dgpu_powerstate = "unknown"
        if not hasattr(self, 'dgpu_load_zero_count'):
            self.dgpu_load_zero_count = 0
    
        current_time = time.time()
    
        # "d" in dark orange (columns 0-2)
        d_grid = [[1,1,0], [1,0,1], [1,0,1], [1,1,0]]
        for row in range(4):
            for col in range(3):
                if d_grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + col] = dark_orange
    
        # Check dGPU power state
        if not self.dgpu_sleeping or current_time - self.last_dgpu_powerstate_time >= dgpu_sleep_time:
            dgpu_powerstate = self.get_dgpu_powerstate()
            self.last_dgpu_powerstate = dgpu_powerstate
            self.last_dgpu_powerstate_time = current_time
            logger.debug(f"DEBUG - dGPU power state: {dgpu_powerstate}")
        else:
            dgpu_powerstate = self.last_dgpu_powerstate
    
        # Determine dGPU state
        if dgpu_powerstate == "d3cold":
            # In D3cold: Use last values, show "OFF"
            dgpu_load = self.last_dgpu_load
            dgpu_temp = self.last_dgpu_temp
            dgpu_mem_percent = self.last_dgpu_mem_percent
            self.dgpu_sleeping = True
            self.last_dgpu_poll_time = current_time
            self.dgpu_load_zero_count = 0  # Reset counter in D3cold
            logger.debug(f"DEBUG - dGPU in D3cold, sleeping: {self.dgpu_sleeping}, load: {dgpu_load}")
    
            # Show scrolling zig-zag for power off
            off_grid = [[1,0,0,0,0,0,0,1], [0,1,0,0,0,0,1,0], [0,0,1,0,0,1,0,0], [0,0,0,1,1,0,0,0]]
            display_width = 6
            word_length = 8
            gap = 0
            total_pattern_length = word_length + gap
            # t = (time.time() % 1)
            t = (current_time % 1)
            offset = int(t * total_pattern_length)
            for row in range(4):
                for col in range(display_width):
                    pattern_col = (col + offset) % total_pattern_length
                    if pattern_col < word_length and off_grid[row][pattern_col]:
                        frame[(start_row + 4 + row) * 9 + col + 3] = self.colors['purple']

            """  
            # Show scrolling "OFF"
            o_grid = [[0,1,0], [1,0,1], [1,0,1], [0,1,0]]
            f_grid = [[1,1,1], [1,0,0], [1,1,0], [1,0,0]]
            display_width = 6
            word_length = 9
            gap = 2
            total_pattern_length = word_length + gap
            t = (time.time() % 2) / 2
            offset = int(t * total_pattern_length)
            for row in range(4):
                for col in range(display_width):
                    pattern_col = (col + offset) % total_pattern_length
                    if pattern_col < 3 and o_grid[row][pattern_col]:
                        frame[(start_row + 4 + row) * 9 + col + 3] = dark_orange
                    elif 3 <= pattern_col < 6 and f_grid[row][pattern_col - 3]:
                        frame[(start_row + 4 + row) * 9 + col + 3] = dark_orange
                    elif 6 <= pattern_col < 9 and f_grid[row][pattern_col - 6]:
                        frame[(start_row + 4 + row) * 9 + col + 3] = dark_orange """
        elif self.dgpu_sleeping and current_time - self.last_dgpu_poll_time < dgpu_sleep_time:
            # Sleeping but not D3cold: Use last values, no "OFF"
            dgpu_load = self.last_dgpu_load
            dgpu_temp = self.last_dgpu_temp
            dgpu_mem_percent = self.last_dgpu_mem_percent
            self.dgpu_load_zero_count = 0  # Reset counter while sleeping
            logger.debug(f"DEBUG - dGPU in sleep (not D3cold), sleeping: {self.dgpu_sleeping}, load: {dgpu_load}, time left: {dgpu_sleep_time - (current_time - self.last_dgpu_poll_time):.1f}s")
        else:
            # Active or sleep expired: Poll normally
            dgpu_load = self.get_gpu_load('dgpu_load')
            dgpu_temp = self.get_gpu_temp('dgpu_temp')
            dgpu_mem_percent = self.get_gpu_mem_percent('dgpu_mem_total', 'dgpu_mem_used')
            self.last_dgpu_load = dgpu_load
            self.last_dgpu_temp = dgpu_temp
            self.last_dgpu_mem_percent = dgpu_mem_percent
    
            if dgpu_load <= 0.0:
                self.dgpu_load_zero_count += 1
                logger.debug(f"DEBUG - dGPU load zero count: {self.dgpu_load_zero_count}, load: {dgpu_load}")
                if self.dgpu_load_zero_count >= 10:
                    # Enter sleep mode after 10 consecutive zeros
                    self.dgpu_sleeping = True
                    self.last_dgpu_poll_time = current_time
                    self.dgpu_load_zero_count = 0  # Reset after entering sleep
                    logger.debug(f"DEBUG - dGPU entering sleep, load: {dgpu_load}")
            else:
                # Stay active, reset counter
                self.dgpu_sleeping = False
                self.dgpu_load_zero_count = 0
                logger.debug(f"DEBUG - dGPU active, sleeping: {self.dgpu_sleeping}, load: {dgpu_load}")
    
        # dGPU Load (rows 4, 5, 6)
        target_load_pixels = int(dgpu_load * max_load_leds * 3 + 0.5)
        if self.current_dgpu_load < target_load_pixels:
            self.current_dgpu_load += 1
        elif self.current_dgpu_load > target_load_pixels:
            self.current_dgpu_load -= 1
        temp_factor = max(0.0, min(1.0, (dgpu_temp - 50) / 50))
        load_color = tuple(int(a + (b - a) * temp_factor) for a, b in
                           zip(self.colors['blue'], self.colors['red']))
    
        middle_row_leds = min(max_load_leds, (self.current_dgpu_load + 1) // 2)
        side_rows_leds = min(max_load_leds, self.current_dgpu_load // 2)
        for col in range(max_load_leds):
            if col < middle_row_leds:
                frame[(start_row + 5) * 9 + (8 - col)] = load_color
            if col < side_rows_leds:
                frame[(start_row + 4) * 9 + (8 - col)] = load_color
                frame[(start_row + 6) * 9 + (8 - col)] = load_color
    
        # dGPU Memory (row 7)
        target_mem_leds = int(dgpu_mem_percent * max_mem_leds + 0.5)
        if self.current_dgpu_mem < target_mem_leds:
            self.current_dgpu_mem += 1
        elif self.current_dgpu_mem > target_mem_leds:
            self.current_dgpu_mem -= 1
        for col in range(max_mem_leds):
            if col < self.current_dgpu_mem:
                factor = col / (max_mem_leds - 1) if max_mem_leds > 1 else 0
                mem_color = tuple(int(a + (b - a) * factor) for a, b in
                                 zip(self.colors['aqua'], self.colors['aqua']))
                frame[(start_row + 7) * 9 + (8 - col)] = mem_color
    
        return frame

    def line_module(self, start_row, module_name):
        frame = [(0, 0, 0)] * self.total_leds
        color_name = module_name.split('_')[1]

        if color_name == 'rainbow':
            t = time.time() % 2
            for col in range(9):
                hue = (col / 9 + t / 2) % 1 * 360
                if hue < 60:
                    r, g, b = 255, int(255 * (hue / 60)), 0
                elif hue < 120:
                    r, g, b = int(255 * ((120 - hue) / 60)), 255, 0
                elif hue < 180:
                    r, g, b = 0, 255, int(255 * ((hue - 120) / 60))
                elif hue < 240:
                    r, g, b = 0, int(255 * ((240 - hue) / 60)), 255
                elif hue < 300:
                    r, g, b = int(255 * ((hue - 240) / 60)), 0, 255
                else:
                    r, g, b = 255, 0, int(255 * ((360 - hue) / 60))
                frame[start_row * 9 + col] = (r, g, b)
        else:
            color = self.colors.get(color_name, (0, 0, 0))
            for col in range(9):
                frame[start_row * 9 + col] = color

        return frame

    def clock1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        current_time = time.localtime()

        # Get clock type (12 or 24) from config, default to 12
        clock_type = self.config['Settings'].get('clock1_type', '12')
        if clock_type not in ['12', '24']:
            clock_type = '12'  # Fallback to 12 if invalid

        # Determine hour based on clock type
        if clock_type == '12':
            hour = current_time.tm_hour % 12  # Convert to 12-hour format
            hour = 12 if hour == 0 else hour  # Replace 0 with 12 for 12 AM/PM
        else:  # 24-hour
            hour = current_time.tm_hour  # Use 0-23 directly
        minute = current_time.tm_min

        # Get clock1 colors from config, default to a varied list
        clock_colors_str = self.config['Settings'].get('clock1_colors', 'red,orange,yellow,green,blue,purple')
        color_names = [name.strip() for name in clock_colors_str.split(',')]
        # Ensure we have exactly 6 colors, padding with white or trimming excess
        while len(color_names) < 6:
            color_names.append('white')
        clock_colors = [self.colors.get(name, self.colors['white']) for name in color_names[:6]]

        # Hour display (rows 0-3)
        # First digit (cols 0-2): '1' or '2' if hour >= 10, else blank (24-hour can be 0-2)
        first_hour_digit = str(hour // 10) if hour >= 10 else ' '
        grid = self.alpha_grids[first_hour_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + row) * 9 + col] = clock_colors[0]

        # Second digit (cols 4-6)
        second_hour_digit = str(hour % 10)
        grid = self.alpha_grids[second_hour_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + row) * 9 + (4 + col)] = clock_colors[1]

        # Colon (col 8), blinks every second
        if int(time.time()) % 2 == 0:  # On even seconds
            grid = self.alpha_grids[':']
            for row in range(4):
                if grid[row][1]:  # Middle column of colon
                    frame[(start_row + row) * 9 + 8] = clock_colors[2]

        # Minute display (rows 4-7)
        # First digit (cols 2-4)
        first_minute_digit = str(minute // 10)
        grid = self.alpha_grids[first_minute_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + (2 + col)] = clock_colors[3]

        # Second digit (cols 6-8)
        second_minute_digit = str(minute % 10)
        grid = self.alpha_grids[second_minute_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + (6 + col)] = clock_colors[4]

        return frame


    def clock2_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        current_time = time.localtime()
        hour = current_time.tm_hour % 12  # Convert to 12-hour format
        hour = 12 if hour == 0 else hour  # Replace 0 with 12 for 12 AM/PM
        minute = current_time.tm_min
        is_pm = current_time.tm_hour >= 12

        # Get clock2 colors from config, default to a varied list
        clock_colors_str = self.config['Settings'].get('clock2_colors', 'red,orange,yellow,green,blue,purple')
        color_names = [name.strip() for name in clock_colors_str.split(',')]
        # Ensure we have exactly 6 colors, padding with white or trimming excess
        while len(color_names) < 6:
            color_names.append('white')
        clock_colors = [self.colors.get(name, self.colors['white']) for name in color_names[:6]]

        # Hour display (rows 0-3)
        # First digit (cols 0-2): '1' or '0' (or ' ' for <10 if preferred)
        first_hour_digit = str(hour // 10) if hour >= 10 else ' '
        grid = self.alpha_grids[first_hour_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + row) * 9 + col] = clock_colors[0]

        # Second digit (cols 3-5)
        second_hour_digit = str(hour % 10)
        grid = self.alpha_grids[second_hour_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + row) * 9 + (3 + col)] = clock_colors[1]

        # Colon (cols 6-8), blinks every second
        if int(time.time()) % 2 == 0:  # On even seconds
            grid = self.alpha_grids[':']
            for row in range(4):
                for col in range(3):
                    if grid[row][col]:
                        frame[(start_row + row) * 9 + (6 + col)] = clock_colors[2]

        # Minute display (rows 4-7)
        # First digit (cols 0-2)
        first_minute_digit = str(minute // 10)
        grid = self.alpha_grids[first_minute_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + col] = clock_colors[3]

        # Second digit (cols 3-5)
        second_minute_digit = str(minute % 10)
        grid = self.alpha_grids[second_minute_digit]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + (3 + col)] = clock_colors[4]

        # AM/PM (cols 6-8)
        am_pm_char = 'P' if is_pm else 'A'
        grid = self.alpha_grids[am_pm_char]
        for row in range(4):
            for col in range(3):
                if grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + (6 + col)] = clock_colors[5]

        return frame

    def power1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        battery = psutil.sensors_battery()

        # Check if on battery
        if battery and not battery.power_plugged:
            # Calculate wattage using sysfs values
            try:
                with open('/sys/class/power_supply/BAT1/current_now', 'r') as f:
                    current_now = int(f.read().strip())  # in microamperes
                with open('/sys/class/power_supply/BAT1/voltage_now', 'r') as f:
                    voltage_now = int(f.read().strip())  # in microvolts
                # Watts = (current in A) * (voltage in V)
                # Convert microamperes to amperes (*1e-6) and microvolts to volts (*1e-6)
                # So, (current_now * 1e-6) * (voltage_now * 1e-6) = watts / 1e12
                watts = (current_now * voltage_now) / 1e12
                watts_int = max(0, min(99, int(watts + 0.5)))  # Cap at 99W, round to integer
            except (FileNotFoundError, ValueError, IOError) as e:
                logger.warning(f"Failed to read power data: {e}")
                watts_int = 0

            # Convert wattage to string, two digits
            watts_str = f"{watts_int:02d}"  # Pad to 2 digits (e.g., "05", "42")
            display_chars = [watts_str[0], watts_str[1], 'W']  # 2 digits + 'W'

            # Colors for each character
            char_colors = [
                self.colors['orange'],    # First digit
                self.colors['orange'],  # Second digit
                self.colors['yellow']  # 'W'
            ]

            # Display characters in 3x4 grids, left-justified
            for char_idx, char in enumerate(display_chars):
                grid = self.alpha_grids[char]
                color = char_colors[char_idx]
                col_offset = char_idx * 3  # Each character takes 3 columns
                if col_offset + 3 <= 9:  # Ensure we fit within 9 columns
                    for row in range(4):
                        for col in range(3):
                            if grid[row][col]:
                                frame[(start_row + row) * 9 + (col_offset + col)] = color
        else:
            # Display plug icon when not on battery
            plug_icon = [
                [0, 1, 0, 1, 0, 0, 0, 0, 0],  # Row 0:
                [1, 1, 1, 1, 1, 0, 0, 0, 0],  # Row 1:
                [1, 1, 1, 1, 1, 0, 0, 1, 1],  # Row 2:
                [0, 1, 1, 1, 1, 1, 1, 0, 0]   # Row 3:
            ]
            plug_color = self.colors['yellow']  # Use yellow for the plug icon
            for row in range(4):
                for col in range(9):
                    if plug_icon[row][col]:
                        frame[(start_row + row) * 9 + col] = plug_color

        return frame


    def ram3_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram_percent = ram.percent
        swap_percent = swap.percent if swap.total > 0 else 0

        # Original edge_coords from drive1 (9x9 grid)
        original_edge_coords = [
            (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 8), (5, 8),
            (6, 7), (7, 6), (8, 5), (8, 4), (8, 3), (7, 2), (6, 1),
            (5, 0), (4, 0), (3, 0), (2, 1), (1, 2), (0, 3)
        ]

        # Original circle_coords from drive1 (fill order)
        original_circle_coords = [
            (0, 5), (1, 5), (2, 5), (1, 6), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7),
            (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7), (5, 6), (6, 7),
            (5, 5), (6, 6), (7, 6), (6, 5), (5, 4), (7, 5), (8, 5), (6, 4), (7, 4),
            (8, 4), (8, 3), (7, 3), (6, 3), (7, 2), (5, 3), (6, 2), (6, 1), (5, 2),
            (4, 3), (5, 1), (5, 0), (4, 2), (4, 1), (4, 0), (3, 0), (3, 1), (3, 2),
            (2, 1), (3, 3), (2, 2), (2, 3), (1, 2), (1, 3), (0, 3)
        ]

        # Flip coordinates horizontally and vertically
        # Horizontal flip: col -> (8 - col)
        # Vertical flip: row -> (8 - row)
        edge_coords = [(8 - row, 8 - col) for row, col in original_edge_coords]
        circle_coords = [(8 - row, 8 - col) for row, col in original_circle_coords]

        # Draw the circle edge
        edge_color = self.colors['aqua']
        for row, col in edge_coords:
            frame[(start_row + row) * 9 + col] = edge_color

        # Draw the purple vertical line from row 4, col 4 to row 7, col 4
        for row in range(4, 8):  # Rows 4, 5, 6, 7
            frame[(start_row + row) * 9 + 4] = self.colors['purple']

        # Fill the circle based on RAM percentage
        total_leds = len(circle_coords)
        target_leds = int((ram_percent / 100) * total_leds + 0.5)
        fill_color = self.colors['blue']
        if swap_percent > 50:
            fill_color = self.colors['red']

        for i in range(min(target_leds, total_leds)):
            row, col = circle_coords[i]
            frame[(start_row + row) * 9 + col] = fill_color

        return frame
        
    def cpu_temp_module(self, start_row):
        """Display CPU core temperature in 4 rows with up to 3 digits, adjusted layout"""
        frame = [(0, 0, 0)] * self.total_leds
        temp = self.get_cpu_temp()  # Get CPU temp in Celsius
        
        # Calculate temp factor (0 to 1) for color transition, same as cpu_load2
        temp_factor = max(0, min(1, (temp - 50) / 50))
        temp_color = tuple(int(a + (b - a) * temp_factor) for a, b in
                           zip(self.colors['blue'], self.colors['red']))

        # Convert temperature to integer and extract digits
        temp_int = int(temp + 0.5)  # Round to nearest integer
        temp_str = str(temp_int)  # Convert to string without padding

        # Determine number of digits and adjust layout
        if len(temp_str) == 3:
            # 3 digits: cols 0-2 (hundreds), 3-5 (tens), 6-8 (ones)
            grid_hundreds = self.alpha_grids[temp_str[0]]
            grid_tens = self.alpha_grids[temp_str[1]]
            grid_ones = self.alpha_grids[temp_str[2]]
            for row in range(4):
                # Hundreds (cols 0-2)
                for col in range(3):
                    if grid_hundreds[row][col]:
                        frame[(start_row + row) * 9 + col] = temp_color
                # Tens (cols 3-5)
                for col in range(3):
                    if grid_tens[row][col]:
                        frame[(start_row + row) * 9 + (3 + col)] = temp_color
                # Ones (cols 6-8)
                for col in range(3):
                    if grid_ones[row][col]:
                        frame[(start_row + row) * 9 + (6 + col)] = temp_color
        elif len(temp_str) == 2:
            # 2 digits: col 0 blank, cols 1-3 (tens), col 4 blank, cols 5-7 (ones), col 8 blank
            grid_tens = self.alpha_grids[temp_str[0]]
            grid_ones = self.alpha_grids[temp_str[1]]
            for row in range(4):
                # Tens (cols 1-3)
                for col in range(3):
                    if grid_tens[row][col]:
                        frame[(start_row + row) * 9 + (1 + col)] = temp_color
                # Ones (cols 5-7)
                for col in range(3):
                    if grid_ones[row][col]:
                        frame[(start_row + row) * 9 + (5 + col)] = temp_color
        else:  # len(temp_str) == 1
            # 1 digit: cols 0-2 blank, cols 3-5 (ones), cols 6-8 blank
            grid_ones = self.alpha_grids[temp_str[0]]
            for row in range(4):
                # Ones (cols 3-5)
                for col in range(3):
                    if grid_ones[row][col]:
                        frame[(start_row + row) * 9 + (3 + col)] = temp_color

        return frame
        
    def dgpu_temp_module(self, start_row):
        # Display dGPU temperature in 4 rows with up to 3 digits, adjusted layout
        frame = [(0, 0, 0)] * self.total_leds

        dgpu_sleep_time = float(self.config['Settings'].get('dgpu_sleep_time', 6))
    
        # Initialize tracking attributes
        if not hasattr(self, 'last_dgpu_poll_time'):
            self.last_dgpu_poll_time = 0
        if not hasattr(self, 'last_dgpu_powerstate_time'):
            self.last_dgpu_powerstate_time = 0
        if not hasattr(self, 'dgpu_sleeping'):
            self.dgpu_sleeping = False
        if not hasattr(self, 'last_dgpu_load'):
            self.last_dgpu_load = 0.0
            self.last_dgpu_temp = 0.0
            self.last_dgpu_mem_percent = 0.0
        if not hasattr(self, 'last_dgpu_powerstate'):
            self.last_dgpu_powerstate = "unknown"
        if not hasattr(self, 'dgpu_load_zero_count'):
            self.dgpu_load_zero_count = 0
    
        current_time = time.time()

        # Check dGPU power state
        if not self.dgpu_sleeping or current_time - self.last_dgpu_powerstate_time >= dgpu_sleep_time:
            dgpu_powerstate = self.get_dgpu_powerstate()
            self.last_dgpu_powerstate = dgpu_powerstate
            self.last_dgpu_powerstate_time = current_time
            logger.debug(f"DEBUG - dGPU power state: {dgpu_powerstate}")
        else:
            dgpu_powerstate = self.last_dgpu_powerstate
    

        # Determine dGPU state
        if dgpu_powerstate == "d3cold":
            dgpu_temp = self.last_dgpu_temp
            self.dgpu_sleeping = True
            self.last_dgpu_poll_time = current_time
            self.dgpu_load_zero_count = 0  # Reset counter in D3cold
            logger.debug(f"DEBUG - dGPU in D3cold, sleeping: {self.dgpu_sleeping}, temp: {dgpu_temp}")
    
            # Show 0 temp for power off
            off_grid = [[0,0,0,0,1,0,0,0,0], [1,1,0,1,0,1,0,1,1], [1,1,0,1,0,1,0,1,1], [0,0,0,0,1,0,0,0,0]]
            display_width = 9
            word_length = 9
            total_pattern_length = word_length
            for row in range(4):
                for col in range(9):
                    if off_grid[row][col]:
                        frame[(start_row + row) * 9 + col] = self.colors['green']
        elif self.dgpu_sleeping and current_time - self.last_dgpu_poll_time < dgpu_sleep_time:
            # Sleeping but not D3cold: Use last values, no "OFF"
            dgpu_temp = self.last_dgpu_temp
            self.dgpu_load_zero_count = 0  # Reset counter while sleeping
            logger.debug(f"DEBUG - dGPU in sleep (not D3cold), sleeping: {self.dgpu_sleeping}, temp: {dgpu_temp}, time left: {dgpu_sleep_time - (current_time - self.last_dgpu_poll_time):.1f}s")
        else:
            # Active or sleep expired: Poll normally
            dgpu_temp = self.get_gpu_temp('dgpu_temp')
            self.last_dgpu_temp = dgpu_temp

            temp = dgpu_temp  # Get dGPU temp in Celsius
            
            # Calculate temp factor (0 to 1) for color transition, same as cpu_load2
            temp_factor = max(0, min(1, (temp - 50) / 50))
            temp_color = tuple(int(a + (b - a) * temp_factor) for a, b in
                            zip(self.colors['blue'], self.colors['red']))

            # Convert temperature to integer and extract digits
            temp_int = int(temp + 0.5)  # Round to nearest integer
            temp_str = str(temp_int)  # Convert to string without padding

            # Determine number of digits and adjust layout
            if len(temp_str) == 3:
                # 3 digits: cols 0-2 (hundreds), 3-5 (tens), 6-8 (ones)
                grid_hundreds = self.alpha_grids[temp_str[0]]
                grid_tens = self.alpha_grids[temp_str[1]]
                grid_ones = self.alpha_grids[temp_str[2]]
                for row in range(4):
                    # Hundreds (cols 0-2)
                    for col in range(3):
                        if grid_hundreds[row][col]:
                            frame[(start_row + row) * 9 + col] = temp_color
                    # Tens (cols 3-5)
                    for col in range(3):
                        if grid_tens[row][col]:
                            frame[(start_row + row) * 9 + (3 + col)] = temp_color
                    # Ones (cols 6-8)
                    for col in range(3):
                        if grid_ones[row][col]:
                            frame[(start_row + row) * 9 + (6 + col)] = temp_color
            elif len(temp_str) == 2:
                # 2 digits: col 0 blank, cols 1-3 (tens), col 4 blank, cols 5-7 (ones), col 8 blank
                grid_tens = self.alpha_grids[temp_str[0]]
                grid_ones = self.alpha_grids[temp_str[1]]
                for row in range(4):
                    # Tens (cols 1-3)
                    for col in range(3):
                        if grid_tens[row][col]:
                            frame[(start_row + row) * 9 + (1 + col)] = temp_color
                    # Ones (cols 5-7)
                    for col in range(3):
                        if grid_ones[row][col]:
                            frame[(start_row + row) * 9 + (5 + col)] = temp_color
            else:  # len(temp_str) == 1
                # 1 digit: cols 0-2 blank, cols 3-5 (ones), cols 6-8 blank
                grid_ones = self.alpha_grids[temp_str[0]]
                for row in range(4):
                    # Ones (cols 3-5)
                    for col in range(3):
                        if grid_ones[row][col]:
                            frame[(start_row + row) * 9 + (3 + col)] = temp_color
        return frame
        
    def run(self):
        num_panels = int(self.config['Settings']['number_of_panels'])
        modules = {
            'battery1': (self.battery1_module, 12), # huge battery module, spiral fill
            'battery2': (self.battery2_module, 4), # small battery module
            'cpu_load1': (self.cpu_load1_module, 16), # cpu load, u-shaped lines
            'cpu_load2': (self.cpu_load2_module, 16), # cpu load, spiral
            'cpu_temp': (self.cpu_temp_module, 4),  # CPU temp module, 4 rows
            'dgpu_temp': (self.dgpu_temp_module, 4), # dGPU temp module, 4 rows
            'drive1': (self.drive1_module, 9), # drive activity, circle
            'net1': (self.net1_module, 4), # 
            'net2': (self.net2_module, 3), # network acivity, incremental bars
            'ram1': (self.ram1_module, 4), 
            'ram2': (self.ram2_module, 2),
            'ram3': (self.ram3_module, 9), # ram usage, circle dial
            'gpu_text': (self.gpu_text_module, 4),  # GPU text module, 4 rows. Just shows GPU in configurable colors.
            'gpu1': (self.gpu1_module, 4), # iGPU only (use if no dGPU)
            'gpu2': (self.gpu2_module, 8),  # iGPU and dGPU load and vram usage
            'clock1': (self.clock1_module, 8), # digital clock, 12 or 24 hour time
            'clock2': (self.clock2_module, 8),
            'power1': (self.power1_module, 4), # overall battery watts draining
            'line_red': (lambda row: self.line_module(row, 'line_red'), 1), # seperation lines, use one row in color listed
            'line_blue': (lambda row: self.line_module(row, 'line_blue'), 1),
            'line_green': (lambda row: self.line_module(row, 'line_green'), 1),
            'line_yellow': (lambda row: self.line_module(row, 'line_yellow'), 1),
            'line_orange': (lambda row: self.line_module(row, 'line_orange'), 1),
            'line_purple': (lambda row: self.line_module(row, 'line_purple'), 1),
            'line_pink': (lambda row: self.line_module(row, 'line_pink'), 1),
            'line_rainbow': (lambda row: self.line_module(row, 'line_rainbow'), 1),
            'line_aqua': (lambda row: self.line_module(row, 'line_aqua'), 1),
            'line_white': (lambda row: self.line_module(row, 'line_white'), 1),
            'line_brown': (lambda row: self.line_module(row, 'line_brown'), 1)
        }

        self.connect_panels()

        try:
            while self.running:
                current_lid_state = self.is_lid_closed()
                
                if current_lid_state != self.lid_closed:
                    self.lid_closed = current_lid_state
                    if self.lid_closed:
                        logger.info("Lid closed, disconnecting panels")
                        self.disconnect_panels()
                    else:
                        logger.info("Lid opened, reconnecting panels")
                        self.connect_panels()

                if not self.lid_closed:
                    for panel_id in range(min(num_panels, 2)):
                        if not self.panels[panel_id]['serial']:
                            logger.info(f"Attempting to reconnect panel {panel_id}")
                            self.connect_panels()
                            continue

                        frame = [(0, 0, 0)] * self.total_leds
                        panel_order = self.config['Settings'][f'panel_{panel_id + 1}_order'].split(',')
                        row_offset = 0

                        for module_name in panel_order:
                            if module_name in modules:
                                module_func, rows = modules[module_name]
                                if row_offset + rows <= self.matrix_height:
                                    module_frame = module_func(row_offset)
                                    for row in range(rows):
                                        for col in range(self.matrix_width):
                                            idx = (row_offset + row) * 9 + col
                                            frame[idx] = module_frame[idx]
                                    row_offset += rows

                        self.send_frame(panel_id, frame)

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Shutting down via KeyboardInterrupt")
            self.running = False

        finally:
            logger.info("Executing final cleanup")
            self.disconnect_panels()

if __name__ == '__main__':
    logger.info("Starting LED Matrix Controller...")
    controller = LEDMatrixController()
    controller.run()
