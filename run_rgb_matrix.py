#!/usr/bin/env python3

import configparser
import serial
import time
import os
import sys
import psutil
import logging
import numpy as np
import glob

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

            effective_max_read = self.highest_read_rate * 0.50
            effective_max_write = self.highest_write_rate * 0.50

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
        self.last_active_interfaces = set()

    def get(self):
        try:
            net_io = psutil.net_io_counters()
            sent_usage = net_io.bytes_sent
            recv_usage = net_io.bytes_recv

            # Detect network connection change
            stats = psutil.net_if_stats()
            current_active = {iface for iface, stat in stats.items() if stat.isup}
            if current_active != self.last_active_interfaces:
                logger.info(f"Network connection changed from {self.last_active_interfaces} to {current_active}")
                self.highest_sent_rate = 0.00001
                self.highest_recv_rate = 0.00001
                self.sent_usage_history = [sent_usage]
                self.recv_usage_history = [recv_usage]
                self.history_times = [time.time()]
                self.last_active_interfaces = current_active
            else:
                self.last_active_interfaces = current_active

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

        if int(self.config['Settings']['number_of_panels']) == 2:
            self.panels = {
                0: {'serial': None, 'device': self.config['Settings']['panel_1_dev']},
                1: {'serial': None, 'device': self.config['Settings']['panel_2_dev']}
            }
        else: # not 2, then must be 1. 
            self.panels = {
                0: {'serial': None, 'device': self.config['Settings']['panel_1_dev']}
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
        self.colors = {  # R, G, B
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 100, 0),
            'dark_orange': (200, 50, 0),
            'light_green': (144, 238, 144),
            'light_blue': (173, 216, 230),
            'purple': (128, 0, 128),
            'brown': (165, 42, 42),
            'aqua': (0, 255, 255),
            'pink': (255, 105, 180),
            'white': (255, 255, 255),
            'black': (0, 0, 0) # off
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
            'C': [[0,1,1], [1,0,0], [1,0,0], [0,1,1]],
            'D': [[1,1,0], [1,0,1], [1,0,1], [1,1,0]],
            'F': [[1,1,1], [1,0,0], [1,1,0], [1,0,0]],
            'G': [[0,1,1], [1,0,0], [1,0,1], [0,1,1]],
            'I': [[1,1,1], [0,1,0], [0,1,0], [1,1,1]],
            'O': [[0,1,0], [1,0,1], [1,0,1], [0,1,0]],
            'P': [[1,1,1], [1,0,1], [1,1,1], [1,0,0]],
            'U': [[1,0,1], [1,0,1], [1,0,1], [1,1,1]],
            'W': [[1,0,1], [1,0,1], [1,1,1], [1,1,1]],
            ' ': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]  # Blank space
        }

        # Missing module backoff
        self.serial_retry_delays = {0: 0.1, 1: 0.1}  # Initial delay for each panel
        self.serial_retry_counts = {0: 0, 1: 0}      # Retry count for each panel
        self.max_retry_delay = 20.0                   # Max disconnected delay (seconds)
        self.backoff_factor = 2.0                    # Double delay each retry
        # Lid pull backoff
        self.lid_poll_interval = 3.0  # Poll every 3 seconds
        self.last_lid_check_time = 0

        # Cache dGPU sleep time
        dgpu_autosuspend_path = f"{self.dgpu_dev}/device/power/autosuspend_delay_ms"
        try:
            with open(dgpu_autosuspend_path, 'r') as f:
                autosuspend_ms = int(f.read().strip())
            self.dgpu_sleep_time = int((autosuspend_ms / 1000) + 1)
        except Exception as e:
            logger.warning(f"Failed to read dGPU autosuspend: {e}")
            self.dgpu_sleep_time = 6  # Fallback

        # Cache CPU temp path
        self.cpu_temp_path = None
        for hwmon in os.listdir('/sys/class/hwmon'):
            name_path = f'/sys/class/hwmon/{hwmon}/name'
            if os.path.exists(name_path):
                try:
                    with open(name_path, 'r') as f:
                        if f.read().strip() == 'k10temp':
                            self.cpu_temp_path = f'/sys/class/hwmon/{hwmon}/temp1_input'
                            break
                except:
                    pass

    def interpolate_color(self, color1: tuple, color2: tuple, factor: float) -> tuple:
        """Interpolate between two RGB colors."""
        return tuple(int(a + (b - a) * factor) for a, b in zip(color1, color2))

    def load_config(self):
        logger.debug("Loading configuration")
        self.config.read('led_config.ini')
        if not self.config.sections():
            logger.info("Creating default configuration")
            self.config['Settings'] = {
                'number_of_panels': '2',
                'panel_1_dev': '/dev/ttyRGBM1',
                'panel_2_dev': '/dev/ttyRGBM2',
                'panel_1_order': 'cpu_temp,cpu_load2,gpu2,dgpu_temp',
                'panel_2_order': 'battery2,power1,line_pink,ram3,line_brown,net2,line_brown,drive1',
                'brightness_source': '/sys/class/backlight/amdgpu_bl2/brightness',
                'cpu_temp': 'k10temp',
                'igpu_dev': '/sys/class/drm/card2',
                'dgpu_dev': '/sys/class/drm/card1',
                'max_brightness': '150',
                'min_brightness': '1',
                'debug': 'false',
                'clock1_colors': 'red,orange,yellow,green,blue',  # 5 colors for digits
                'clock1_type': '12',  # Default to 12-hour clock
                'clock2_colors': 'red,orange,yellow,green,blue,purple',  # Default colors for 6 characters
                'cpu_text_colors': 'aqua,blue,aqua',  # Default colors for "C", "P", "U"
                'gpu_text_colors': 'aqua,blue,aqua'  # Default colors for "G", "P", "U"
            }
            with open('led_config.ini', 'w') as configfile:
                self.config.write(configfile)

        # Cache base paths as instance attrs for easy access
        self.igpu_dev = self.config['Settings'].get('igpu_dev', '/sys/class/drm/card2')
        self.dgpu_dev = self.config['Settings'].get('dgpu_dev', '/sys/class/drm/card1')
        self.brightness_source = self.config['Settings'].get('brightness_source', '/sys/class/backlight/amdgpu_bl2/brightness')
        self.max_brightness_path = '/'.join(self.brightness_source.split('/')[:-1]) + '/max_brightness'


        # Add others as needed, e.g., self.cpu_temp = self.config['Settings'].get('cpu_temp', 'k10temp')

    def find_hwmon_file_path(self, base_dev, filename='temp1_input'):
        """
        Dynamically find the hwmon subdir containing the specified file (e.g., 'temp1_input').
        
        Args:
            base_dev (str): Base device path, e.g., '/sys/class/drm/card2'.
            filename (str): File to search for, e.g., 'temp1_input'.
        
        Returns:
            str or None: Full path to the file if found, else None.
        """
        hwmon_base = f"{base_dev}/device/hwmon"
        if not os.path.exists(hwmon_base):
            logger.warning(f"hwmon dir not found: {hwmon_base}")
            return None
        
        for subdir in os.listdir(hwmon_base):
            candidate_path = f"{hwmon_base}/{subdir}/{filename}"
            if os.path.isfile(candidate_path):  # Use isfile for efficiency
                logger.debug(f"Found {filename} in hwmon subdir: {subdir}")
                return candidate_path
        
        logger.warning(f"{filename} not found in any hwmon subdir under {hwmon_base}")
        return None

    def get_igpu_load_path(self):
        """Derive iGPU load path from base device."""
        return f"{self.igpu_dev}/device/gpu_busy_percent"

    def get_igpu_mem_total_path(self):
        """Derive iGPU total memory path from base device."""
        return f"{self.igpu_dev}/device/mem_info_vram_total"

    def get_igpu_mem_used_path(self):
        """Derive iGPU used memory path from base device."""
        return f"{self.igpu_dev}/device/mem_info_vram_used"

    def get_igpu_temp_path(self):
        """Dynamically derive iGPU temp path."""
        return self.find_hwmon_file_path(self.igpu_dev, 'temp1_input')

    def get_dgpu_load_path(self):
        """Derive dGPU load path from base device."""
        return f"{self.dgpu_dev}/device/gpu_busy_percent"

    def get_dgpu_mem_total_path(self):
        """Derive dGPU total memory path from base device."""
        return f"{self.dgpu_dev}/device/mem_info_vram_total"

    def get_dgpu_mem_used_path(self):
        """Derive dGPU used memory path from base device."""
        return f"{self.dgpu_dev}/device/mem_info_vram_used"

    def get_dgpu_temp_path(self):
        """Dynamically derive dGPU temp path."""
        return self.find_hwmon_file_path(self.dgpu_dev, 'temp1_input')

    def get_dgpu_powerstate_path(self):
        """Derive dGPU powerstate path."""
        return f"{self.dgpu_dev}/device/power_state"

    # Optional: Generic helper for common suffixes (even more flexible)
    def get_device_path(self, base_dev, suffix):
        """Generic: Build path like f'{base_dev}/{suffix}'."""
        return f"{base_dev}/{suffix}"
    # Usage: self.get_device_path(self.igpu_dev, 'device/gpu_busy_percent')  # Same as get_igpu_load_path()

    def get_brightness(self):
        try:
            with open(self.brightness_source, 'r') as f:
                brightness = int(f.read().strip())
            logger.debug(f"Got brightness: {brightness}")

            with open(self.max_brightness_path, 'r') as f:
                max_brightness_source = int(f.read().strip())
            logger.debug(f"Got max_brightness_source: {max_brightness_source}")

            # led matrix configured ranges
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

    def get_gpu_load(self, gpu_type='igpu'): # Or pass 'dgpu'
        """Get GPU utilization percentage from sysfs"""

        if gpu_type == 'igpu':
            path = self.get_igpu_load_path()
        elif gpu_type == 'dgpu':
            path = self.get_dgpu_load_path()
        else:
            raise ValueError(f"Unknown GPU load type: {gpu_type}")
        
        if not path:
            raise FileNotFoundError(f"Load path not discovered for {gpu_type}")

        try:
            with open(path, 'r') as f:
                gpu_load = float(f.read().strip())
            logger.debug(f"GPU load ({gpu_type}): {gpu_load}%")
            return max(0.0, min(100.0, gpu_load)) / 100.0  # Return as fraction (0.0 to 1.0)
        except Exception as e:
            logger.warning(f"Failed to read GPU load ({gpu_type}): {e}")
            return 0.0

    def get_gpu_temp(self, gpu_type='igpu'): # Or pass 'dgpu'
        """Get GPU temperature in degrees Celsius from hwmon sysfs"""
        if gpu_type == 'igpu':
            path = self.get_igpu_temp_path()
        elif gpu_type == 'dgpu':
            path = self.get_dgpu_temp_path()
        else:
            raise ValueError(f"Unknown GPU temp type: {gpu_type}")
        
        if not path:
            raise FileNotFoundError(f"Temp path not discovered for {gpu_type}")
        try:
            with open(path, 'r') as f:
                temp_millidegrees = int(f.read().strip())
            temp_celsius = temp_millidegrees / 1000.0  # Convert to C
            logger.debug(f"Got {gpu_type}: {temp_celsius}°C")
            return temp_celsius
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read {gpu_type}: {e}")
            return 0.0  # Or return fallback, e.g., 0.0

    def get_gpu_mem_percent(self, gpu_type='igpu'):
        """Get GPU memory usage percentage"""
        if gpu_type == 'igpu':
            total_file = self.get_igpu_mem_total_path()
            used_file = self.get_igpu_mem_used_path()
        elif gpu_type == 'dgpu':
            total_file = self.get_dgpu_mem_total_path()
            used_file = self.get_dgpu_mem_used_path()
        else:
            raise ValueError(f"Unknown GPU type: {gpu_type}")
        
        if not total_file or not used_file:
            raise FileNotFoundError(f"Path not discovered for {gpu_type}")
        try:
            with open(total_file, 'r') as f:
                mem_total = int(f.read().strip())
            with open(used_file, 'r') as f:
                mem_used = int(f.read().strip())
            if mem_total > 0:
                mem_percent = (mem_used / mem_total) * 100.0
                logger.debug(f"GPU memory {mem_percent}%")
                return max(0.0, min(100.0, mem_percent)) / 100.0
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to read GPU memory {gpu_type}: {e}")
            return 0.0

    def get_dgpu_powerstate(self):
        """Get dGPU power state"""
        try:
            powerstate_file = self.get_dgpu_powerstate_path()
            with open(powerstate_file, 'r') as f:
                state = f.read().strip().lower()
            logger.debug(f"dGPU power state: {state}")
            return state
        except Exception as e:
            logger.warning(f"Failed to read dGPU power state: {e}")
            return "unknown"

    def get_dgpu_metrics(self) -> tuple[float, float, float, bool, str]:
        """Get dGPU load, temp, mem %; returns (load, temp, mem, sleeping, powerstate)."""
        # Initialize attrs if needed
        for attr in ['last_dgpu_poll_time', 'last_dgpu_powerstate_time', 'dgpu_sleeping',
                    'last_dgpu_load', 'last_dgpu_temp', 'last_dgpu_mem_percent',
                    'last_dgpu_powerstate', 'dgpu_load_zero_count']:
            if not hasattr(self, attr):
                setattr(self, attr, 0 if 'time' in attr or 'count' in attr else ('unknown' if 'state' in attr else 0.0))

        current_time = time.time()
        # Poll powerstate periodically
        if not self.dgpu_sleeping or current_time - self.last_dgpu_powerstate_time >= self.dgpu_sleep_time:
            powerstate = self.get_dgpu_powerstate()
            self.last_dgpu_powerstate = powerstate
            self.last_dgpu_powerstate_time = current_time
        else:
            powerstate = self.last_dgpu_powerstate

        if powerstate == "d3cold":
            load, temp, mem = self.last_dgpu_load, self.last_dgpu_temp, self.last_dgpu_mem_percent
            self.dgpu_sleeping = True
            self.last_dgpu_poll_time = current_time
            self.dgpu_load_zero_count = 0  # Reset counter in D3cold
            logger.debug(f"dGPU D3cold: sleeping={self.dgpu_sleeping}, temp={temp}, zero_count={self.dgpu_load_zero_count}")
            return load, temp, mem, True, powerstate
        elif self.dgpu_sleeping and current_time - self.last_dgpu_poll_time < self.dgpu_sleep_time:
            load, temp, mem = self.last_dgpu_load, self.last_dgpu_temp, self.last_dgpu_mem_percent
            self.dgpu_load_zero_count = 0  # Reset counter while sleeping
            logger.debug(f"dGPU sleeping: temp={temp}, time_left={self.dgpu_sleep_time - (current_time - self.last_dgpu_poll_time):.1f}s, zero_count={self.dgpu_load_zero_count}")
            return load, temp, mem, True, powerstate
        else:
            # Poll dGPU metrics
            load = self.get_gpu_load('dgpu')
            temp = self.get_gpu_temp('dgpu')
            mem = self.get_gpu_mem_percent('dgpu')
            self.last_dgpu_load, self.last_dgpu_temp, self.last_dgpu_mem_percent = load, temp, mem
            if load <= 0.0:
                self.dgpu_load_zero_count += 1
                if self.dgpu_load_zero_count >= 10:
                    self.dgpu_sleeping = True
                    self.last_dgpu_poll_time = current_time
                    self.dgpu_load_zero_count = 0  # Reset after entering sleep
                    logger.debug(f"dGPU entering sleep: load={load}, zero_count={self.dgpu_load_zero_count}")
            else:
                self.dgpu_sleeping = False
                self.dgpu_load_zero_count = 0  # Reset on non-zero load
                logger.debug(f"dGPU active: sleeping={self.dgpu_sleeping}, load={load}, zero_count={self.dgpu_load_zero_count}")
            return load, temp, mem, self.dgpu_sleeping, powerstate


    def is_lid_closed(self):
        """Check if laptop lid is closed (Linux-specific)"""
        try:
            with open('/proc/acpi/button/lid/LID0/state', 'r') as f:
                state = f.read().strip()
                is_closed = 'closed' in state.lower()
                logger.debug(f"Lid state: {'closed' if is_closed else 'open'}")
                return is_closed
        except FileNotFoundError:
            logger.warning("Lid state file not found, assuming lid is open")
            return False
        except Exception as e:
            logger.warning(f"Error checking lid state: {e}")
            return self.lid_closed  # Fallback to last known state

    def connect_panels(self):
        for panel_id, panel in self.panels.items():
            if not panel['serial'] and self.running and not self.lid_closed:
                device = self.panels[panel_id]['device']
                retry_delay = self.serial_retry_delays[panel_id]
                try:
                    panel['serial'] = serial.Serial(panel['device'], 115200, timeout=1)
                    time.sleep(2)
                    response = panel['serial'].readline()
                    self.serial_retry_delays[panel_id] = 0.1  # Reset delay
                    self.serial_retry_counts[panel_id] = 0    # Reset count
                    logger.info(f"Connected to panel {panel_id} at {panel['device']}, response: {response}")
                except serial.SerialException as e:
                    self.serial_retry_counts[panel_id] += 1
                    self.serial_retry_delays[panel_id] = min(
                        self.max_retry_delay,
                        retry_delay * self.backoff_factor
                    )
                    logger.warning(
                        f"Failed to connect to panel {panel_id} at {device}: {e}. "
                        f"Retry {self.serial_retry_counts[panel_id]} after {self.serial_retry_delays[panel_id]}s"
                    )
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
        if not self.cpu_temp_path:
            logger.warning("k10temp path not cached")
            return 50.0
        try:
            with open(self.cpu_temp_path, 'r') as f:
                return int(f.read().strip()) / 1000.0
        except Exception as e:
            logger.warning(f"Failed to read CPU temp: {e}")
            return 50.0

    def cpu_load1_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        cpu_load = psutil.cpu_percent(percpu=True)
        try:
            temp = self.get_cpu_temp()
        except:
            temp = 50

        temp_factor = max(0, min(1, (temp - 50) / 50))
        temp_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)
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
        temp_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)
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
        if percent <= 15:
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
        gpu_load = self.get_gpu_load('igpu')
        gpu_temp = self.get_gpu_temp('igpu')

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

    def cpu_text_module(self, start_row):
        """Display 'CPU' text in the first 4 rows with configurable colors"""
        frame = [(0, 0, 0)] * self.total_leds
    
        # Get cpu_text colors from config, default to aqua, blue, purple
        cpu_colors_str = self.config['Settings'].get('cpu_text_colors', 'aqua,blue,purple')
        color_names = [name.strip() for name in cpu_colors_str.split(',')]
        # Ensure we have exactly 3 colors, padding with white or trimming excess
        while len(color_names) < 3:
            color_names.append('white')
        cpu_colors = [self.colors.get(name, self.colors['white']) for name in color_names[:3]]
    
        # Define grids for "GPU"
        c_grid = self.alpha_grids['C']
        p_grid = self.alpha_grids['P']
        u_grid = self.alpha_grids['U']
        
        # Draw "GPU" across cols 0-8 with configurable colors
        for row in range(4):
            for col in range(3):
                if c_grid[row][col]:
                    frame[(start_row + row) * 9 + col] = cpu_colors[0]  # Color for "C"
                if p_grid[row][col]:
                    frame[(start_row + row) * 9 + (3 + col)] = cpu_colors[1]  # Color for "P"
                if u_grid[row][col]:
                    frame[(start_row + row) * 9 + (6 + col)] = cpu_colors[2]  # Color for "U"
    
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
        g_grid = self.alpha_grids['G']
        p_grid = self.alpha_grids['P']
        u_grid = self.alpha_grids['U']
        
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
        """Dual GPU monitor with 'OFF' when in D3cold, now 8 rows"""
        frame = [(0, 0, 0)] * self.total_leds
    
        # Rows 0-3: iGPU Info
        igpu_load = self.get_gpu_load('igpu')
        igpu_temp = self.get_gpu_temp('igpu')
        igpu_mem_percent = self.get_gpu_mem_percent('igpu')
    
        # "i" in yellow (columns 0-2)
        i_grid = self.alpha_grids['I']
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
        load_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)
    
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
                mem_color = self.interpolate_color(self.colors['aqua'], self.colors['aqua'], factor)
                frame[(start_row + 3) * 9 + (8 - col)] = mem_color
    
        # Rows 4-7: dGPU Info
        dgpu_load, dgpu_temp, dgpu_mem_percent, dgpu_sleeping, dgpu_powerstate = self.get_dgpu_metrics()

        # "d" in dark orange (columns 0-2)
        d_grid = self.alpha_grids['D']
        for row in range(4):
            for col in range(3):
                if d_grid[row][col]:
                    frame[(start_row + 4 + row) * 9 + col] = self.colors['dark_orange']
        # Determine dGPU state
        if dgpu_powerstate == "d3cold":
            # Show scrolling zig-zag for power off
            off_grid_1 = [[1,0,0,0,0,0,0,1], [0,1,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,1,1,0,0,0]]
            off_grid_2 = [[0,0,0,0,1,1,0,0], [0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,1], [1,1,0,0,0,0,0,0]]
            display_width = 6
            word_length = 8
            gap = 0
            total_pattern_length = word_length + gap
            t = (time.time() % 1)
            # t = (current_time % 1)
            offset = int(t * total_pattern_length)
            for row in range(4):
                for col in range(display_width):
                    pattern_col = (col + offset) % word_length
                    if pattern_col < word_length:
                        if off_grid_1[row][pattern_col]:
                            frame[(start_row + 4 + row) * 9 + col + 3] = self.colors['purple']
                        elif off_grid_2[row][pattern_col]:
                            frame[(start_row + 4 + row) * 9 + col + 3] = self.colors['blue']

            """  
            # Show scrolling "OFF"
            o_grid = self.alpha_grids['O']
            f_grid = self.alpha_grids['F']
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
                        frame[(start_row + 4 + row) * 9 + col + 3] = self.colors['dark_orange']
                    elif 3 <= pattern_col < 6 and f_grid[row][pattern_col - 3]:
                        frame[(start_row + 4 + row) * 9 + col + 3] = self.colors['dark_orange']
                    elif 6 <= pattern_col < 9 and f_grid[row][pattern_col - 6]:
                        frame[(start_row + 4 + row) * 9 + col + 3] = self.colors['dark_orange'] """
        else:
            # dGPU Load (rows 4, 5, 6)
            target_load_pixels = int(dgpu_load * max_load_leds * 3 + 0.5)
            if self.current_dgpu_load < target_load_pixels:
                self.current_dgpu_load += 1
            elif self.current_dgpu_load > target_load_pixels:
                self.current_dgpu_load -= 1
            temp_factor = max(0.0, min(1.0, (dgpu_temp - 50) / 50))
            load_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)
        
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
                    mem_color = self.interpolate_color(self.colors['aqua'], self.colors['aqua'], factor)
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
            battery_paths = glob.glob('/sys/class/power_supply/BAT*')
            if not battery_paths:
                watts_int = 0
            else:
                bat_dir = battery_paths[0]  # First BAT*
                try:
                    with open(f'{bat_dir}/current_now', 'r') as f:
                        current_now = int(f.read().strip())  # in microamperes
                    with open(f'{bat_dir}/voltage_now', 'r') as f:
                        voltage_now = int(f.read().strip())  # in microvolts
                    # Watts = (current in A) * (voltage in V)
                    # Convert microamperes to amperes (*1e-6) and microvolts to volts (*1e-6)
                    # So, (current_now * 1e-6) * (voltage_now * 1e-6) = watts / 1e12
                    watts = (current_now * voltage_now) / 1e12
                    watts_int = max(0, min(99, int(watts + 0.5)))
                except Exception as e:
                    logger.warning(f"Failed to read power from {bat_dir}: {e}")
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
        temp_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)

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

        # Get dGPU metrics
        _, dgpu_temp, _, dgpu_sleeping, dgpu_powerstate = self.get_dgpu_metrics()

        # Determine dGPU state
        if dgpu_powerstate == "d3cold":
            # Show 0 temp for power off
            off_grid = [[0,0,0,0,1,0,0,0,0], [1,1,0,1,0,1,0,1,1], [1,1,0,1,0,1,0,1,1], [0,0,0,0,1,0,0,0,0]]
            display_width = 9
            word_length = 9
            total_pattern_length = word_length
            for row in range(4):
                for col in range(9):
                    if off_grid[row][col]:
                        frame[(start_row + row) * 9 + col] = self.colors['green']
        else:
            # Calculate temp factor (0 to 1) for color transition
            temp_factor = max(0, min(1, (dgpu_temp - 50) / 50))
            temp_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)

            # Convert temperature to integer and extract digits
            temp_int = int(dgpu_temp + 0.5)  # Round to nearest integer
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
        
    def cpu_load3_module(self, start_row):
        frame = [(0, 0, 0)] * self.total_leds
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            temp = self.get_cpu_temp()  # Get CPU temp in Celsius, same as cpu_load2_module
            logger.debug(f"CPU Usage: percent={cpu_percent:.1f}%, temp={temp:.1f}°C")
        except Exception as e:
            logger.warning(f"Failed to get CPU usage/temp: {e}")
            cpu_percent = 0
            temp = 50.0  # Default temp
        
        # Edge coordinates for 9x9 grid
        edge_coords = [
            (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 8), (5, 8),
            (6, 7), (7, 6), (8, 5), (8, 4), (8, 3), (7, 2), (6, 1),
            (5, 0), (4, 0), (3, 0), (2, 1), (1, 2), (0, 3)
        ]

        # Circle coordinates for fill order
        circle_coords = [
            (4, 3), (4, 2), (4, 1), (4, 0), 
            (3, 0), (3, 1), (3, 2), (2, 1), (3, 3), (2, 2), (2, 3), (1, 2), (1, 3), (0, 3), (3, 4), (2, 4), (1, 4), (0, 4),
            (0, 5), (1, 5), (2, 5), (1, 6), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), 
            (5, 8), (5, 7), (5, 6), (6, 7), (5, 5), (6, 6), (7, 6), (6, 5), (5, 4), (7, 5), (8, 5), (6, 4), (7, 4), (8, 4), 
            (8, 3), (7, 3), (6, 3), (7, 2), (5, 3), (6, 2), (6, 1), (5, 2), (5, 1), (5, 0)
        ]

        """
        # Flip coordinates horizontally and vertically
        # Horizontal flip: col -> (8 - col)
        # Vertical flip: row -> (8 - row)
        edge_coords = [(8 - row, 8 - col) for row, col in original_edge_coords]
        circle_coords = [(8 - row, 8 - col) for row, col in original_circle_coords]
        """

        # Draw the circle edge
        edge_color = self.colors['aqua']
        for row, col in edge_coords:
            frame[(start_row + row) * 9 + col] = edge_color

        # Draw the purple vertical line from row 4, col 4 to row 7, col 4
        for col in range(1, 5):  # Rows 4, 5, 6, 7
            frame[(start_row + 4) * 9 + col ] = self.colors['orange']

        # Calculate temp factor (0 to 1) for color transition, same as cpu_load2_module
        temp_factor = max(0, min(1, (temp - 50) / 50))
        fill_color = self.interpolate_color(self.colors['blue'], self.colors['red'], temp_factor)

        # Fill the circle based on CPU usage percentage with smoothing, same rate as cpu_load2_module
        total_leds = len(circle_coords)
        target_leds = int((cpu_percent / 100) * total_leds + 0.5)
        if not hasattr(self, 'current_cpu_load3_leds'):
            self.current_cpu_load3_leds = 0  # Initialize in first call (or add to __init__)
        if self.current_cpu_load3_leds < target_leds:
            self.current_cpu_load3_leds += 2
        elif self.current_cpu_load3_leds > target_leds:
            self.current_cpu_load3_leds -= 2

        for i in range(min(self.current_cpu_load3_leds, total_leds)):
            row, col = circle_coords[i]
            frame[(start_row + row) * 9 + col] = fill_color

        return frame

    def run(self):
        num_panels = int(self.config['Settings']['number_of_panels'])
        modules = {
            'battery1': (self.battery1_module, 12), # huge battery module, spiral fill, 12 rows
            'battery2': (self.battery2_module, 4), # small battery module, 4 rows
            'cpu_text': (self.cpu_text_module, 4),  # CPU text module, 4 rows. Just shows "CPU" in configurable colors.
            'cpu_load1': (self.cpu_load1_module, 16), # cpu load, u-shaped lines, temp-color line down middle
            'cpu_load2': (self.cpu_load2_module, 16), # cpu load, spiral, temp-color line down middle
            'cpu_load3': (self.cpu_load3_module, 9), # cpu load, circle dial
            'cpu_temp': (self.cpu_temp_module, 4),  # CPU temp module, 4 rows temperature numbers
            'drive1': (self.drive1_module, 9), # drive activity, circle dial
            'net1': (self.net1_module, 4), # network acivity, incremental bars
            'net2': (self.net2_module, 3), # network acivity, left/right
            'ram1': (self.ram1_module, 4), # fills up as rows
            'ram2': (self.ram2_module, 2), # Small ram bar, good for lower ram machines.
            'ram3': (self.ram3_module, 9), # ram usage, circle dial
            'gpu_text': (self.gpu_text_module, 4),  # GPU text module, 4 rows. Just shows "GPU" in configurable colors.
            'gpu1': (self.gpu1_module, 4), # iGPU only (use if you do not have the dGPU installed)
            'gpu2': (self.gpu2_module, 8),  # iGPU and dGPU load and vram usage graphs
            'dgpu_temp': (self.dgpu_temp_module, 4), # dGPU temp module, 4 rows temperature numbers
            'clock1': (self.clock1_module, 8), # digital clock, 12 or 24 hour time
            'clock2': (self.clock2_module, 8),
            'power1': (self.power1_module, 4), # overall battery watts while draining, plug image when charging.
            'line_red': (lambda row: self.line_module(row, 'line_red'), 1), # seperation lines, use one row in color listed
            'line_blue': (lambda row: self.line_module(row, 'line_blue'), 1),
            'line_green': (lambda row: self.line_module(row, 'line_green'), 1),
            'line_yellow': (lambda row: self.line_module(row, 'line_yellow'), 1),
            'line_orange': (lambda row: self.line_module(row, 'line_orange'), 1),
            'line_purple': (lambda row: self.line_module(row, 'line_purple'), 1),
            'line_pink': (lambda row: self.line_module(row, 'line_pink'), 1),
            'line_rainbow': (lambda row: self.line_module(row, 'line_rainbow'), 1), # this one is animated
            'line_aqua': (lambda row: self.line_module(row, 'line_aqua'), 1),
            'line_white': (lambda row: self.line_module(row, 'line_white'), 1),
            'line_black': (lambda row: self.line_module(row, 'line_black'), 1), # this one is off
            'line_brown': (lambda row: self.line_module(row, 'line_brown'), 1)
        }

        self.connect_panels()

        try:
            last_retry_times = {0: 0, 1: 0}  # Track last retry time per panel
            while self.running:
                current_time = time.time()
                current_lid_state = self.is_lid_closed()
                # Check lid state only every lid_poll_interval seconds
                if current_time - self.last_lid_check_time >= self.lid_poll_interval:
                    current_lid_state = self.is_lid_closed()
                    self.last_lid_check_time = current_time
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
                            # Only retry if enough time has passed
                            if current_time - last_retry_times[panel_id] >= self.serial_retry_delays[panel_id]:
                                logger.info(f"Attempting to reconnect panel {panel_id}")
                                self.connect_panels()
                                last_retry_times[panel_id] = current_time
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
