# framework16rgbmatrix
Python script to display stats and info on the rgb matrix for framework16.

This is something I have been toying around with. 
Ideas for displaying data about the system on these rgb led matrix modules by Joe Schroedl. - https://jschroedl.com/rgb-start/

Very rough script at the moment that needs much cleanup and organization, but it works (on Linux).
I recommend using udev rules to set persistent names for these input modules, so removing them and inserting them will keep the same names. You can look at my udev rule for an example, just update the serial to match your modules.
***
### Simple Installation Steps (Linux):

1) Place led_config.ini and run_rgb_matrix.py under /usr/local/share/framework16rgbmatrix/

2) Place rgbmatrix.service under /etc/systemd/system/
   -Run "systemctl daemon-reload" to make sysemd aware of it.

3) Place 98-rgb-led-matrix.rules under /etc/udev/rules.d/
   -Modify 98-rgb-led-matrix.rules to contain serial numvers of your rgb matrix modules. Get serials with something like 'lsusb -vv | grep "iProduct\|iSerial" | grep -A1 "RP2040"'.
   -Run the folloing to apply new rules: "udevadm control --reload-rules && udevadm trigger"

4) Edit the /usr/local/share/framework16rgbmatrix/led_config.ini config file and make any modifications needed.
   
5) Enable and start the service:
   systemctl enable --now rgbmatrix.service
***


-Configure them by modifying the led_config.ini file. Options are explained in comments.
-To re-arange the order of each module and enable ones you want to use, edit the panel_1_order and panel_2_order in the config. Currently available modules are the following:

            'battery1': (self.battery1_module, 12), # huge battery module, spiral fill
            'battery2': (self.battery2_module, 4), # small battery module 
            'cpu_load1': (self.cpu_load1_module, 16), # cpu load, u-shaped lines
            'cpu_load2': (self.cpu_load2_module, 16), # cpu load, spiral (inspired bu Jeremy's awesome monitor for the gray-scale framework modules)
            'cpu_temp': (self.cpu_temp_module, 4),  # CPU temp module, 4 rows
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
            'power1': (self.power1_module, 4), # overall battery watts draining, plug icon when plugged in and charging.
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


<img src="docs/images/IMG_20250402_004616_HDR.jpg" height="400" />
