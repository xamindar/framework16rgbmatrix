# framework16rgbmatrix
Python script to display stats and info on the rgb matrix for framework16.

This is something I have been toying around with. 
Ideas for displaying data about the system on these rgb led matrix modules by Joe Schroedl. - https://jschroedl.com/rgb-start/

This was only designed to work on Linux.
I recommend using udev rules to set persistent names for these input modules, so removing them and inserting them will keep the same names. You can look at my udev rule for an example, just update the serial to match your modules.
***

### Features:

Various graphical modules that display the following:

-battery level graphs

-CPU load percentage graphs

-CPU temperature display

-i/dGPU load percentage graphs

-dGPU temperature display

-Drive read/write activity

-Network in/out activity

-Memory usage percentage with high swap usage alert

-Clocks

-Power usage in watts while on battery

-Cosmetic line separators and labels

### Installation Steps (Linux):

0. Dependancies: Python 3.6+, pyserial, psutil, numpy.
   
   -Arch example: "pacman -S python-pyserial python-psutil python-numpy"

   -Arch example: "pacman -S python-pyserial python-psutil python-numpy"
   -Fedora example: "dnf install python3-pyserial python3-psutil python3-numpy"

1. Place **led_config.ini** and **run_rgb_matrix.py** under /usr/local/share/framework16rgbmatrix/
2. Place **rgbmatrix.service** under /etc/systemd/system/

   -Run "systemctl daemon-reload" to make systemd aware of it.

3. Place **98-rgb-led-matrix.rules** under /etc/udev/rules.d/

   -Modify **98-rgb-led-matrix.rules** to contain serial numbers of your rgb matrix modules. Get serials with something like 'lsusb -vv | grep "iProduct\|iSerial" | grep -A1 "RP2040"'.
   -On Fedora 43+, you may need to change the group from "uucp" to "dialout".

   -Run the following to apply new rules: "udevadm control --reload-rules && udevadm trigger"

4. Edit the **/usr/local/share/framework16rgbmatrix/led_config.ini** config file and make any modifications needed.
   
5. Enable and start the service:

   systemctl enable --now rgbmatrix.service

***


-Configure them by modifying the led_config.ini file. Options are explained in comments.

-To re-arange the order of each module and enable/disable different ones, edit the panel_1_order and panel_2_order in the config. 

-There are a total of 32 rows of LEDs. If the list of modules for a panel exceed 32 rows, any module that exceeds pass row 32 will simply not display.

Currently available modules are the following (module name, function name, number of occuplied lines, explaination comment):

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

<img src="docs/images/IMG_20250402_004616_HDR.jpg" height="400" />
