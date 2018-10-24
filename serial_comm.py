import glob
import serial
from datetime import datetime
import time

# https://github.com/pyserial/pyserial/issues/216
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)
    # adding this from the answer of [gskielian] on a stackoverflow page - "How to export Pyserial reading
    # data into txt file or Matlab"
    def send_and_receive(self):
        self.s.write(b'a')

port = glob.glob('/dev/ttyACM*')[0]

# This should be the same as what you set for the arduino
baud_rate = 115200 

# creating file to write the arduino data
path = "%s_LOG.txt" % ('_'.join(str(datetime.now()).split()))

# open a serial port
arduino = serial.Serial(port, baud_rate, bytesize = serial.EIGHTBITS,
        parity = serial.PARITY_NONE)

#arduino.setDTR(False)
#time.sleep(2)

#arduino.flushInput()
#arduino.setDTR(True)

arduino_obj = ReadLine(arduino)
# arduino seems to reset itself when we start new communication,
# so wait until the reset begins

arduino_obj.s.setDTR(False)
time.sleep(2)

arduino_obj.s.flushInput()
arduino_obj.s.setDTR(True)

for i in range(5):
    line = arduino_obj.readline()
    print(line)

arduino_obj.send_and_receive()
time.sleep(3)

with open(path, 'wb') as f:
    while True:
        line = arduino_obj.readline()
        print(line)

'''with open(path, 'wb') as f:
    arduino_obj.send_and_receive() 
    while True:
        arduino_vals = arduino_obj.readline()
        f.write(arduino_vals)'''

'''with open(path, 'w+') as f:
    while True:
        line = arduino.read(arduino.in_waiting)
        f.writelines([line, "\t", "%s\n" %(datetime.now())])
'''

