'''
new estimation method, sacrifices some accuracy for simpler and faster code
NOTE from here on the angles printed are now in (x,y) and not in (y,x) as in the codes before
'''
import numpy as np
import matplotlib.pyplot as plt
import time 
import math
import json
import paho.mqtt.client as mqtt
from math import atan, sin, cos, sqrt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev
from copy import deepcopy
import keyboard
import csv
import threading

# --- MQTT CONFIGURATION ---
MQTT_BROKER = "10.220.249.223"  # CHANGE TO YOUR BROKER IP
MQTT_PORT = 1883
MQTT_TOPIC = "servos/sync_command"
ESP_IDS = tuple(f"ESP_{i:02d}" for i in range(1, 6))

# Reverse the travelling gait by sending its frames in reverse time order.
# False restores the original direction; servo calibration stays unchanged.
REVERSE_PLAYBACK = True

# Commands address all five ESPs; feedback is currently tracked for ESP_01.
FEEDBACK_TOPIC = "servos/ESP_01/feedback"
RESPONSE_TIMEOUT = 5.0

response_event = threading.Event()
response_lock = threading.Lock()

expected_request_id = None
received_feedback = None
next_request_id = 1

CSV_FILE = "mpu_feedback.csv"


def on_feedback_message(client, userdata, msg):
    global received_feedback

    try:
        data = json.loads(msg.payload.decode())
        request_id = data.get("request_id")

        with response_lock:
            if request_id != expected_request_id:
                return

            received_feedback = data
            response_event.set()

    except (ValueError, TypeError):
        print("Invalid feedback received.")

INITIAL_DELAY = 1
PUBLISH_INTERVAL = 0.03  # <--- NEW: Seconds to wait between sending commands (Prevents ESP queue overflow)

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"✓ Connected successfully to {MQTT_BROKER}")
        client.subscribe(FEEDBACK_TOPIC)
        print(f"Subscribed to {FEEDBACK_TOPIC}")
    else:
        print(f"✗ Connection failed with code: {reason_code}")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"Disconnected with reason code: {reason_code}")

def build_movement_data(calculated_pairs):
    if len(calculated_pairs) != len(ESP_IDS):
        raise ValueError(
            f"Expected {len(ESP_IDS)} angle pairs for ESP_01 through ESP_05, "
            f"got {len(calculated_pairs)}"
        )
    return dict(zip(ESP_IDS, calculated_pairs))


def send_movement(calculated_pairs, target_timestamp_ms):
    """
    Maps five angle pairs to ESP_01 through ESP_05.
    Takes an exact target timestamp in milliseconds.
    """
    global next_request_id

    data_payload = build_movement_data(calculated_pairs)
    request_id = next_request_id
    next_request_id += 1

    payload = {
        "request_id": request_id,
        "ts": target_timestamp_ms,
        "data": data_payload
    }
    
    result = client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"✗ Failed to send message")


def send_command_wait_feedback(client, calculated_pairs, target_timestamp_ms):
    global expected_request_id, received_feedback, next_request_id

    data_payload = build_movement_data(calculated_pairs)
    request_id = next_request_id
    next_request_id += 1

    with response_lock:
        expected_request_id = request_id
        received_feedback = None
        response_event.clear()

    payload = {
        "request_id": request_id,
        "ts": target_timestamp_ms,
        "data": data_payload
    }

    info = client.publish(
        MQTT_TOPIC,
        json.dumps(payload),
        qos=0
    )

    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"Failed to publish command #{request_id}")

    info.wait_for_publish()

    print(
        f"Sent command #{request_id} | "
        f"Target time: {target_timestamp_ms} | "
        f"Angles: {calculated_pairs}"
    )

    if not response_event.wait(timeout=RESPONSE_TIMEOUT):
        raise TimeoutError(
            f"No matching feedback for command #{request_id} "
            f"within {RESPONSE_TIMEOUT} seconds. Stopping."
        )

    with response_lock:
        feedback = received_feedback.copy() if received_feedback else None

    if feedback is None:
        raise RuntimeError("Feedback data is missing.")

    if not feedback.get("sensor_valid", False):
        print(
            f"Warning: ESP_01 MPU readings invalid for command "
            f"#{request_id}; continuing movement without logging this sample."
        )
        return feedback

    fields = [
        "request_id", "command_ts",
        "roll", "pitch", "gx", "gy", "gz"
    ]

    file_exists = False
    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "request_id": request_id,
            "command_ts": feedback.get("command_ts"),
            "roll": feedback["roll"],
            "pitch": feedback["pitch"],
            "gx": feedback["gx"],
            "gy": feedback["gy"],
            "gz": feedback["gz"]
        })

    print(
        f"Feedback #{request_id} | "
        f"Roll: {feedback['roll']:.2f}° | "
        f"Pitch: {feedback['pitch']:.2f}°"
    )

    return feedback


# spline_func should be a function that returns x, y, z given a parameter `u`
def spline_func(u):
    return splev(u, tck)  

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Initial point on the curve
x0, y0, z0 = 0, 0, 0  

n=1
iterations =0
answers = []
error_margin=5
accuracylevel =100000
still_not_found =True
colors =['red','blue']

# User inputs for required number of segments, length, amplitudes, wavelength, and error margin
num_segments=6
req_length = 1.14
req_amplitude_z = 0.4
req_amplitude_y = 0
A1=0
num_path_points = 0
R=req_length/2
approximation_circle_dist=0.17

path_points_x = [0,3,4,6]
path_points_y = [0,3,4,6]
frequency =2# Math frequency for the sine wave

# Convert lists to a numpy array
points = np.array([path_points_x, path_points_y])

# Create a parameterized spline
tck, u = splprep(points, s=0)

# Generate points along the spline
u_new = np.linspace(0, 1, accuracylevel)  
x_new, y_new = splev(u_new, tck)

# Calculate the derivatives (tangent vectors) along the spline
dx, dy = splev(u_new, tck, der=1)

# Normalize the tangent vectors
magnitude = np.sqrt(dx**2 + dy**2)
dx_normalized = -dy / magnitude
dy_normalized = dx / magnitude

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_box_aspect([1, 1, 1])  

# Plot the original spline in 3D (XY plane)
# ax.plot(x_new, y_new, np.zeros_like(x_new), label='Original Spline', color='red')

# Initial sine wave plot in 3D
phase = 0
multiplier = 1.5*u_new
x_sine_wave = A1 + x_new + multiplier*req_amplitude_y * np.sin(2 * np.pi *frequency * u_new+1+phase) * dx_normalized
y_sine_wave = A1 + y_new + multiplier*req_amplitude_y * np.sin(2 * np.pi *frequency * u_new+1+phase) * dy_normalized
z_wave = req_amplitude_z * np.abs(np.sin(2 * np.pi *frequency * u_new+1+phase))

sine_wave_line, = ax.plot(x_sine_wave, y_sine_wave, z_wave, label='3D Spline with Sine Waves', color='green')

angles_ground_apparent =[]
angles_ground_real=[]
angles_relative=[]
angles_real =[]
answers_per_iteration =[]
angles_relative_per_iteration=[]
angles_real_per_iteration=[]

offset =0
answers_per_iteration.append([0,0,0])
once =True
phaseshiftercount=0
kitneAngleChahiye=200
while(phaseshiftercount<kitneAngleChahiye):
    for i in range(len(x_sine_wave)):
        if(sqrt((x_sine_wave[i]-answers_per_iteration[-1][0])**2+(y_sine_wave[i]-answers_per_iteration[-1][1])**2+(z_wave[i]-answers_per_iteration[-1][2])**2)>req_length):
            answers_per_iteration.append([x_sine_wave[i],y_sine_wave[i],z_wave[i]])
        if(len(answers_per_iteration)==num_segments+1):
            for j in range(0,len(answers_per_iteration)-1):
                angles_ground_apparent.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi])
                angles_ground_real.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][2]-answers_per_iteration[j][2])**2))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][1]-answers_per_iteration[j][1])**2))*180/np.pi])
                if j:
                    angles_relative_per_iteration.append(tuple([round(180+angles_ground_apparent[j][0]-angles_ground_apparent[j-1][0],3),round(180+angles_ground_apparent[j][1]-angles_ground_apparent[j-1][1],3)]))
                    angles_real_per_iteration.append(tuple([round(180+angles_ground_real[j][0]-angles_ground_real[j-1][0],3),round(180+angles_ground_real[j][1]-angles_ground_real[j-1][1],3)]))

            break
    if(len(answers_per_iteration)<num_segments+1):
        break
    # print("Then This appends...")
    answers.append(deepcopy(answers_per_iteration))#normal copy was giving some issues

    angles_real_per_iteration
    angles_real.append(deepcopy(angles_real_per_iteration))
    # print(angles_real)
    angles_relative.append(deepcopy(angles_relative_per_iteration))
    answers_per_iteration.clear()
    angles_real_per_iteration.clear()
    angles_ground_apparent.clear()
    angles_ground_real.clear()


    angles_relative_per_iteration.clear()
    answers_per_iteration.append([x_sine_wave[offset],y_sine_wave[offset],z_wave[offset]])
    phase +=5*(2*np.pi/kitneAngleChahiye)
    x_sine_wave =A1+ x_new + multiplier*req_amplitude_y * np.sin(2 * np.pi *frequency * u_new+1+phase) * dx_normalized
    y_sine_wave = A1+y_new + multiplier*req_amplitude_y * np.sin(2 * np.pi *frequency * u_new+1+phase) * dy_normalized
    z_wave = req_amplitude_z * np.abs(np.sin(2 * np.pi *frequency * u_new+1+phase))
    phaseshiftercount+=1

# while(offset<accuracylevel):
#     for i in range(offset,len(x_sine_wave)):
#         if(sqrt((x_sine_wave[i]-answers_per_iteration[-1][0])**2+(y_sine_wave[i]-answers_per_iteration[-1][1])**2+(z_wave[i]-answers_per_iteration[-1][2])**2)>req_length):
#             answers_per_iteration.append([x_sine_wave[i],y_sine_wave[i],z_wave[i]])
#         if(len(answers_per_iteration)==num_segments+1):
#             for j in range(0,len(answers_per_iteration)-1):
#                 angles_ground_apparent.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi])
#                 angles_ground_real.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][2]-answers_per_iteration[j][2])**2))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][1]-answers_per_iteration[j][1])**2))*180/np.pi])
#                 if j:
#                     angles_relative_per_iteration.append(tuple([round(180+angles_ground_apparent[j][0]-angles_ground_apparent[j-1][0],3),round(180+angles_ground_apparent[j][1]-angles_ground_apparent[j-1][1],3)]))
#                     angles_real_per_iteration.append(tuple([round(180+angles_ground_real[j][0]-angles_ground_real[j-1][0],3),round(180+angles_ground_real[j][1]-angles_ground_real[j-1][1],3)]))

#             break

            
#     if(len(answers_per_iteration)<num_segments+1):
#         break
        
#     answers.append(deepcopy(answers_per_iteration))
#     angles_real.append(deepcopy(angles_real_per_iteration))
#     angles_relative.append(deepcopy(angles_relative_per_iteration))
#     answers_per_iteration.clear()
#     angles_real_per_iteration.clear()
#     angles_ground_apparent.clear()
#     angles_ground_real.clear()
#     angles_relative_per_iteration.clear()
#     offset+=int(accuracylevel/150)
#     answers_per_iteration.append([x_sine_wave[offset],y_sine_wave[offset],z_wave[offset]])

for item in angles_real:
    print(item)#(horizontal,vertial)
max_dist =max(max(path_points_x),max(path_points_y))
ax.set_xlim(0,max_dist)
ax.set_ylim(0,max_dist)
ax.set_zlim(0,max_dist)

line, = ax.plot([], [], [], 'o-', lw=2)

def update(frame):
    data = np.array(answers[frame])  
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

ani = FuncAnimation(fig, update, frames=len(answers), interval=50, blit=False)

plt.legend()
plt.grid(True)
plt.show()

# --- INITIALIZE MQTT CLIENT ---
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect

client.message_callback_add(
    FEEDBACK_TOPIC,
    on_feedback_message
)

print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    exit(1)

time.sleep(1) # Wait for connection to stabilize

# --- MAIN SENDING LOOP ---
tempcount = 1
wantToSend=True

# Track the exact start time for the first batch of angles
next_execution_time = time.time() + 1.0

try:
    while(wantToSend):
        for angles_set in (reversed(angles_real) if REVERSE_PLAYBACK else angles_real):
            calculated_pairs = []
            
            for angle_pair in angles_set:  
                '''this modified angle is due to the design of the modules in v2(virtual rolling joint)'''
                angle_1 = atan(R*sin((180-angle_pair[0])*np.pi/180)/(R*cos((180-angle_pair[0])*np.pi/180)-approximation_circle_dist))*180/np.pi
                angle_2 = atan(R*sin((180-angle_pair[1])*np.pi/180)/(R*cos((180-angle_pair[1])*np.pi/180)-approximation_circle_dist))*180/np.pi
                
                # Format exactly as before: min 70, max 110 constraints
                s1 = min(115,max(90-int(angle_1),65))#this only for catterpillar design, for snake design it should be angle_1 instead of 0
                s2 = min(115,max(90-int(angle_2),65))
                calculated_pairs.append([s2, s1]) # Note the order swap to match (x,y) -> (servo1, servo2)
            
            # Convert float seconds into integer milliseconds for the MQTT payload
            target_ts_ms = int(next_execution_time * 1000)
            
            # Publish via MQTT
            # Send one command and wait for its matching MPU feedback
            send_command_wait_feedback(
                client,
                calculated_pairs,
                target_ts_ms
            )

            next_execution_time = max(
                next_execution_time + PUBLISH_INTERVAL,
                time.time() + 0.05
            )

            tempcount += 1
            
            
            if(keyboard.is_pressed('a')):
                print("paused")
                time.sleep(2)
                while(True):
                    if(keyboard.is_pressed('a')):
                        print("continuing")
                        # Realignment: Push the target timestamp into the future to account for the pause
                        next_execution_time = time.time() + INITIAL_DELAY
                        time.sleep(2)
                        break
                    if(keyboard.is_pressed('q')):
                        print("quitting")
                        wantToSend=False
                        break
                        
            if(not wantToSend):
                break
                
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    client.loop_stop()
    client.disconnect()
    print("✓ MQTT Disconnected safely.")
