# [cite_start]Autonomous Intelligent Service Robot in a Dynamic Environment [cite: 337]

[cite_start]This project involves the development and simulation of an autonomous wheeled service robot operating within a dynamic office environment using the **MuJoCo** physics engine[cite: 337, 357]. [cite_start]The primary objective is to enable the robot to navigate from a starting point to a target location efficiently and safely, successfully avoiding both static furniture (desks) and moving obstacles[cite: 341, 342].

## Key Features
* [cite_start]**Hybrid Navigation Architecture:** Combines a high-level Global Planner (Pure Pursuit) for goal-seeking behavior with a real-time Reactive Avoidance Layer powered by Artificial Potential Fields (APF)[cite: 383, 385].
* **Proportional Kinematic Control:** Implements a continuous, smooth differential drive controller that translates vector fields directly into left and right wheel velocities, eliminating jerky movements.
* [cite_start]**SOLID Principles:** The software architecture strictly adheres to S.O.L.I.D. design patterns, ensuring complete decoupling between sensing, logic, and actuation[cite: 354, 362].
* [cite_start]**Sensor Filtering:** Includes an integrated low-pass noise filter within the SensorSuite to ensure robust obstacle distance processing[cite: 408].

---

## Project Structure

* [cite_start]`main.py` - The main entry point that executes the closed-loop control loop[cite: 400].
* [cite_start]`simulation_manager.py` - Handles the MuJoCo interface, environment rendering, and physics stepping[cite: 365].
* [cite_start]`navigation_planner.py` - Contains the logic for the Global Planner, Reactive Layer (APF), and Recovery Behavior[cite: 365, 385, 408].
* [cite_start]`sensor_suite.py` - Manages multi-directional Rangefinder data acquisition and preprocessing[cite: 349, 365].
* [cite_start]`motion_controller.py` - Converts navigation decisions into velocity commands for the robot's differential wheels[cite: 365].

---

## Prerequisites

Before running the simulation, ensure you have Python installed along with the required dependencies. You can install them via pip:

```bash
pip install mujoco numpy
```
How to Run
To start the simulation loop and open the passive MuJoCo viewer, simply execute the main script:

```Bash
python main.py
```
To terminate the simulation, close the MuJoCo visualization window or press Ctrl + C in your terminal.

Performance Metrics & KPIs   
The system evaluates success based on programmatic logging against the following target values defined in the Design Review:  

Collision Rate: 0% contact force on all scenario runs (Critical).  

Path Efficiency: Total distance traveled ÷ straight-line distance ratio 
≤1.5

Time to Goal: Destination arrival within 60 seconds on baseline test S1.  


Robustness: Dead-end recovery (back-up + wall-follow behavior) success rate 
≥80%
 (at least 4/5 attempts).  
