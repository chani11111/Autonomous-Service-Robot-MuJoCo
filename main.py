"""
main.py
───────
נקודת כניסה ראשית — מחבר את כל הקלאסות.

סדר התלויות:
  SimulationManager
      ↓
  SensorSuite(sim)   MotionController(sim)
              ↓         ↓
           NavigationPlanner(target)
                  ↓
               main loop
"""

import time
import numpy as np
from simulation_manager import SimulationManager
from sensor_suite        import SensorSuite
from motion_controller   import MotionController
from navigation_planner  import NavigationPlanner, NavState

# ─── הגדרות ריצה ────────────────────────────────────────────────
TARGET_X       =  3.0    # מטרה — ציר X
TARGET_Y       =  0.0    # מטרה — ציר Y
PRINT_EVERY    =  100    # הדפסת סטטוס כל N צעדים
USE_FILTER     =  True   # שימוש בפילטר low-pass על החיישנים
MAX_STEPS      =  30000  # מקסימום צעדים לפני עצירה (בטיחות)


def print_status(step: int,
                 sim: SimulationManager,
                 sensor_reading,
                 planner: NavigationPlanner,
                 last_cmd) -> None:
    """הדפסת סטטוס מרוכז"""
    x, y, yaw = sim.get_robot_pose()
    dist = np.sqrt((TARGET_X - x)**2 + (TARGET_Y - y)**2)
    state = planner.get_state()

    state_icon = {
        NavState.NAVIGATING: "🟢",
        NavState.AVOIDING:   "🟡",
        NavState.RECOVERING: "🔴",
        NavState.ARRIVED:    "✅",
    }.get(state, "❓")

    print(f"\n── Step {step:>5} ──────────────────────────────────────")
    print(f"  Pos    : x={x:+.3f}  y={y:+.3f}  yaw={yaw:+.1f}°")
    print(f"  Target : ({TARGET_X}, {TARGET_Y})  dist={dist:.3f}m")
    print(f"  State  : {state_icon} {state}")
    if last_cmd:
        print(f"  Command: {last_cmd}")
    print(f"  Sensors:")
    print(sensor_reading)


def run() -> None:
    print("=" * 55)
    print("  Autonomous Service Robot — שלב 2 (SOLID)")
    print(f"  מטרה: ({TARGET_X}, {TARGET_Y})")
    print("  סגור את חלון MuJoCo לסיום")
    print("=" * 55)

    # ── אתחול כל הקלאסות ──────────────────────────────────────
    sim      = SimulationManager(target_x=TARGET_X, target_y=TARGET_Y)
    sensors  = SensorSuite(sim)
    motor    = MotionController(sim)
    planner  = NavigationPlanner(target_x=TARGET_X, target_y=TARGET_Y)

    step     = 0
    last_cmd = None
    dt       = sim.get_timestep()

    with sim:   # פותח את חלון MuJoCo
        while sim.is_running() and step < MAX_STEPS:

            step_start = time.time()

            # 1. קריאת חיישנים
            if USE_FILTER:
                reading = sensors.read_filtered(alpha=0.75)
            else:
                reading = sensors.read()

            # 2. עדכון עצמים דינמיים
            sim.move_dynamic_objects(step * dt)

            # 3. חישוב פקודת ניווט
            x, y, yaw = sim.get_robot_pose()
            cmd        = planner.compute(x, y, yaw, reading)
            last_cmd   = cmd

            # 4. ביצוע הפקודה
            motor.execute(cmd)

            # 5. צעד פיזיקה + עדכון תצוגה
            sim.step()
            sim.sync_viewer()

            # 6. הדפסת סטטוס
            if step % PRINT_EVERY == 0:
                print_status(step, sim, reading, planner, last_cmd)

            # 7. בדיקת הגעה
            if planner.has_arrived():
                motor.stop()
                sim.set_wheel_velocities(0, 0)
                print(f"\n✅  הגענו למטרה ({TARGET_X}, {TARGET_Y})! צעד {step}")
                while sim.is_running():
                    sim.step()
                    sim.sync_viewer()
                    time.sleep(dt)
                break

            # 8. שמירת קצב
            elapsed = time.time() - step_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            step += 1

        if step >= MAX_STEPS:
            print(f"\n⏹  הגענו למקסימום צעדים ({MAX_STEPS})")

    # ── סיכום ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  סיכום ריצה")
    print("=" * 55)
    x, y, _ = sim.get_robot_pose()
    dist     = np.sqrt((TARGET_X - x)**2 + (TARGET_Y - y)**2)
    history  = motor.get_command_history()
    counts   = {cmd: history.count(cmd) for cmd in set(history)}

    print(f"  צעדים כולל  : {step}")
    print(f"  מרחק סופי   : {dist:.3f}m מהמטרה")
    print(f"  פקודות שבוצעו:")
    for cmd_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / max(len(history), 1) * 100
        print(f"    {cmd_name:<15} {count:>5}x  ({pct:.1f}%)")


if __name__ == "__main__":
    run()