"""
motion_controller.py
────────────────────
S — Single Responsibility: אחראי אך ורק על המרת החלטות ניווט לפקודות גלגלים.
     לא מחליט לאן ללכת — רק מבצע.

D — Dependency Inversion: מקבל SimulationManager מבחוץ.
"""

import numpy as np
from simulation_manager import SimulationManager


# מהירויות סטנדרטיות (rad/s)
SPEED_FAST   =  5.0
SPEED_NORMAL =  3.5
SPEED_SLOW   =  2.0
SPEED_TURN   =  3.0
SPEED_STOP   =  0.0


class MotionCommand:
    """
    מבנה נתונים המייצג פקודת תנועה.
    NavigationPlanner מייצר → MotionController מבצע.
    """
    def __init__(self, left: float, right: float, label: str = ""):
        self.left  = left
        self.right = right
        self.label = label

    # ── פקודות מוכנות מראש (factory methods) ──────────────────
    @staticmethod
    def forward(speed: float = SPEED_NORMAL) -> "MotionCommand":
        return MotionCommand(speed, speed, "forward")

    @staticmethod
    def turn_left(speed: float = SPEED_TURN) -> "MotionCommand":
        """סיבוב שמאלה במקום"""
        return MotionCommand(-speed, speed, "turn_left")

    @staticmethod
    def turn_right(speed: float = SPEED_TURN) -> "MotionCommand":
        """סיבוב ימינה במקום"""
        return MotionCommand(speed, -speed, "turn_right")

    @staticmethod
    def curve_left(speed: float = SPEED_NORMAL) -> "MotionCommand":
        """עקומה שמאלה (גלגל שמאל איטי יותר)"""
        return MotionCommand(speed * 0.3, speed, "curve_left")

    @staticmethod
    def curve_right(speed: float = SPEED_NORMAL) -> "MotionCommand":
        """עקומה ימינה (גלגל ימין איטי יותר)"""
        return MotionCommand(speed, speed * 0.3, "curve_right")

    @staticmethod
    def backward(speed: float = SPEED_NORMAL) -> "MotionCommand":
        return MotionCommand(-speed, -speed, "backward")

    @staticmethod
    def stop() -> "MotionCommand":
        return MotionCommand(0.0, 0.0, "stop")

    def __repr__(self) -> str:
        return f"MotionCommand({self.label}: L={self.left:.1f}, R={self.right:.1f})"


class MotionController:
    """
    אחראי על:
      - קבלת MotionCommand מה-NavigationPlanner
      - המרה לפקודות ctrl לגלגלים
      - הגבלת מהירויות מקסימום (safety clamp)
      - לוגינג של פקודות לניתוח ביצועים

    לא יודע כלום על חיישנים או ניווט.
    """

    MAX_SPEED = 6.0  # מהירות מקסימום מוחלטת (rad/s)

    def __init__(self, sim: SimulationManager):
        self._sim = sim
        self._last_command: MotionCommand | None = None
        self._command_log: list = []   # לניתוח KPI בהמשך

    def execute(self, cmd: MotionCommand) -> None:
        """
        מבצע פקודת תנועה — המתודה המרכזית.
        מחיל clamp על מהירויות ומעדכן את ה-actuators.
        """
        left  = np.clip(cmd.left,  -self.MAX_SPEED, self.MAX_SPEED)
        right = np.clip(cmd.right, -self.MAX_SPEED, self.MAX_SPEED)

        self._sim.set_wheel_velocities(left, right)
        self._last_command = cmd
        self._command_log.append(cmd.label)

    def stop(self) -> None:
        """עצירה מיידית"""
        self.execute(MotionCommand.stop())

    def get_last_command(self) -> MotionCommand | None:
        return self._last_command

    def get_command_history(self) -> list:
        return list(self._command_log)
