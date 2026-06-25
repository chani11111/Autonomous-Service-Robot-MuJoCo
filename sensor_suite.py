"""
sensor_suite.py
───────────────
S — Single Responsibility: אחראי אך ורק על קריאה ועיבוד נתוני חיישנים.
     לא מחליט לאן ללכת — רק אומר מה רואים.

I — Interface Segregation: חושף רק את המתודות שה-NavigationPlanner צריך.
D — Dependency Inversion: מקבל SimulationManager כ-dependency (לא יוצר אותו).
"""

import numpy as np
from simulation_manager import SimulationManager


# שמות החיישנים לפי סדר — חשוב שיתאים לסדר ב-XML
SENSOR_NAMES = [
    "rf_front",       # 0°
    "rf_front_left",  # 45°
    "rf_left",        # 90°
    "rf_rear_left",   # 135°
    "rf_rear",        # 180°
    "rf_rear_right",  # 225°
    "rf_right",       # 270°
    "rf_front_right", # 315°
]

# ספי מרחק מעודכנים להתמודדות עם מכשולים דינמיים מהירים
DANGER_THRESHOLD  = 1.4   # מטר — מתחת לזה מופעלת ההתחמקות הריאקטיבית
WARNING_THRESHOLD = 2.2   # מטר — טווח אזהרה כללי
NO_OBSTACLE       = -1.0  # ערך MuJoCo כשאין מכשול בטווח


class SensorReading:
    """
    מבנה נתונים פשוט שמייצג קריאת חיישנים מעובדת.
    מועבר מ-SensorSuite ל-NavigationPlanner.
    """
    def __init__(self, raw: np.ndarray):
        self.raw: np.ndarray = raw                   # 8 ערכים גולמיים
        self.distances: dict = self._parse(raw)      # dict שם → מרחק
        self.danger_directions: list = []            # כיוונים מסוכנים
        self.warning_directions: list = []           # כיוונים עם אזהרה
        self._classify()

    def _parse(self, raw: np.ndarray) -> dict:
        """ממיר מערך גולמי ל-dict קריא"""
        return {
            name: float(val)
            for name, val in zip(SENSOR_NAMES, raw)
        }

    def _classify(self) -> None:
        """מסווג כל כיוון כ-danger / warning / clear"""
        for name, dist in self.distances.items():
            if dist == NO_OBSTACLE:
                continue
            if dist < DANGER_THRESHOLD:
                self.danger_directions.append(name)
            elif dist < WARNING_THRESHOLD:
                self.warning_directions.append(name)

    def is_clear(self) -> bool:
        """האם אין מכשולים מסוכנים בכלל"""
        return len(self.danger_directions) == 0

    def get_front_distance(self) -> float:
        """מרחק למכשול קדימה (-1 אם אין)"""
        return self.distances["rf_front"]

    def get_left_distance(self) -> float:
        return self.distances["rf_left"]

    def get_right_distance(self) -> float:
        return self.distances["rf_right"]

    def obstacle_ahead(self) -> bool:
        """האם יש מכשול מסוכן קדימה"""
        front_sensors = ["rf_front", "rf_front_left", "rf_front_right"]
        return any(
            self.distances[s] != NO_OBSTACLE and self.distances[s] < DANGER_THRESHOLD
            for s in front_sensors
        )

    def obstacle_on_left(self) -> bool:
        left_sensors = ["rf_left", "rf_front_left", "rf_rear_left"]
        return any(
            self.distances[s] != NO_OBSTACLE and self.distances[s] < DANGER_THRESHOLD
            for s in left_sensors
        )

    def obstacle_on_right(self) -> bool:
        right_sensors = ["rf_right", "rf_front_right", "rf_rear_right"]
        return any(
            self.distances[s] != NO_OBSTACLE and self.distances[s] < DANGER_THRESHOLD
            for s in right_sensors
        )

    def __repr__(self) -> str:
        lines = ["SensorReading:"]
        for name, dist in self.distances.items():
            if dist == NO_OBSTACLE:
                status = "  clear  "
                bar = ""
            else:
                status = f"{dist:6.3f}m"
                bar_len = max(0, min(20, int(dist * 6)))
                color = "🔴" if dist < DANGER_THRESHOLD else ("🟡" if dist < WARNING_THRESHOLD else "🟢")
                bar = color + "█" * bar_len
            lines.append(f"  {name:<18} {status}  {bar}")
        return "\n".join(lines)


class SensorSuite:
    """
    אחראי על:
      - קריאת נתוני Rangefinder גולמיים מה-SimulationManager
      - עיבוד הנתונים ל-SensorReading מסווג
      - החזרת קריאה נקייה ל-NavigationPlanner
    """

    def __init__(self, sim: SimulationManager):
        self._sim = sim
        self._last_reading: SensorReading | None = None

    def read(self) -> SensorReading:
        raw = self._sim.get_raw_sensor_data()
        self._last_reading = SensorReading(raw)
        return self._last_reading

    def get_last_reading(self) -> SensorReading | None:
        return self._last_reading

    def apply_low_pass_filter(self, new_raw: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        if self._last_reading is None:
            return new_raw
        old_raw = self._last_reading.raw
        filtered = np.where(
            (new_raw > 0) & (old_raw > 0),
            alpha * new_raw + (1 - alpha) * old_raw,
            new_raw
        )
        return filtered

    def read_filtered(self, alpha: float = 0.7) -> SensorReading:
        raw    = self._sim.get_raw_sensor_data()
        f_raw  = self.apply_low_pass_filter(raw, alpha)
        self._last_reading = SensorReading(f_raw)
        return self._last_reading