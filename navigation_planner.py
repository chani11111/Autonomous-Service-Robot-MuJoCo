"""
navigation_planner.py
─────────────────────
S — Single Responsibility: אחראי אך ורק על החלטות ניווט.
     לא קורא חיישנים ישירות, לא שולח פקודות ישירות.

O — Open/Closed: ניתן להוסיף אסטרטגיית ניווט חדשה ללא שינוי קוד קיים.
L — Liskov: GlobalPlanner ו-ReactiveLayer ניתנים להחלפה זה בזה.
D — Dependency Inversion: מקבל dependencies מבחוץ.
"""

import numpy as np
from sensor_suite import SensorReading
from motion_controller import MotionCommand, SPEED_NORMAL, SPEED_SLOW, SPEED_TURN


# ─── מצבי הניווט ────────────────────────────────────────────────
class NavState:
    NAVIGATING  = "navigating"    # נסיעה רגילה לכיוון המטרה
    AVOIDING    = "avoiding"      # הימנעות ממכשול
    RECOVERING  = "recovering"    # שחזור ממבוי סתום
    ARRIVED     = "arrived"       # הגענו!


# ─── Interface — כל Planner חייב לממש ──────────────────────────
class INavigationStrategy:
    def compute(self,
                robot_x: float, robot_y: float, robot_yaw: float,
                target_x: float, target_y: float,
                sensor: SensorReading) -> MotionCommand:
        raise NotImplementedError


# ─── שכבה גלובלית ───────────────────────────────────────────────
class GlobalPlanner(INavigationStrategy):
    """
    מחשב כיוון כללי לעבר המטרה.
    משתמש ב-Pure Pursuit פשוט — מסתובב לכיוון המטרה ונוסע.
    """

    GOAL_RADIUS    = 0.35   # מטר — רדיוס "הגעה"
    ANGLE_THRESH   = 15.0   # מעלות — סף ליישור כיוון

    def compute(self,
                robot_x, robot_y, robot_yaw,
                target_x, target_y,
                sensor: SensorReading) -> MotionCommand:

        dx    = target_x - robot_x
        dy    = target_y - robot_y
        dist  = np.sqrt(dx**2 + dy**2)

        if dist < self.GOAL_RADIUS:
            return MotionCommand.stop()

        target_angle  = np.degrees(np.arctan2(dy, dx))
        angle_error   = target_angle - robot_yaw

        while angle_error >  180: angle_error -= 360
        while angle_error < -180: angle_error += 360

        if abs(angle_error) > self.ANGLE_THRESH * 3:
            if angle_error > 0:
                return MotionCommand.turn_left(SPEED_TURN)
            else:
                return MotionCommand.turn_right(SPEED_TURN)

        elif abs(angle_error) > self.ANGLE_THRESH:
            if angle_error > 0:
                return MotionCommand.curve_left(SPEED_NORMAL)
            else:
                return MotionCommand.curve_right(SPEED_NORMAL)

        else:
            speed = SPEED_SLOW if dist < 1.0 else SPEED_NORMAL
            return MotionCommand.forward(speed)


# ─── שכבה ריאקטיבית (מעודכנת לשדות פוטנציאל APF) ───────────────────
class ReactiveLayer(INavigationStrategy):
    """
    מגיב למכשולים בזמן אמת באמצעות שדות פוטנציאל מלאכותיים (APF).
    היעד מפעיל כוח משיכה, והמכשולים מפעילים כוח דחייה בפרם הרובוט.
    """

    REPULSE_GAIN = 7.0      # עוצמת כוח הדחייה מהמכשולים
    ATTRACT_GAIN = 1.0      # עוצמת כוח המשיכה אל היעד
    FRONT_WEIGHT = 2.5      # משקל בטיחות מוגבר לחזית להתחמקות ממכשולים דינמיים

    def compute(self,
                robot_x: float, robot_y: float, robot_yaw: float,
                target_x: float, target_y: float,
                sensor: SensorReading) -> MotionCommand:

        # 1. חישוב וקטור המשיכה אל היעד במערכת הצירים של העולם
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist_to_goal = np.sqrt(dx**2 + dy**2)

        attr_w_x = (dx / dist_to_goal) * self.ATTRACT_GAIN
        attr_w_y = (dy / dist_to_goal) * self.ATTRACT_GAIN

        # העברת וקטור המשיכה מהעולם למערכת הצירים המקומית של הרובוט (Local Frame)
        yaw_rad = np.radians(robot_yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        attr_x =  c * attr_w_x + s * attr_w_y
        attr_y = -s * attr_w_x + c * attr_w_y

        # 2. חישוב וקטור הדחייה מהמכשולים (ישירות בפרם המקומי של הרובוט)
        repulse_x = 0.0
        repulse_y = 0.0

        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        names = list(sensor.distances.keys())

        for name, angle_deg in zip(names, angles_deg):
            dist = sensor.distances[name]
            if dist <= 0:
                continue

            # דחייה חזקה יותר ככל שהעצם קרוב (ביחס ריבועי)
            repulse_strength = self.REPULSE_GAIN / max(dist**2, 0.01)

            # אם המכשול מגיע מלפנים, נגדיל משמעותית את הדחייה כדי לברוח מהכדור הדינמי
            if name in ["rf_front", "rf_front_left", "rf_front_right"]:
                repulse_strength *= self.FRONT_WEIGHT

            angle_rad = np.radians(angle_deg)
            # וקטור הדחייה פועל בכיוון הפוך למיקום המכשול
            repulse_x -= repulse_strength * np.cos(angle_rad)
            repulse_y -= repulse_strength * np.sin(angle_rad)

        # 3. סכימת הכוחות לוקטור תנועה משולב
        total_x = attr_x + repulse_x
        total_y = attr_y + repulse_y

        if total_x == 0 and total_y == 0:
            return MotionCommand.forward(SPEED_SLOW)

        # 4. תרגום הוקטור המשולב לשגיאת היגוי (heading error)
        heading_error = np.degrees(np.arctan2(total_y, total_x))

        while heading_error > 180: heading_error -= 360
        while heading_error < -180: heading_error += 360

        # קבלת החלטת היגוי חלקה והחלטית
        if heading_error > 35:
            return MotionCommand.turn_left(SPEED_TURN)
        elif heading_error < -35:
            return MotionCommand.turn_right(SPEED_TURN)
        elif heading_error > 10:
            return MotionCommand.curve_left(SPEED_NORMAL)
        elif heading_error < -10:
            return MotionCommand.curve_right(SPEED_NORMAL)
        else:
            return MotionCommand.forward(SPEED_NORMAL)


# ─── מנגנון התאוששות ─────────────────────────────────────────────
class RecoveryBehavior:
    BACKUP_STEPS  = 80
    TURN_STEPS    = 60

    def __init__(self):
        self._phase  = "backup"
        self._counter = 0

    def reset(self) -> None:
        self._phase   = "backup"
        self._counter = 0

    def step(self) -> tuple[MotionCommand, bool]:
        self._counter += 1

        if self._phase == "backup":
            if self._counter >= self.BACKUP_STEPS:
                self._phase   = "turn"
                self._counter = 0
            return MotionCommand.backward(SPEED_SLOW), False

        elif self._phase == "turn":
            if self._counter >= self.TURN_STEPS:
                return MotionCommand.stop(), True
            return MotionCommand.turn_right(SPEED_TURN), False

        return MotionCommand.stop(), True


# ─── NavigationPlanner — המאחד ───────────────────────────────────
class NavigationPlanner:
    STUCK_THRESHOLD    = 500
    PROGRESS_MIN_DIST  = 0.02

    def __init__(self, target_x: float, target_y: float):
        self.target_x = target_x
        self.target_y = target_y

        self._global   = GlobalPlanner()
        self._reactive = ReactiveLayer()
        self._recovery = RecoveryBehavior()

        self._state          = NavState.NAVIGATING
        self._stuck_counter  = 0
        self._last_pos       = None
        self._progress_timer = 0

    def set_target(self, x: float, y: float) -> None:
        self.target_x = x
        self.target_y = y
        self._state   = NavState.NAVIGATING

    def has_arrived(self) -> bool:
        return self._state == NavState.ARRIVED

    def get_state(self) -> str:
        return self._state

    def compute(self, robot_x: float, robot_y: float, robot_yaw: float, sensor: SensorReading) -> MotionCommand:
        dist = np.sqrt((self.target_x - robot_x)**2 + (self.target_y - robot_y)**2)

        if dist < GlobalPlanner.GOAL_RADIUS:
            self._state = NavState.ARRIVED
            return MotionCommand.stop()

        if self._state == NavState.RECOVERING:
            cmd, done = self._recovery.step()
            if done:
                self._state       = NavState.NAVIGATING
                self._stuck_counter = 0
            return cmd

        self._progress_timer += 1
        if self._last_pos is not None:
            moved = np.sqrt((robot_x - self._last_pos[0])**2 + (robot_y - self._last_pos[1])**2)
            if moved < self.PROGRESS_MIN_DIST:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0

        if self._progress_timer % 50 == 0:
            self._last_pos = (robot_x, robot_y)

        if self._stuck_counter >= self.STUCK_THRESHOLD:
            print("⚠️  תקוע! מפעיל שחזור...")
            self._state = NavState.RECOVERING
            self._recovery.reset()
            self._stuck_counter = 0
            return MotionCommand.backward(SPEED_SLOW)

        # כעת בזכות השינוי ב-SensorSuite, התנאי הזה יתפוס עצמים ממרחק של 1.4 מטרים
        if not sensor.is_clear():
            self._state = NavState.AVOIDING
            return self._reactive.compute(
                robot_x, robot_y, robot_yaw,
                self.target_x, self.target_y, sensor
            )

        self._state = NavState.NAVIGATING
        return self._global.compute(
            robot_x, robot_y, robot_yaw,
            self.target_x, self.target_y, sensor
        )