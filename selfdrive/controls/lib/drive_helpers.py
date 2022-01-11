import math
from cereal import car
from common.numpy_fast import clip, interp
from common.realtime import DT_MDL
from selfdrive.config import Conversions as CV
from selfdrive.modeld.constants import T_IDXS

# cruise button by neokii
ButtonType = car.CarState.ButtonEvent.Type
ButtonPrev = ButtonType.unknown
ButtonCnt = 0
LongPressed = False

# kph
V_CRUISE_MAX = 135
V_CRUISE_MIN = 5
V_CRUISE_DELTA = 5
V_CRUISE_ENABLE_MIN = 40
V_CRUISE_DELTA_MI = 5 * CV.MPH_TO_KPH
V_CRUISE_DELTA_KM = 10
LAT_MPC_N = 16
LON_MPC_N = 32
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
REGEN_THRESHOLD = 5

# this corresponds to 80deg/s and 20deg/s steering angle in a toyota corolla
MAX_CURVATURE_RATES = [0.03762194918267951, 0.003441203371932992]
MAX_CURVATURE_RATE_SPEEDS = [0, 35]

CRUISE_LONG_PRESS = 50
CRUISE_NEAREST_FUNC = {
  car.CarState.ButtonEvent.Type.accelCruise: math.ceil,
  car.CarState.ButtonEvent.Type.decelCruise: math.floor,
}
CRUISE_INTERVAL_SIGN = {
  car.CarState.ButtonEvent.Type.accelCruise: +1,
  car.CarState.ButtonEvent.Type.decelCruise: -1,
}


class MPC_COST_LAT:
  PATH = 1.0
  HEADING = 1.0
  STEER_RATE = 1.0


class MPC_COST_LONG:
  TTC = 5.0
  DISTANCE = 0.1
  ACCELERATION = 10.0
  JERK = 20.0


def rate_limit(new_value, last_value, dw_step, up_step):
  return clip(new_value, last_value + dw_step, last_value + up_step)


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


def update_v_cruise(v_cruise_kph, buttonEvents, button_timers, enabled, metric):
  # handle button presses. TODO: this should be in state_control, but a decelCruise press
  # would have the effect of both enabling and changing speed is checked after the state transition
  global ButtonCnt, LongPressed, ButtonPrev
  if enabled:
    if ButtonCnt:
      ButtonCnt += 1
    for b in buttonEvents:
      if b.pressed and not ButtonCnt and (b.type == ButtonType.accelCruise or \
                                          b.type == ButtonType.decelCruise):
        ButtonCnt = 1
        ButtonPrev = b.type
      elif not b.pressed and ButtonCnt:
        if not LongPressed and b.type == ButtonType.accelCruise:
          v_cruise_kph += 5 if metric else 2 * CV.MPH_TO_KPH
        elif not LongPressed and b.type == ButtonType.decelCruise:
          v_cruise_kph -= 5 if metric else 2 * CV.MPH_TO_KPH
        LongPressed = False
        ButtonCnt = 0
    if ButtonCnt > 80:
      LongPressed = True
      V_CRUISE_DELTA = V_CRUISE_DELTA_KM if metric else V_CRUISE_DELTA_MI
      if ButtonPrev == ButtonType.accelCruise:
        v_cruise_kph += V_CRUISE_DELTA - v_cruise_kph % V_CRUISE_DELTA
      elif ButtonPrev == ButtonType.decelCruise:
        v_cruise_kph -= V_CRUISE_DELTA - -v_cruise_kph % V_CRUISE_DELTA
      ButtonCnt %= 80
    v_cruise_kph = clip(v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

  return v_cruise_kph

def update_v_cruise_regen(v_ego, v_cruise_kph, regen, enabled):
  if enabled and regen:
    if (v_cruise_kph - v_ego * CV.MS_TO_KPH) < REGEN_THRESHOLD:
      v_cruise_kph -= REGEN_THRESHOLD - -v_cruise_kph % 5

  v_cruise_kph = clip(v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

  return v_cruise_kph


def initialize_v_cruise(v_ego, buttonEvents, v_cruise_last):
  for b in buttonEvents:
    # 250kph or above probably means we never had a set speed
    if b.type == car.CarState.ButtonEvent.Type.accelCruise and v_cruise_last < 250:
      return v_cruise_last

  return int(round(clip(v_ego * CV.MS_TO_KPH, V_CRUISE_ENABLE_MIN, V_CRUISE_MAX)))


def get_lag_adjusted_curvature(CP, v_ego, psis, curvatures, curvature_rates):
  if len(psis) != CONTROL_N:
    psis = [0.0 for i in range(CONTROL_N)]
    curvatures = [0.0 for i in range(CONTROL_N)]
    curvature_rates = [0.0 for i in range(CONTROL_N)]

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  delay = CP.steerActuatorDelay + .2
  current_curvature = curvatures[0]
  psi = interp(delay, T_IDXS[:CONTROL_N], psis)
  desired_curvature_rate = curvature_rates[0]

  # MPC can plan to turn the wheel and turn back before t_delay. This means
  # in high delay cases some corrections never even get commanded. So just use
  # psi to calculate a simple linearization of desired curvature
  curvature_diff_from_psi = psi / (max(v_ego, 1e-1) * delay) - current_curvature
  desired_curvature = current_curvature + 2 * curvature_diff_from_psi

  max_curvature_rate = interp(v_ego, MAX_CURVATURE_RATE_SPEEDS, MAX_CURVATURE_RATES)
  safe_desired_curvature_rate = clip(desired_curvature_rate,
                                          -max_curvature_rate,
                                          max_curvature_rate)
  safe_desired_curvature = clip(desired_curvature,
                                     current_curvature - max_curvature_rate/DT_MDL,
                                     current_curvature + max_curvature_rate/DT_MDL)
  return safe_desired_curvature, safe_desired_curvature_rate
