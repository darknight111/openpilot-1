#!/usr/bin/env python3
from cereal import car
from common.numpy_fast import interp
from math import fabs
from selfdrive.config import Conversions as CV
from selfdrive.car.gm.values import CAR, CruiseButtons, AccState, CarControllerParams
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from common.params import Params
GearShifter = car.CarState.GearShifter
ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    #params = CarControllerParams()
    #return params.ACCEL_MIN, params.ACCEL_MAX
    v_current_kph = current_speed * CV.MS_TO_KPH
    
    gas_max_bp = [0.0, 5.0, 9.0, 35.0] #felger
    gas_max_v =  [0.4, 0.5, 0.7, 0.7]
    
#    gas_max_bp = [0., 10., 25., 40., 60., 80., 100., 110.]
#    gas_max_v = [0.5, 0.52, 0.55, 0.6, 0.67, 0.67, 0.67, 0.67]

    brake_max_bp = [0, 70., 130.]
    brake_max_v = [-4., -3., -2.1]

    return interp(v_current_kph, brake_max_bp, brake_max_v), interp(v_current_kph, gas_max_bp, gas_max_v)

  # Volt determined by iteratively plotting and minimizing error for f(angle, speed) = steer.
  @staticmethod
  def get_steer_feedforward_volt(desired_angle, v_ego):
    # maps [-inf,inf] to [-1,1]: sigmoid(34.4 deg) = sigmoid(1) = 0.5
    # 1 / 0.02904609 = 34.4 deg ~= 36 deg ~= 1/10 circle? Arbitrary?
    desired_angle *= 0.02904609
    sigmoid = desired_angle / (1 + fabs(desired_angle))
    return 0.10006696 * sigmoid * (v_ego + 3.12485927)

  def get_steer_feedforward_function(self):
    if self.CP.carFingerprint in [CAR.VOLT]:
      return self.get_steer_feedforward_volt
    else:
      return CarInterfaceBase.get_steer_feedforward_default

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)
    ret.carName = "gm"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.gm)]
    ret.pcmCruise = False  # stock cruise control is kept off

    # GM port is a community feature
    # TODO: make a port that uses a car harness and it only intercepts the camera
    ret.communityFeature = True

    # Presence of a camera on the object bus is ok.
    # Have to go to read_only if ASCM is online (ACC-enabled cars),
    # or camera is on powertrain bus (LKA cars without ACC).
    ret.enableGasInterceptor = 0x201 in fingerprint[0]
    ret.openpilotLongitudinalControl = ret.enableGasInterceptor

    tire_stiffness_factor = 0.5

    ret.minSteerSpeed = 8 * CV.KPH_TO_MS
    ret.steerRateCost = 0.5 # def : 2.0
    ret.steerActuatorDelay = 0.  # def: 0.2 Default delay, not measured yet

    ret.minEnableSpeed = -1
    ret.mass = 1625. + STD_CARGO_KG
    ret.wheelbase = 2.60096
    ret.steerRatio = 16.8
    ret.steerRatioRear = 0.
    ret.centerToFront = ret.wheelbase * 0.49 # wild guess
    
    ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[10., 41.0], [10., 41.0]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.18, 0.26], [0.01, 0.02]]
    ret.lateralTuning.pid.kdBP = [0.]
    ret.lateralTuning.pid.kdV = [0.319]  # very sensitive to changes greater than 0.001
    ret.lateralTuning.pid.kf = 0.0001

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    # longitudinal
#    ret.longitudinalTuning.kpBP = [-5.* CV.KPH_TO_MS, 0., 10.*CV.KPH_TO_MS, 25.*CV.KPH_TO_MS, 40.*CV.KPH_TO_MS, 60.*CV.KPH_TO_MS, 80.*CV.KPH_TO_MS, 100.*CV.KPH_TO_MS, 110.*CV.KPH_TO_MS]
#    ret.longitudinalTuning.kpV = [1.5, 1.25, 1.1, 0.81, 0.61, 0.57, 0.54, 0.51, 0.48]
    ret.longitudinalTuning.kpBP = [0.0, 5.0, 10.0, 20.0, 35.0] 
    ret.longitudinalTuning.kpV = [0.6, 0.95, 1.19, 1.27, 1.18] 
  
    ret.longitudinalTuning.kiBP = [0., 35.]
    ret.longitudinalTuning.kiV = [0.31, 0.26]
#    ret.longitudinalTuning.kiV = [0.075, 0.05]
    
    #ret.longitudinalTuning.kfBP = [15., 20., 25.]
    #ret.longitudinalTuning.kfV = [1., 0.5, 0.2]
    
    ret.longitudinalTuning.deadzoneBP = [0., 30.*CV.KPH_TO_MS]
    ret.longitudinalTuning.deadzoneV = [0., 0.001]
    # ret.longitudinalActuatorDelay = 0.2
    ret.longitudinalActuatorDelayLowerBound = 0.15
    ret.longitudinalActuatorDelayUpperBound = 0.15
    
    ret.startAccel = -1.0 # Toyota requets 0 instantly, the hands off to some controller
    ret.stopAccel = -2.0 # Toyota requests -0.4 when stopped
    ret.startingAccelRate = 0.6 # brake_travel/s while releasing on restart
    ret.stoppingDecelRate = 2.4 # brake_travel/s while trying to stop
    ret.vEgoStopping = 0.5 # when car starts requesting stopping accel
    ret.vEgoStarting = 0.5 needs to be > or == vEgoStopping
    ret.stoppingControl = True
    
    ret.steerLimitTimer = 0.4
    ret.radarTimeStep = 0.0667  # GM radar runs at 15Hz instead of standard 20Hz

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.cp.update_strings(can_strings)

    ret = self.CS.update(self.cp)

    ret.cruiseState.enabled = self.CS.main_on or self.CS.adaptive_Cruise
    ret.canValid = self.cp.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    buttonEvents = []

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons and self.CS.prev_cruise_buttons != CruiseButtons.INIT:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_buttons != CruiseButtons.UNPRESS:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        be.type = ButtonType.accelCruise
      elif but == CruiseButtons.DECEL_SET:
        be.type = ButtonType.decelCruise
      elif but == CruiseButtons.CANCEL:
        be.type = ButtonType.cancel
      elif but == CruiseButtons.MAIN:
        be.type = ButtonType.altButton3
      buttonEvents.append(be)

    ret.buttonEvents = buttonEvents

    events = self.create_common_events(ret)

    if ret.vEgo < self.CP.minEnableSpeed:
      events.add(EventName.belowEngageSpeed)
    if self.CS.park_brake:
      events.add(EventName.parkBrake)
    if ret.vEgo < self.CP.minSteerSpeed:
      events.add(car.CarEvent.EventName.belowSteerSpeed)
    if self.CP.enableGasInterceptor:
      if self.CS.adaptive_Cruise and ret.brakePressed:
        self.CS.adaptive_Cruise = False
        self.CS.enable_lkas = False
        events.add(EventName.pedalPressed)

    # handle button presses
    if self.CP.enableGasInterceptor:
      if not self.CS.main_on : #lat dis-engage
        for b in ret.buttonEvents:
          if (b.type == ButtonType.decelCruise and not b.pressed) and not self.CS.adaptive_Cruise:
            self.CS.adaptive_Cruise = True
            self.CS.enable_lkas = True
            events.add(EventName.buttonEnable)
            break
          if (b.type == ButtonType.accelCruise and not b.pressed) and not self.CS.adaptive_Cruise:
            self.CS.adaptive_Cruise = True
            self.CS.enable_lkas = False
            events.add(EventName.buttonEnable)
            break
          if (b.type == ButtonType.cancel and b.pressed) and self.CS.adaptive_Cruise:
            self.CS.adaptive_Cruise = False
            self.CS.enable_lkas = False
            events.add(EventName.buttonCancel)
            break
          if (b.type == ButtonType.altButton3 and b.pressed) : #and self.CS.adaptive_Cruise
            self.CS.adaptive_Cruise = False
            self.CS.enable_lkas = True
            break
      else :#lat engage
        for b in ret.buttonEvents:
          if not self.CS.adaptive_Cruise and (b.type == ButtonType.altButton3 and b.pressed) : #and self.CS.adaptive_Cruise
            self.CS.adaptive_Cruise = False
            self.CS.enable_lkas = False
            break

    else :
      if self.CS.main_on: #wihtout pedal case
        self.CS.adaptive_Cruise = False
        self.CS.enable_lkas = True
      else:
        self.CS.adaptive_Cruise = False
        self.CS.enable_lkas = False

    ret.events = events.to_msg()

    # copy back carState packet to CS
    self.CS.out = ret.as_reader()

    return self.CS.out

  def apply(self, c):
    hud_v_cruise = c.hudControl.setSpeed
    if hud_v_cruise > 70:
      hud_v_cruise = 0

    # For Openpilot, "enabled" includes pre-enable.
    can_sends = self.CC.update(c.enabled, self.CS, self.frame,
                               c.actuators,
                               hud_v_cruise, c.hudControl.lanesVisible,
                               c.hudControl.leadVisible, c.hudControl.visualAlert)

    self.frame += 1
    return can_sends
