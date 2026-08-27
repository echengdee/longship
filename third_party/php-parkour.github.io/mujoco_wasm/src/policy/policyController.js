import * as ort from 'onnxruntime-web';

const DEFAULT_COMMAND = [0, 0, 0];

const parseCsv = (value) => {
  if (Array.isArray(value)) {
    return value.slice();
  }
  if (value == null) {
    return [];
  }
  if (typeof value !== 'string') {
    return [String(value)];
  }
  return value
    .split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
};

const parseNumberCsv = (value) => parseCsv(value).map((entry) => Number(entry));

const ensureLength = (values, length, fillValue = 0) => {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const source = values[i];
    out[i] = Number.isFinite(source) ? source : fillValue;
  }
  return out;
};

const quatRotate = (q, v) => {
  const w = q[0];
  const x = q[1];
  const y = q[2];
  const z = q[3];
  const vx = v[0];
  const vy = v[1];
  const vz = v[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx)
  ];
};

const quatApplyInverse = (quat, vec) => {
  const x = quat[1];
  const y = quat[2];
  const z = quat[3];
  const vx = vec[0];
  const vy = vec[1];
  const vz = vec[2];
  const t0 = (y * vz - z * vy) * 2;
  const t1 = (z * vx - x * vz) * 2;
  const t2 = (x * vy - y * vx) * 2;
  return [
    vx - quat[0] * t0 + (y * t2 - z * t1),
    vy - quat[0] * t1 + (z * t0 - x * t2),
    vz - quat[0] * t2 + (x * t1 - y * t0)
  ];
};

const quatMultiply = (a, b) => {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
  ];
};

const yawQuatFromRoot = (rootQuat) => {
  const w = rootQuat[0];
  const x = rootQuat[1];
  const y = rootQuat[2];
  const z = rootQuat[3];
  const siny = 2 * (w * z + x * y);
  const cosy = 1 - 2 * (y * y + z * z);
  const yaw = Math.atan2(siny, cosy);
  const half = yaw * 0.5;
  return [Math.cos(half), 0, 0, Math.sin(half)];
};

const quatToEulerXYZ = (quat) => {
  const w = quat[0];
  const x = quat[1];
  const y = quat[2];
  const z = quat[3];
  const t0 = 2 * (w * x + y * z);
  const t1 = 1 - 2 * (x * x + y * y);
  const roll = Math.atan2(t0, t1);
  let t2 = 2 * (w * y - z * x);
  t2 = Math.max(-1, Math.min(1, t2));
  const pitch = Math.asin(t2);
  const t3 = 2 * (w * z + x * y);
  const t4 = 1 - 2 * (y * y + z * z);
  const yaw = Math.atan2(t3, t4);
  return [roll, pitch, yaw];
};

const normalizeQuat = (q) => {
  const norm = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm === 0) {
    return [1, 0, 0, 0];
  }
  return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
};

const quatInvert = (q) => [q[0], -q[1], -q[2], -q[3]];

const normalizeVec3 = (v) => {
  const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (norm === 0) {
    return [0, 0, -1];
  }
  return [v[0] / norm, v[1] / norm, v[2] / norm];
};

const readNameAt = (namesArray, startIdx) => {
  let endIdx = startIdx;
  while (endIdx < namesArray.length && namesArray[endIdx] !== 0) {
    endIdx++;
  }
  const textDecoder = new TextDecoder('utf-8');
  return textDecoder.decode(namesArray.subarray(startIdx, endIdx));
};

const textDecoder = new TextDecoder('utf-8');

const readVarint = (bytes, offset) => {
  let result = 0;
  let shift = 0;
  let pos = offset;
  while (pos < bytes.length) {
    const byte = bytes[pos++];
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return { value: result, offset: pos };
    }
    shift += 7;
  }
  throw new Error('Invalid varint encoding');
};

const skipField = (bytes, offset, wireType) => {
  switch (wireType) {
    case 0: {
      const res = readVarint(bytes, offset);
      return res.offset;
    }
    case 1:
      return offset + 8;
    case 2: {
      const lenRes = readVarint(bytes, offset);
      return lenRes.offset + lenRes.value;
    }
    case 5:
      return offset + 4;
    default:
      throw new Error(`Unsupported wire type: ${wireType}`);
  }
};

const parseMetadataEntry = (bytes) => {
  let offset = 0;
  let key = '';
  let value = '';
  while (offset < bytes.length) {
    const tagRes = readVarint(bytes, offset);
    const tag = tagRes.value;
    offset = tagRes.offset;
    const field = tag >> 3;
    const wire = tag & 0x7;
    if (wire !== 2) {
      offset = skipField(bytes, offset, wire);
      continue;
    }
    const lenRes = readVarint(bytes, offset);
    const len = lenRes.value;
    const start = lenRes.offset;
    const end = start + len;
    const str = textDecoder.decode(bytes.subarray(start, end));
    if (field === 1) {
      key = str;
    } else if (field === 2) {
      value = str;
    }
    offset = end;
  }
  return { key, value };
};

const parseOnnxMetadata = (bytes) => {
  const metadata = new Map();
  let offset = 0;
  while (offset < bytes.length) {
    const tagRes = readVarint(bytes, offset);
    const tag = tagRes.value;
    offset = tagRes.offset;
    const field = tag >> 3;
    const wire = tag & 0x7;
    if (field === 14 && wire === 2) {
      const lenRes = readVarint(bytes, offset);
      const len = lenRes.value;
      const start = lenRes.offset;
      const end = start + len;
      const entry = parseMetadataEntry(bytes.subarray(start, end));
      if (entry.key) {
        metadata.set(entry.key, entry.value ?? '');
      }
      offset = end;
    } else {
      offset = skipField(bytes, offset, wire);
    }
  }
  return metadata;
};

const getNameMap = (model, prefix) => {
  const namesArray = new Uint8Array(model.names);
  const nameAdr = model[`name_${prefix}adr`];
  if (!nameAdr) {
    return new Map();
  }
  const map = new Map();
  for (let i = 0; i < nameAdr.length; i++) {
    const startIdx = nameAdr[i];
    if (!Number.isFinite(startIdx)) {
      continue;
    }
    const name = readNameAt(namesArray, startIdx);
    map.set(name, i);
  }
  return map;
};

const rotateVectorWithXmat = (xmat, baseIndex, vec) => {
  const r00 = xmat[baseIndex + 0];
  const r01 = xmat[baseIndex + 1];
  const r02 = xmat[baseIndex + 2];
  const r10 = xmat[baseIndex + 3];
  const r11 = xmat[baseIndex + 4];
  const r12 = xmat[baseIndex + 5];
  const r20 = xmat[baseIndex + 6];
  const r21 = xmat[baseIndex + 7];
  const r22 = xmat[baseIndex + 8];
  const gx = vec[0];
  const gy = vec[1];
  const gz = vec[2];
  return [
    r00 * gx + r10 * gy + r20 * gz,
    r01 * gx + r11 * gy + r21 * gz,
    r02 * gx + r12 * gy + r22 * gz
  ];
};


export class PolicyController {
  constructor(mujoco, config = {}) {
    this.mujoco = mujoco;
    this.modelPath = config.modelPath ?? '../policy.onnx';
    this.depthModelPath = config.depthModelPath ?? '../2026-01-09_08-35-46_student-yaw-random-10-realistic-setting_student.onnx';
    this.controlDt = config.controlDt ?? 0.02;

    this.session = null;
    this.inputName = null;
    this.outputName = null;

    this.depthSession = null;
    this.depthInputName = null;
    this.depthOutputName = null;
    this.depthInputShape = null;
    this.latestDepth = null;
    this.depthFeature = new Float32Array(32);
    this.clippingRange = [0.3, 3.0];
    this.depthResize = { width: 87, height: 58 };
    this.depthCrop = { top: 2, left: 4, right: 4, bottom: 0 };
    this.lastProcessedDepth = null;
    this.lastProcessedDepthSize = { width: this.depthResize.width, height: this.depthResize.height };
    this.depthLatencySteps = 7;
    this.depthLatentQueue = [];

    this.metadata = null;
    this.observationNames = [];
    this.jointNames = [];

    this.actionScale = null;
    this.defaultJointPos = null;
    this.kp = null;
    this.kd = null;

    this.prevActions = null;
    this.latestAction = null;
    this.latestTarget = null;

    this.jointInfo = [];
    this.torsoBodyId = null;
    this.rootJointId = null;
    this.rootQposAdr = 0;
    this.rootDofAdr = 0;
    this.gravityDir = [0, 0, -1];

    this.obsSize = 0;
    this.obsBuffer = null;

    this.joystickState = new Float32Array(15);
    this.pressedKeys = new Set();
    this.highSpeedMode = true;
    this._keyboardBound = false;
    this._debugStep = 0;

    this.isReady = false;
    this.inFlight = false;
  }

  async init(model) {
    this._bindKeyboard();
    await this._initOrt();
    await this._initSession();
    await this._initDepthSession();
    this._readMetadata();
    this._buildMappings(model);
    this.reset();
    this.isReady = true;
  }

  async rebuild(model) {
    this._buildMappings(model);
    this.reset();
  }

  reset() {
    if (!this.jointNames.length) {
      return;
    }
    this.prevActions = new Float32Array(this.jointNames.length);
    this.latestAction = new Float32Array(this.jointNames.length);
    this.latestTarget = new Float32Array(this.jointNames.length);
    this.latestTarget.set(this.defaultJointPos);
    this.joystickState.fill(0);
    this.joystickState[0] = 1;
    this.depthLatentQueue = [];
  }

  _updateCommandState() {
    const arr = new Float32Array(15);

    const isHighSpeed = this.highSpeedMode;
    const isW = this.pressedKeys.has('w');
    const isA = this.pressedKeys.has('a');
    const isD = this.pressedKeys.has('d');
    const isQ = this.pressedKeys.has('q');
    const isE = this.pressedKeys.has('e');

    let commandIdx = 0;
    let baseCmd = 0;
    if ((isW && isA) || isQ) {
        baseCmd = 2;
    } else if ((isW && isD) || isE) {
        baseCmd = 4;
    } else if (isW) {
        baseCmd = 1;
    } else if (isA) {
        baseCmd = 3;
    } else if (isD) {
        baseCmd = 5;
    }

    if (baseCmd !== 0 && isHighSpeed) {
        if (baseCmd === 1) commandIdx = 6;
        else if (baseCmd === 2) commandIdx = 7;
        else if (baseCmd === 3) commandIdx = 8;
        else if (baseCmd === 4) commandIdx = 9;
        else if (baseCmd === 5) commandIdx = 10;
    } else {
        commandIdx = baseCmd;
    }

    if (commandIdx === 0) {
      arr[0] = 1;
    } else {
      arr[commandIdx] = 1;
    }

    this.joystickState = arr;
  }

  _bindKeyboard() {
    if (this._keyboardBound) {
      return;
    }
    this._keyboardBound = true;

    window.addEventListener('keydown', (event) => {
      if ((event.key === 'y' || event.key === 'Y') && !event.repeat) {
        this.highSpeedMode = !this.highSpeedMode;
      }
      if (event.key && event.key.length === 1) {
        this.pressedKeys.add(event.key.toLowerCase());
      }
      this._updateCommandState();
    });

    window.addEventListener('keyup', (event) => {
      if (event.key && event.key.length === 1) {
        this.pressedKeys.delete(event.key.toLowerCase());
      }
      this._updateCommandState();
    });

    window.addEventListener('blur', () => {
      this.pressedKeys.clear();
      this._updateCommandState();
    });

    this._updateCommandState();
  }

  async _initOrt() {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  }

  async _initSession() {
    const modelUrl = new URL(this.modelPath, window.location.href).toString();
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch policy model: ${modelUrl} (${response.status} ${response.statusText})`);
    }
    const arrayBuffer = await response.arrayBuffer();
    this.modelBytes = new Uint8Array(arrayBuffer);

    this.session = await ort.InferenceSession.create(arrayBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    this.inputName = this.session.inputNames[0];
    this.inputNames = this.session.inputNames.slice();
    const preferredOutput = this.session.outputNames.find((name) => name.toLowerCase().includes('action'));
    this.outputName = preferredOutput ?? this.session.outputNames[0];
    console.log('Policy session ready:', {
      inputNames: this.session.inputNames,
      outputNames: this.session.outputNames
    });
  }

  async _initDepthSession() {
    if (!this.depthModelPath) {
      return;
    }
    try {
      const modelUrl = new URL(this.depthModelPath, window.location.href).toString();
      const response = await fetch(modelUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch depth backbone: ${modelUrl} (${response.status} ${response.statusText})`);
      }
      const arrayBuffer = await response.arrayBuffer();
      this.depthSession = await ort.InferenceSession.create(arrayBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      this.depthInputName = this.depthSession.inputNames[0];
      this.depthOutputName = this.depthSession.outputNames[0];
      const inputMeta = this.depthSession.inputMetadata?.[this.depthInputName];
      const shape = inputMeta?.dimensions ?? null;
      if (Array.isArray(shape)) {
        this.depthInputShape = shape.map((dim) => (typeof dim === 'number' && dim > 0 ? dim : null));
      }
      console.log('Depth backbone ready:', {
        inputNames: this.depthSession.inputNames,
        outputNames: this.depthSession.outputNames,
        inputShape: this.depthInputShape
      });
    } catch (error) {
      console.warn('Depth backbone not available, using zeros:', error);
      this.depthSession = null;
    }
  }

  _readMetadata() {
    let meta = this.session?.metadata?.customMetadataMap ?? this.session?.metadata?.customMetadata ?? null;
    let fallbackMeta = null;
    const getMeta = (key) => {
      if (!meta && !fallbackMeta) {
        return undefined;
      }
      if (meta && typeof meta.get === 'function') {
        return meta.get(key);
      }
      if (meta) {
        return meta[key];
      }
      if (fallbackMeta && typeof fallbackMeta.get === 'function') {
        return fallbackMeta.get(key);
      }
      return fallbackMeta?.[key];
    };

    if (!meta || (typeof meta.get === 'function' && meta.size === 0)) {
      console.warn('policy.onnx metadata is unavailable in this runtime. Using ONNX fallback parser.');
      if (this.modelBytes) {
        try {
          fallbackMeta = parseOnnxMetadata(this.modelBytes);
        } catch (error) {
          console.warn('Failed to parse ONNX metadata from model bytes:', error);
        }
      }
    }

    const activeMeta = meta ?? fallbackMeta;
    if (!activeMeta) {
      console.warn('policy.onnx metadata could not be read.');
    } else if (typeof activeMeta.get === 'function') {
      console.log('policy.onnx metadata keys:', Array.from(activeMeta.keys()));
    } else {
      console.log('policy.onnx metadata keys:', Object.keys(activeMeta));
    }

    const jointNames = parseCsv(getMeta('joint_names'));
    if (!jointNames.length) {
      throw new Error('policy.onnx metadata missing joint_names');
    }

    const observationNames = parseCsv(getMeta('observation_names'));
    if (!observationNames.length) {
      throw new Error('policy.onnx metadata missing observation_names');
    }

    const actionScaleRaw = parseNumberCsv(getMeta('action_scale'));
    const defaultJointPosRaw = parseNumberCsv(getMeta('default_joint_pos'));
    const stiffnessRaw = parseNumberCsv(getMeta('joint_stiffness'));
    const dampingRaw = parseNumberCsv(getMeta('joint_damping'));

    this.metadata = {
      jointNames,
      observationNames,
      actionScaleRaw,
      defaultJointPosRaw,
      stiffnessRaw,
      dampingRaw
    };

    this.jointNames = jointNames;
    this.observationNames = observationNames;

    const numActions = this.jointNames.length;
    if (actionScaleRaw.length === 1 && numActions > 1) {
      this.actionScale = ensureLength(new Array(numActions).fill(actionScaleRaw[0]), numActions, 1.0);
    } else {
      this.actionScale = ensureLength(actionScaleRaw, numActions, 1.0);
    }

    // log jointNames here
    console.log('Policy joint names:', this.jointNames);

    this.defaultJointPos = ensureLength(defaultJointPosRaw, numActions, 0.0);
    this.kp = ensureLength(stiffnessRaw, numActions, 0.0);
    this.kd = ensureLength(dampingRaw, numActions, 0.0);

    this.prevActions = new Float32Array(numActions);
    this.latestAction = new Float32Array(numActions);
    this.latestTarget = new Float32Array(numActions);

    this.obsSize = this._computeObsSize();
    this.obsBuffer = new Float32Array(this.obsSize);
  }

  _computeObsSize() {
    let size = 0;
    for (const name of this.observationNames) {
      switch (name) {
        case 'base_lin_vel':
        case 'base_ang_vel':
        case 'projected_gravity':
          size += 3;
          break;
        case 'robot_anchor_projected_gravity':
          size += 3;
          break;
        case 'command':
          size += 3;
          break;
        case 'placeholder':
          size += 15;
          break;
        case 'joint_pos':
        case 'joint_vel':
        case 'actions':
          size += this.jointNames.length;
          break;
        default:
          throw new Error(`Unknown observation name: ${name}`);
      }
    }
    return size;
  }

  _buildMappings(model) {
    const jointNameToId = getNameMap(model, 'jnt');
    const bodyNameToId = getNameMap(model, 'body');
    const rootJointId = model.jnt_type.findIndex(
      (value) => value === this.mujoco.mjtJoint.mjJNT_FREE.value
    );
    this.rootJointId = rootJointId >= 0 ? rootJointId : 0;
    this.rootQposAdr = model.jnt_qposadr[this.rootJointId] ?? 0;
    this.rootDofAdr = model.jnt_dofadr[this.rootJointId] ?? 0;
    this.torsoBodyId = bodyNameToId.get('torso_link');

    const gravity = model.opt?.gravity ?? [0, 0, -9.81];
    this.gravityDir = normalizeVec3([gravity[0], gravity[1], gravity[2]]);

    const actuatorJointIds = new Map();
    if (model.actuator_trnid) {
      for (let i = 0; i < model.nu; i++) {
        const jointId = model.actuator_trnid[i * 2];
        if (Number.isInteger(jointId) && jointId >= 0) {
          if (!actuatorJointIds.has(jointId)) {
            actuatorJointIds.set(jointId, i);
          }
        }
      }
    }

    this.jointInfo = this.jointNames.map((name) => {
      const jointId = jointNameToId.get(name);
      const qposAdr = Number.isInteger(jointId) ? model.jnt_qposadr[jointId] : null;
      const qvelAdr = Number.isInteger(jointId) ? model.jnt_dofadr[jointId] : null;
      const ctrlIndex = Number.isInteger(jointId) ? actuatorJointIds.get(jointId) ?? null : null;
      return { name, jointId, qposAdr, qvelAdr, ctrlIndex };
    });

    const missingJoints = this.jointInfo.filter((info) => !Number.isInteger(info.jointId)).map((info) => info.name);
    if (missingJoints.length > 0) {
      console.warn('Policy joint names missing from model:', missingJoints);
    }

    const missingActuators = this.jointInfo.filter((info) => Number.isInteger(info.jointId) && !Number.isInteger(info.ctrlIndex)).map((info) => info.name);
    if (missingActuators.length > 0) {
      console.warn('Policy joints without actuators:', missingActuators);
    }
  }

  _buildObservation(model, data) {
    const rootPosAdr = this.rootQposAdr;
    const rootQuatRaw = [
      data.qpos[rootPosAdr + 3],
      data.qpos[rootPosAdr + 4],
      data.qpos[rootPosAdr + 5],
      data.qpos[rootPosAdr + 6]
    ];
    const rootQuat = normalizeQuat(rootQuatRaw);

    const rootVelAdr = this.rootDofAdr;
    const baseLinVel = [
      data.qvel[rootVelAdr + 0],
      data.qvel[rootVelAdr + 1],
      data.qvel[rootVelAdr + 2]
    ];
    const baseAngVel = [
      data.qvel[rootVelAdr + 3],
      data.qvel[rootVelAdr + 4],
      data.qvel[rootVelAdr + 5]
    ];

    const projectedGravity = quatApplyInverse(rootQuat, this.gravityDir);

    let torsoProjectedGravity = projectedGravity;
    if (Number.isInteger(this.torsoBodyId)) {
      if (data.xquat) {
        const torsoQuatRaw = [
          data.xquat[(this.torsoBodyId * 4) + 0],
          data.xquat[(this.torsoBodyId * 4) + 1],
          data.xquat[(this.torsoBodyId * 4) + 2],
          data.xquat[(this.torsoBodyId * 4) + 3]
        ];
        const torsoQuat = normalizeQuat(torsoQuatRaw);
        torsoProjectedGravity = quatApplyInverse(torsoQuat, this.gravityDir);
      } 
    }


    let offset = 0;
    for (const name of this.observationNames) {
      switch (name) {
        case 'base_lin_vel':
          this.obsBuffer.set(baseLinVel, offset);
          offset += 3;
          break;
        case 'base_ang_vel':
          this.obsBuffer.set(baseAngVel, offset);
          offset += 3;
          break;
        case 'projected_gravity':
          this.obsBuffer.set(projectedGravity, offset);
          offset += 3;
          break;
        case 'robot_anchor_projected_gravity': 
          this.obsBuffer.set(torsoProjectedGravity, offset);
          offset += 3;
          break;
        case 'placeholder':
          this.obsBuffer.set(this.joystickState, offset);
          offset += 15;
          break;
        case 'command':
          this.obsBuffer.set(DEFAULT_COMMAND, offset);
          offset += 3;
          break;
        case 'joint_pos': {
          for (let i = 0; i < this.jointInfo.length; i++) {
            const info = this.jointInfo[i];
            const pos = Number.isInteger(info.qposAdr) ? data.qpos[info.qposAdr] : 0.0;
            this.obsBuffer[offset + i] = pos - this.defaultJointPos[i];
          }
          offset += this.jointInfo.length;
          break;
        }
        case 'joint_vel': {
          for (let i = 0; i < this.jointInfo.length; i++) {
            const info = this.jointInfo[i];
            const vel = Number.isInteger(info.qvelAdr) ? data.qvel[info.qvelAdr] : 0.0;
            this.obsBuffer[offset + i] = vel;
          }
          offset += this.jointInfo.length;
          break;
        }
        case 'actions':
          this.obsBuffer.set(this.prevActions, offset);
          offset += this.prevActions.length;
          break;
        default:
          throw new Error(`Unknown observation name: ${name}`);
      }
    }

    return this.obsBuffer;
  }

  setDepthImage(depthData, width, height) {
    if (!depthData || !Number.isFinite(width) || !Number.isFinite(height)) {
      this.latestDepth = null;
      return;
    }
    this.latestDepth = {
      data: depthData,
      width,
      height
    };
  }

  _resizeBilinear(input, inW, inH, outW, outH) {
    const output = new Float32Array(outW * outH);
    if (inW === outW && inH === outH) {
      output.set(input);
      return output;
    }
    const scaleX = inW / outW;
    const scaleY = inH / outH;
    for (let y = 0; y < outH; y++) {
      const srcY = (y + 0.5) * scaleY - 0.5;
      const y0 = Math.max(0, Math.min(inH - 1, Math.floor(srcY)));
      const y1 = Math.min(inH - 1, y0 + 1);
      const wy = srcY - y0;
      for (let x = 0; x < outW; x++) {
        const srcX = (x + 0.5) * scaleX - 0.5;
        const x0 = Math.max(0, Math.min(inW - 1, Math.floor(srcX)));
        const x1 = Math.min(inW - 1, x0 + 1);
        const wx = srcX - x0;
        const v00 = input[y0 * inW + x0];
        const v10 = input[y0 * inW + x1];
        const v01 = input[y1 * inW + x0];
        const v11 = input[y1 * inW + x1];
        const v0 = v00 * (1 - wx) + v10 * wx;
        const v1 = v01 * (1 - wx) + v11 * wx;
        output[y * outW + x] = v0 * (1 - wy) + v1 * wy;
      }
    }
    return output;
  }

  _prepareDepthInput() {
    const { data, width, height } = this.latestDepth;
    const cropTop = this.depthCrop.top;
    const cropLeft = this.depthCrop.left;
    const cropRight = this.depthCrop.right;
    const cropBottom = this.depthCrop.bottom;
    const croppedW = Math.max(0, width - cropLeft - cropRight);
    const croppedH = Math.max(0, height - cropTop - cropBottom);
    const cropped = new Float32Array(croppedW * croppedH);
    const minDepth = this.clippingRange[0];
    const maxDepth = this.clippingRange[1];
    const pixelCount = croppedW * croppedH;
    for (let i = 0; i < pixelCount; i++) {
      const y = Math.floor(i / croppedW);
      const x = i - y * croppedW;
      const srcY = y + cropTop;
      const srcX = x + cropLeft;
      cropped[i] = data[srcY * width + srcX];
    }

    const resized = this._resizeBilinear(
      cropped,
      croppedW,
      croppedH,
      this.depthResize.width,
      this.depthResize.height
    );

    const clipped = new Float32Array(resized.length);
    const range = maxDepth - minDepth;

    for (let i = 0; i < clipped.length; i++) {
      clipped[i] = (resized[i] - minDepth) / range - 0.5;
    }
    this.lastProcessedDepth = clipped;
    this.lastProcessedDepthSize = { width: this.depthResize.width, height: this.depthResize.height };
    return clipped;
  }

  getProcessedDepthPreview() {
    if (!this.lastProcessedDepth) {
      return null;
    }
    return {
      data: this.lastProcessedDepth,
      width: this.lastProcessedDepthSize.width,
      height: this.lastProcessedDepthSize.height
    };
  }

  async _runDepthBackbone() {
    if (!this.depthSession) {
      return null;
    }
    const depthInput = this._prepareDepthInput();

    const shape = [1, this.depthResize.width, this.depthResize.height];
    const W = shape[1];
    const H = shape[2];

    const flippedDepth = new Float32Array(depthInput.length);
    for (let y = 0; y < H; y++) {
      const srcRow = (H - 1 - y) * W;
      const dstRow = y * W;
      for (let x = 0; x < W; x++) {
        flippedDepth[dstRow + x] = depthInput[srcRow + x];
      }
    }
    const inputTensor = new ort.Tensor('float32', flippedDepth, [1, H, W]);

    const feeds = { [this.depthInputName]: inputTensor };
    const output = await this.depthSession.run(feeds);
    const outputTensor = output[this.depthOutputName];
    if (!outputTensor || !outputTensor.data) {
      return null;
    }
    if (outputTensor.data.length !== 32) {
      console.warn('Depth backbone output length mismatch:', outputTensor.data.length);
      return null;
    }
    return outputTensor.data;
  }

  async requestAction(model, data) {
    if (!this.isReady || this.inFlight) {
      return;
    }
    this.inFlight = true;
    try {
      const obs = this._buildObservation(model, data);

      let depthFeature = null;
      if (this.depthSession) {
        try {
          depthFeature = await this._runDepthBackbone();
        } catch (error) {
          console.warn('Depth backbone inference failed:', error);
        }
      }

      if (depthFeature && depthFeature.length === 32) {
        this.depthLatentQueue.push(Float32Array.from(depthFeature));
      } else {
        this.depthLatentQueue.push(null);
      }

      let delayedDepthFeature = null;
      if (this.depthLatentQueue.length > this.depthLatencySteps) {
        delayedDepthFeature = this.depthLatentQueue.shift();
      } else {
        delayedDepthFeature = this.depthLatentQueue[0];
      }

      const obsWithZeros = new Float32Array(obs.length + 32);
      obsWithZeros.set(obs, 0);
      obsWithZeros.set(delayedDepthFeature, obs.length);
      
      // set zeros
    //   obsWithZeros.fill(0, obs.length);

      const inputTensor = new ort.Tensor('float32', obsWithZeros, [1, obsWithZeros.length]);
      const feeds = { [this.inputName]: inputTensor };
      if (this.inputNames && this.inputNames.includes('time_step')) {
        feeds['time_step'] = new ort.Tensor('float32', new Float32Array([[0]]), [1, 1]);
      }

      const output = await this.session.run(feeds);
      const actionTensor = output[this.outputName];
      if (!actionTensor || !actionTensor.data) {
        throw new Error('Policy output missing action tensor');
      }
      if (actionTensor.data.length !== this.jointNames.length) {
        throw new Error(`Action length ${actionTensor.data.length} does not match joint count ${this.jointNames.length}`);
      }

      this.latestAction.set(actionTensor.data);
      this.prevActions.set(actionTensor.data);

      for (let i = 0; i < this.latestTarget.length; i++) {
        this.latestTarget[i] = this.defaultJointPos[i] + this.actionScale[i] * this.latestAction[i];
      }
    } finally {
      this.inFlight = false;
    }
  }

  applyControl(model, data) {
    if (!this.isReady || !this.latestTarget) {
      return;
    }

    const ctrlRange = model.actuator_ctrlrange;

    for (let i = 0; i < this.jointInfo.length; i++) {
      const info = this.jointInfo[i];
      if (!Number.isInteger(info.ctrlIndex)) {
        continue;
      }
      const qpos = Number.isInteger(info.qposAdr) ? data.qpos[info.qposAdr] : 0.0;
      const qvel = Number.isInteger(info.qvelAdr) ? data.qvel[info.qvelAdr] : 0.0;
      const target = this.latestTarget[i];
      const torque = this.kp[i] * (target - qpos) + this.kd[i] * (0.0 - qvel);

      let ctrlValue = torque;
      if (ctrlRange && ctrlRange.length >= (info.ctrlIndex + 1) * 2) {
        const min = ctrlRange[info.ctrlIndex * 2];
        const max = ctrlRange[(info.ctrlIndex * 2) + 1];
        if (Number.isFinite(min) && Number.isFinite(max) && min < max) {
          ctrlValue = Math.min(Math.max(ctrlValue, min), max);
        }
      }

      data.ctrl[info.ctrlIndex] = ctrlValue;
    }
  }
}
