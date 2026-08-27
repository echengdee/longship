# Hiking G1 实机部署

这条链路使用 Longship 统一实机 runner；Hiking 只注册一份 profile 和 backend，
不维护模型专属的实机启动器：

```text
Sim2Sim: MuJoCo LowState + torso IMU + depth ─┐
                                               ├─ instinctlab_dds.py ─ LowCmd
Real:    G1 LowState + torso IMU + D435i ─────┘
```

统一 runner 负责 profile 加载、传感器启动、控制器就绪、Unitree motion mode、日志、
键盘和退出清理。backend adapter 负责模型差异，sensor adapter 负责相机差异，target
adapter 负责机器人差异。后续 HoloSoma/SONIC 只需注册 adapter 与 YAML，不复制 runner。

两边共享：Stand/Parkour ONNX、观测顺序、8 帧本体历史、37 帧深度缓存与抽帧、深度
裁剪/归一化、关节映射、PD 参数、模式切换以及 ZMQ 键盘协议。实机只替换 DDS 网卡、
物理 LowState/LowCmd 和深度生产者；实机 LowCmd 另外写入 G1 `mode_pr`、
`mode_machine` 和 CRC，策略按 50 Hz 墙钟运行。

## 1. 配置

默认配置在：

```text
src/longship/rl/deploy/profiles/hiking_g1.yaml
```

确认或覆盖：

- `dds.interface`：连接 G1 的非回环网卡；
- `dds.domain_id`：机器人 DDS domain；
- `camera.serial`：实际 D435i 序列号；
- 相机物理俯仰：调整支架以得到 Sim2Sim 默认视野，不在策略里增加私有角度补偿。

创建统一环境后安装实机依赖：

```bash
conda env update -n longship-rl -f environment.yml --prune
```

也可以为已有环境安装项目的 `rl-deploy` extra。上线前应能导入：

```bash
python -c 'import cv2, cyclonedds, onnxruntime, pyrealsense2, unitree_sdk2py, zmq'
```

## 2. 无动作预检

下面的命令检查模型、依赖、网卡和配置，只打印进程命令，不释放机器人 motion mode，
也不会发布 LowCmd：

```bash
./scripts/deploy/run_real.sh hiking_g1 --print-command
```

临时覆盖现场参数：

```bash
./scripts/deploy/run_real.sh hiking_g1 --print-command \
  --interface enp5s0 \
  --domain-id 0 \
  --camera-serial 346522071778
```

不要同时运行 RealSense Viewer、另一套 Hiking 部署或其他 LowCmd 控制器。

## 3. 正式启动

机器人上电、额定龙门架、遥控器与急停确认完成，且人员离开跌倒区域后：

```bash
export REAL_ROBOT_ENABLED=1
export REAL_ROBOT_CONFIRM=I_UNDERSTAND_THE_RISK
export GANTRY_CONFIRMED=1
export ESTOP_CONFIRMED=1
export REMOTE_CONFIRMED=1
export ROBOT_MODE_CONFIRMED=1
export FALL_ZONE_CLEAR_CONFIRMED=1

./scripts/deploy/run_real.sh hiking_g1
```

启动器先等待真实 D435i 深度和 G1 `LowState + secondary_imu` 到齐，再释放 Unitree
高层 motion mode。此时控制器仍是 `IDLE`，需要操作者在同一终端明确执行：

```text
i  -> 等待初始化完成
]  -> 启用 Stand Actor
2  -> 切 Parkour Actor
w  -> 前进
1  -> 清零速度并切回 Stand Actor
```

首次实机只做龙门架下 Stand 和零速度 `Stand ⇄ Parkour`。楼梯测试应在平地和低矮
障碍分别完成后再进行。当前 Parkour 台阶首触问题尚未调优，不应直接把现有仿真结果
视为楼梯实机放行依据。

`scripts/deploy/run_hiking_g1.sh` 仅是上述统一命令的便捷包装，不包含部署逻辑。
运行日志位于 `outputs/deploy/hiking_g1/<时间>/depth_camera.log` 和 `controller.log`。

## 4. 远程视觉监控

统一 deploy runner 会同时启动本机网页服务，默认地址为 `127.0.0.1:8080`。页面包含：

- `camera_depth`：D435i 在 DDS 发布前的 480×270 米制深度图；
- `model_depth`：当前 Hiking Actor 实际接收的最新 18×32 归一化深度输入。

Stand Actor 按上游实现接收全零深度，因此 Stand 模式下模型画面为固定全零；切换到
Parkour 后才显示实时预处理深度。这是实际输入，不是额外模拟出来的预览。

从开发机建立 SSH 端口转发：

```bash
ssh -N -L 8080:127.0.0.1:8080 <user>@<robot-ip>
```

然后在开发机浏览器打开：

```text
http://127.0.0.1:8080
```

Codex/VS Code Remote 也可以直接转发实机的 `8080` 端口。服务默认不监听机器人局域网，
因为当前调试页面没有登录认证。确实需要局域网直接访问时，可以显式运行：

```bash
./scripts/deploy/run_real.sh hiking_g1 --visualization-bind-host 0.0.0.0
```

端口被占用时可通过 `--visualization-port 18080` 修改；不需要页面时使用
`--no-visualization`。调试帧以 10 Hz 非阻塞发送，浏览器断开不会阻塞策略控制循环。
