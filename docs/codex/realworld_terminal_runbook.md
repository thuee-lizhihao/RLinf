# RLinf 终端操作默认约定（本次对话沉淀）

本文档用于沉淀本次对话中出现过、并被证明是“每次需要/建议先做”的终端操作约定，便于后续复用。

## 基础环境（建议每次跑测试/脚本前先做）

```bash
cd /home/franka/cynws/RLinf
source /home/franka/cynws/RLinf/.venv/bin/activate
export PYTHONPATH=$PWD
```

## 需要联网时（先设置代理）

```bash
export http_proxy="http://192.168.120.133:7890"
export https_proxy="http://192.168.120.133:7890"
export all_proxy="socks5h://192.168.120.133:7890"
```

## GELLO 串口权限（临时方案）

GELLO 设备端口（本次对话）：

- `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0` 

另外还有franka夹爪端口

- `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAK2KK44-if00-port0`

临时赋权（需要 sudo，且会提示输入密码）：

```bash
sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAK2KK44-if00-port0
```

备注：更“持久”的做法通常是把用户加入 `dialout` 组并重新登录（见 RLinf 文档 `franka_gello` 页面）。

## 读取 TCP 位姿：用 RLinf 的脚本

`gello-teleop` 包通常不包含 `python -m gello_teleop.gello_expert` 这种入口模块；读取 TCP 位姿请用 RLinf 自带脚本：

```bash
python -m rlinf.envs.realworld.common.gello.gello_expert \
  --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
```

## 端口不在 `PORT_CONFIG_MAP`：需要校准生成配置

如果出现错误：

`AssertionError: Port ... not in config map`

说明 `third_party/gello_software/gello/agents/gello_agent.py` 里的 `PORT_CONFIG_MAP` 没有你当前串口（例如 `FTA0OUKN`）的条目。

按下面流程生成正确的 `joint_signs/joint_offsets/gripper_config` 并粘贴进去：

1. 先把环境变量指向你的 GELLO 端口：

```bash
export GELLO_PORT=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
export GELLO_BAUDRATE=57600
export FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAK2KK44-if00-port0
```

2. 运行校准脚本（会移动 Franka 并要求你在两处手动对齐 GELLO）：

```bash
bash examples/embodiment/gello_calibrate.sh
```

## Franka 连不上：FCI/libfranka 版本不匹配（server=9, library=10）

如果在运行 `gello_calibrate.sh` / `test_franky_controller.py` 时看到类似报错：

`franky._franky.IncompatibleVersionException: libfranka: Incompatible library version (server version: 9, library version: 10)`

这是 **机器人端 FCI/server 版本** 与你当前 Python 环境里加载到的 **libfranka（由 `franky-control` wheel 自带）** 不兼容导致的。

本机连接 `172.16.0.2`（server=9）时，一个可用的最短修复是把 `franky-control` 降级到带兼容 libfranka 的版本：

```bash
source /home/franka/cynws/RLinf/.venv/bin/activate
export http_proxy="http://192.168.120.133:7890"
export https_proxy="http://192.168.120.133:7890"
export all_proxy="socks5h://192.168.120.133:7890"
python -m pip install -U "franky-control==1.0.2"
```

快速自检（不会移动机器人，只读取 state）：

```bash
python - <<'PY'
import franky
r=franky.Robot("172.16.0.2")
print("connected, q0=", r.state.q[0])
PY
```

3. 校准结束后，将脚本打印的 `DynamixelRobotConfig(...)` 片段粘贴到：

`/home/franka/cynws/third_party/gello_software/gello/agents/gello_agent.py`

4. 重新运行 `gello_expert` / `gello_joint_expert`。

  FRANKA_ROBOT_IP     = 172.16.0.2
  FRANKA_GRIPPER_PORT = /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0
  GELLO_PORT          = /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
  GELLO_BAUDRATE      = 1000000
