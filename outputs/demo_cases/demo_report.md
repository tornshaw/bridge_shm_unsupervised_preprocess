# 多桥测试 Demo（双层评分）

## 说明
- DeviceHealth 低 / Availability 高：偏设备侧问题
- DeviceHealth 高 / Availability 低：偏系统侧问题
- 两者都低：设备与系统可能同时有问题

## 沥心沙大桥_20251201_20260318
- samples=11260, sensors=70
- DeviceHealth=52.767, Availability=82.97, ProjectScore=61.828
- missing(total/system/device)=0.2649/0.1617/0.1032
- 事件摘要: known_system_gap_hours=15594.33, device_gap_hours=13562.5, stuck_events=33410, drift_events=62030, step_events=30417, spike_events=11022
- 最差3个设备: LXS-0-2L-YB-1|frequency(D=0.0,A=82.9,P=24.9), LXS-0-2L-YB-1|strain(D=0.0,A=82.9,P=24.9), LXS-1-3D-BB-2|deflection(D=0.0,A=82.9,P=24.9)

## 浅海大桥_20251201_20260318
- samples=12266, sensors=22
- DeviceHealth=67.149, Availability=76.6, ProjectScore=69.984
- missing(total/system/device)=0.2761/0.213/0.0631
- 事件摘要: known_system_gap_hours=4859.0, device_gap_hours=2837.5, stuck_events=113, drift_events=18125, step_events=7474, spike_events=4791
- 最差3个设备: QH-0-1L-LX-4|measureDisplacement(D=0.0,A=76.6,P=23.0), QH-0-1L-LX-1|measureDisplacement(D=3.9,A=76.6,P=25.7), QH-0-1L-YB-2|frequency(D=24.5,A=76.6,P=40.1)

## 新榄核大桥_20251201_20260318
- samples=11288, sensors=31
- DeviceHealth=81.309, Availability=83.092, ProjectScore=81.844
- missing(total/system/device)=0.1929/0.16/0.0329
- 事件摘要: known_system_gap_hours=6671.0, device_gap_hours=1920.17, stuck_events=65, drift_events=15203, step_events=3640, spike_events=3880
- 最差3个设备: XLH-2-2L-LX-2|measureDisplacement(D=13.8,A=83.1,P=34.6), XLH-2-2L-LX-1|measureDisplacement(D=29.8,A=83.1,P=45.8), XLH-1-1L-LX-2|measureDisplacement(D=31.1,A=83.1,P=46.7)
