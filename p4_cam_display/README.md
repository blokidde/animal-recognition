# Esp32 p4 camera with lcd screen

## To Do
- fix error: 
--- 0x4ff00003: _vector_table at ??:?
MHARTID : 0x00000000
Please enable CONFIG_ESP_SYSTEM_USE_FRAME_POINTER option to have a full backtrace.
E (54278) task_wdt: Task watchdog got triggered. The following tasks/users did not reset the watchdog in time:
E (54278) task_wdt:  - IDLE0 (CPU 0)
E (54278) task_wdt: Tasks currently running:
E (54278) task_wdt: CPU 0: main
E (54278) task_wdt: CPU 1: IDLE1
E (54278) task_wdt: Print CPU 0 (current core) registers
Core  0 register dump:
MEPC    : 0x4fc15476  RA      : 0x4000dbee  SP      : 0x4ff1b740  GP      : 0x4ff15c00
--- 0x4fc15476: __umoddi3 in ROM
--- 0x4000dbee: app_main at C:/Users/jurriaan/animal-recognition/p4_cam_display/main/cam_display.cpp:324
TP      : 0x4ff1ba20  T0      : 0x4fc10cc4  T1      : 0x50000000  T2      : 0xc220a120
--- 0x4fc10cc4: Cache_Start_L2_Cache_Preload in ROM
S0/FP   : 0x00000005  S1      : 0x00000088  A0      : 0x000001c5  A1      : 0x00000000
A2      : 0x00000000  A3      : 0x000001c5  A4      : 0x00000000  A5      : 0x50000000
A6      : 0xf0000000  A7      : 0x000000f0  S2      : 0x50424752  S3      : 0x00000320
S4      : 0x00000320  S5      : 0x4ff2cc80  S6      : 0x00001860  S7      : 0x00000000
S8      : 0x00070800  S9      : 0x48139500  S10     : 0x4821a500  S11     : 0x0000000d
T3      : 0x00000000  T4      : 0x00000000  T5      : 0x00000001  T6      : 0x4ff3c010
MSTATUS : 0x00001888  MTVEC   : 0x4ff00003  MCAUSE  : 0xdeadc0de  MTVAL   : 0xdeadc0de

- add first AI/computer vision 
- design a case for everything
- get the program to work with pi cam 1.3 (without lens), now when its used its bugged
- get faster reaction time on camera