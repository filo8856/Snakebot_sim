param(
    [ValidateRange(-1, 3)][int]$ControllerIndex = -1,
    [ValidateRange(5, 1000)][int]$IntervalMs = 20,
    [ValidateRange(0, 100000)][int]$MaxSamples = 0
)

$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Snakebot {
    public static class GamepadReader {
        [StructLayout(LayoutKind.Sequential)]
        public struct Gamepad {
            public ushort buttons;
            public byte leftTrigger, rightTrigger;
            public short leftX, leftY, rightX, rightY;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct State {
            public uint packet;
            public Gamepad pad;
        }
        [DllImport("xinput1_4.dll", EntryPoint = "XInputGetState")]
        private static extern uint GetState(uint index, out State state);

        public static void Run(int requested, int intervalMs, int maxSamples) {
            // EOF or a stop line terminates the helper when the Python parent
            // closes. Read on another thread so polling never blocks on stdin.
            Task<string> stop = Task.Run(() => Console.ReadLine());
            int count = 0;
            while ((maxSamples > 0 || !stop.IsCompleted) &&
                   (maxSamples == 0 || count < maxSamples)) {
                State state = new State();
                int selected = -1;
                if (requested >= 0) {
                    if (GetState((uint)requested, out state) == 0)
                        selected = requested;
                } else {
                    for (uint i = 0; i < 4; i++) {
                        if (GetState(i, out state) == 0) {
                            selected = (int)i;
                            break;
                        }
                    }
                }
                bool connected = selected >= 0;
                Console.WriteLine(
                    "{\"connected\":" + (connected ? "true" : "false") +
                    ",\"index\":" + selected +
                    ",\"rx\":" + (connected ? state.pad.rightX : 0) +
                    ",\"ry\":" + (connected ? state.pad.rightY : 0) + "}");
                Console.Out.Flush();
                count++;
                Thread.Sleep(intervalMs);
            }
        }
    }
}
'@

[Snakebot.GamepadReader]::Run($ControllerIndex, $IntervalMs, $MaxSamples)
