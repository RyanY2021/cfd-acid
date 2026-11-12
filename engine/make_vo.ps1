# Voice-over lines for engine/audio.py -> cache/vo/vo00.wav .. vo12.wav (Windows TTS, voice "Microsoft Hazel Desktop").
#   powershell -ExecutionPolicy Bypass -File engine/make_vo.ps1
Add-Type -AssemblyName System.Speech
$dir = Join-Path $PSScriptRoot "..\cache\vo"
New-Item -ItemType Directory -Force $dir | Out-Null
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$lines = @(
    @(-1, "Every wave is a solution. To an equation, no one has solved."),
    @(0, "Initiating flow solver."),
    @(0, "Computational. Fluid. Dynamics."),
    @(0, "Assert. A fluid is a continuum."),
    @(0, "Assert. These equations govern every flow."),
    @(0, "Warning. No general solution exists."),
    @(0, "So, we compute."),
    @(0, "Discretizing domain."),
    @(0, "Residual converged."),
    @(0, "Reynolds number. Inertia, over viscosity."),
    @(0, "Applications."),
    @(0, "Simulate the"),
    @(0, "Simulation. Exhausted.")        # generated but no longer used: the end card is not read aloud
)
for ($i = 0; $i -lt $lines.Count; $i++) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice("Microsoft Hazel Desktop")
    $s.Rate = $lines[$i][0]
    $s.SetOutputToWaveFile((Join-Path $dir ("vo{0:D2}.wav" -f $i)), $fmt)
    $s.Speak($lines[$i][1])
    $s.Dispose()
}
"voice lines -> $dir"
