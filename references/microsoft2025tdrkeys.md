# microsoft2025tdrkeys (MICROSOFT DOC)

**Citation:** Microsoft. *Testing and Debugging TDR During Driver Development*. Microsoft Learn, Windows driver documentation (display). ms.date 2025-11-05.
**Link:** https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys (section "TdrDelay")
**Verified on:** 2026-10-17 (page fetched; title and date from the page metadata).
**Lead-group country:** USA (vendor).

## Key facts
- **TdrDelay:** under `HKLM\System\CurrentControlSet\Control\GraphicsDrivers`, REG_DWORD, "The default value is 2 seconds." It is the number of seconds the GPU may delay the scheduler's preempt request, "effectively the timeout threshold."
- TdrLevel default: 3, TdrLevelRecover ("Recover on timeout"). TdrDdiDelay default: 5 s. TdrLimitTime default: 60 s, TdrLimitCount default: 5.
- The page warns that end users "shouldn't manipulate these registry keys" outside driver-development testing. This backs 01-hardware's rule not to raise TdrDelay casually.

## Summary
1. This is Microsoft's own documentation of the TDR registry keys.
2. It fixes the 2 s default timeout that bounds our kernel launch length under WDDM.
3. It defines the timeout as a preemption delay, not as total kernel runtime.
4. It lists the recovery and crash thresholds.
5. It discourages changing the keys outside driver development.

## What it lets us claim
"Under WDDM, a GPU that does not yield to preemption within 2 s (default TdrDelay) triggers TDR" [N:microsoft2025tdrkeys]. This justifies the one-timestep-per-launch rule (CLAUDE.md rule 4).

## Design decision / R3 technique supported
Kernel-launch granularity; persistent-thread designs (R3 #6) must stay under the TDR limit on this display-attached GPU.

## Quote
"TdrDelay is effectively the timeout threshold." (Microsoft Learn)
