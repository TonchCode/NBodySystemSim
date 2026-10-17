# solomon2014pcie16gts (INDUSTRY TECHNICAL ARTICLE)

**Citation:** Solomon, R. "PCI Express Controller Design Challenges at 16 GT/s". Synopsys technical article, 21 July 2014.
**Link:** https://www.synopsys.com/articles/pci-design-challenges.html
**Verified on:** 2026-10-17 (page fetched; title, author and date confirmed).
**Lead-group country:** USA (IP vendor).

## Key facts
- "The same 128/130 encoding scheme introduced with PCIe 3.0 and 8 GT/s will be used in PCIe 4.0's 16 GT/s mode."
- It recalls PCI-SIG's 29 Nov 2011 announcement of the 16 GT/s rate, which agrees with pcisig2011pcie4.

## Caveats [T2]
Written in 2014, three years before the final PCIe 4.0 spec (Oct 2017), and phrased in the future tense ("will be used"). It is a vendor article, not the specification. Use it for the encoding only, labelled as secondary. If the course demands primary sources, the PCI-SIG FAQ or specification has to be fetched by hand (pcisig.com blocks automated fetches).

## Summary
1. A design article by a PCIe controller IP vendor.
2. It states that PCIe 4.0 keeps PCIe 3.0's 128b/130b encoding.
3. It discusses the controller datapath widths needed at 16 GT/s.
4. It predates the final spec.
5. It supplies the encoding factor for our host-bandwidth calculation.

## What it lets us claim
128b/130b encoding at 16 GT/s, so about 1.5 % line overhead and ≈ 31.5 GB/s per direction for ×16 [D].

## Quote
"The same 128/130 encoding scheme ... will be used in PCIe 4.0's 16 GT/s mode." (Solomon, 2014)
