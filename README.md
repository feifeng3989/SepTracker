# SepTracker: Arbitrary One-to-Many Separation-Aware Infrared Small Target Tracking

# Abstract
Rockets, satellites, and their debris operating in deep-space scenarios often exhibit arbitrary One-to-Many separation behaviors, where the number of generated child objects is unknown and varies across events. Most existing multi-object tracking (MOT) methods assume that trajectories do not split, and therefore fail to characterize arbitrary One-to-Many separation events for infrared (IR) small targets in deep space. To this end, we present SepTracker, the first separation-aware MOT framework tailored to IR small targets under arbitrary One-to-Many separations. Meanwhile, we construct SepTrack, the first IR small-target dataset dedicated to arbitrary One-to-Many separation behaviors, as well as a new evaluation metric, i.e., Separation Determination Probability Accuracy. To address the uncertainty in separation scale, we design a Dynamic Adaptive Grouping Head, which adaptively aggregates same-parent child-target clusters based on inter-target similarity, thereby enabling variable-cardinality grouping and separation-event modeling. Furthermore, we design Dual-space Hypergraph Attention mechanism to model local high-order relations among same-parent child targets within the current batch in a node-centric space, and introduce a learnable prototype library in a prototype-centric space to impose cross-batch global semantic constraints for homologous child trajectories. This design yields trajectory representations that jointly capture both local structural awareness and global semantic alignment. Experimental results demonstrate that SepTracker substantially outperforms existing methods in both separation determination and tracking performance under arbitrary One-to-Many separation scenarios.

# Overview
<img width="1612" height="621" alt="image" src="https://github.com/user-attachments/assets/c094c834-a0c6-4465-b4c3-c1f97e559ffb" />

# Envs. for Project
Python 3.13.15
torch 1.12.1+cu116
Requirements: requirements.txt

